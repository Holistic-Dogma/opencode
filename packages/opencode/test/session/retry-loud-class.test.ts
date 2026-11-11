import { describe, expect, test } from "bun:test"
import { APICallError, RetryError } from "ai"
import { Effect, Layer, Stream } from "effect"
import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { Database } from "@opencode-ai/core/database/database"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Provider } from "@/provider/provider"
import { ProviderError } from "../../src/provider/error"
import { Session } from "@/session/session"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionProcessor } from "../../src/session/processor"
import { SessionRetry } from "../../src/session/retry"
import { MessageID, PartID, SessionID } from "@/session/schema"
import { SessionStatus } from "@/session/status"
import { SessionSummary } from "@/session/summary"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import { provideTmpdirInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { SessionProjector } from "@opencode-ai/core/session/projector"
import { LLMEvent } from "@opencode-ai/llm"
import path from "path"
import type { Agent } from "../../src/agent/agent"
import { SessionV1 } from "@opencode-ai/core/v1/session"

const summary = Layer.succeed(
  SessionSummary.Service,
  SessionSummary.Service.of({
    summarize: () => Effect.void,
    diff: () => Effect.succeed([]),
    computeDiff: () => Effect.succeed([]),
  }),
)

const ref = {
  providerID: ProviderV2.ID.make("test"),
  modelID: ModelV2.ID.make("test-model"),
}

const cfg = {
  provider: {
    test: {
      name: "Test",
      id: "test",
      env: [],
      npm: "@ai-sdk/openai-compatible",
      models: {
        "test-model": {
          id: "test-model",
          name: "Test Model",
          attachment: false,
          reasoning: false,
          temperature: false,
          tool_call: true,
          release_date: "2025-01-01",
          limit: { context: 100000, output: 10000 },
          cost: { input: 0, output: 0 },
          options: {},
        },
      },
      options: {
        apiKey: "test-key",
        baseURL: "http://localhost:1/v1",
      },
    },
  },
}

function agent(): Agent.Info {
  return {
    name: "build",
    mode: "primary",
    options: {},
    permission: [{ permission: "*", pattern: "*", action: "allow" }],
  }
}

const root = LayerNode.group([
  SessionProcessor.node,
  Session.node,
  SessionProjector.node,
  Provider.node,
  Database.node,
  EventV2Bridge.node,
  SessionStatus.node,
  CrossSpawnSpawner.node,
])

const replacements = [
  [SessionSummary.node, summary],
  [RuntimeFlags.node, RuntimeFlags.layer({ experimentalEventSystem: true })],
] as const

// The loud class: an explicit APIError raised mid-flight by a remote provider
// when its socket drops ("Cannot connect to API: socket connection closed
// unexpectedly"). This is the charlie-watch death class.
const LOUD = "Cannot connect to API: socket connection closed unexpectedly"

function loudApiError(): APICallError {
  return new APICallError({
    message: LOUD,
    url: "https://api.vendor.test/v1/chat",
    requestBodyValues: {},
    isRetryable: true,
    cause: new Error("socket connection closed unexpectedly"),
  })
}

// A provider-level rejection that must NOT be re-issued: the credential is
// bad, and re-issuing it only burns budget.
function nonRetryableApiError(): APICallError {
  return new APICallError({
    message: "invalid api key",
    url: "https://api.vendor.test/v1/chat",
    requestBodyValues: {},
    isRetryable: false,
    statusCode: 401,
  })
}

let loudCalls = 0
const loudLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () => {
      loudCalls++
      if (loudCalls === 1) return Stream.fail(loudApiError())
      return Stream.make(
        LLMEvent.stepStart({ index: 0 }),
        LLMEvent.textStart({ id: "text-1" }),
        LLMEvent.textDelta({ id: "text-1", text: "recovered" }),
        LLMEvent.textEnd({ id: "text-1" }),
        LLMEvent.stepFinish({ index: 0, reason: "stop" }),
        LLMEvent.finish({ reason: "stop" }),
      )
    },
  }),
)
const itLoud = testEffect(LayerNode.compile(root, [...replacements, [LLM.node, loudLLM]]))

let nonRetryableCalls = 0
const nonRetryableLLM = Layer.succeed(
  LLM.Service,
  LLM.Service.of({
    stream: () => {
      nonRetryableCalls++
      return Stream.fail(nonRetryableApiError())
    },
  }),
)
const itNonRetryable = testEffect(
  LayerNode.compile(root, [...replacements, [LLM.node, nonRetryableLLM]]),
)

const user = Effect.fn("TestSession.user")(function* (sessionID: SessionID, text: string) {
  const session = yield* Session.Service
  const msg = yield* session.updateMessage({
    id: MessageID.ascending(),
    role: "user",
    sessionID,
    agent: "build",
    model: ref,
    time: { created: Date.now() },
  })
  yield* session.updatePart({
    id: PartID.ascending(),
    messageID: msg.id,
    sessionID,
    type: "text",
    text,
  })
  return msg
})

const assistant = Effect.fn("TestSession.assistant")(function* (
  sessionID: SessionID,
  parentID: MessageID,
  rootPath: string,
) {
  const session = yield* Session.Service
  const msg: SessionV1.Assistant = {
    id: MessageID.ascending(),
    role: "assistant",
    sessionID,
    mode: "build",
    agent: "build",
    path: { cwd: rootPath, root: rootPath },
    cost: 0,
    tokens: { total: 0, input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
    modelID: ref.modelID,
    providerID: ref.providerID,
    parentID,
    time: { created: Date.now() },
    finish: "end_turn",
  }
  yield* session.updateMessage(msg)
  return msg
})

const process = Effect.fn("TestSession.process")(function* (dir: string, text: string) {
  const processors = yield* SessionProcessor.Service
  const session = yield* Session.Service
  const provider = yield* Provider.Service

  const chat = yield* session.create({})
  const parent = yield* user(chat.id, text)
  const msg = yield* assistant(chat.id, parent.id, path.resolve(dir))
  const mdl = yield* provider.getModel(ref.providerID, ref.modelID)
  const handle = yield* processors.create({ assistantMessage: msg, sessionID: chat.id, model: mdl })

  const result = yield* handle.process({
    user: {
      id: parent.id,
      sessionID: chat.id,
      role: "user",
      time: parent.time,
      agent: parent.agent,
      model: { providerID: ref.providerID, modelID: ref.modelID },
    } satisfies SessionV1.User,
    sessionID: chat.id,
    model: mdl,
    agent: agent(),
    system: [],
    messages: [{ role: "user", content: text }],
    tools: {},
  })

  return { result, handle, sessionID: chat.id }
})

describe("session.retry.loud-class", () => {
  test("the loud-class message classifies as a retryable APIError", () => {
    const request = MessageV2.fromError(loudApiError(), { providerID: ref.providerID })
    expect(SessionV1.APIError.isInstance(request)).toBe(true)
    expect(SessionRetry.retryable(request, "test")).toEqual({ message: LOUD })
  })

  test("a raw socket-close error (no APICallError wrapper) is still retryable", () => {
    // A provider SDK may surface the transport failure directly, without
    // wrapping it in APICallError. It must not degrade to UnknownError,
    // which is never retried.
    const raw = Object.assign(new TypeError("The socket connection was closed unexpectedly."), {
      code: "ECONNRESET",
    })
    const request = MessageV2.fromError(raw, { providerID: ref.providerID })
    expect(SessionV1.APIError.isInstance(request)).toBe(true)
    expect(SessionRetry.retryable(request, "test")).toBeDefined()
  })

  test("a connection-refused error is retryable", () => {
    const raw = Object.assign(new TypeError("Unable to connect. Is the computer able to access the url?"), {
      code: "ConnectionRefused",
    })
    const request = MessageV2.fromError(raw, { providerID: ref.providerID })
    expect(SessionV1.APIError.isInstance(request)).toBe(true)
    expect(SessionRetry.retryable(request, "test")).toBeDefined()
  })

  test("a provider-level rejection is NOT retryable", () => {
    const request = MessageV2.fromError(nonRetryableApiError(), { providerID: ref.providerID })
    expect(SessionV1.APIError.isInstance(request)).toBe(true)
    expect(SessionRetry.retryable(request, "test")).toBeUndefined()
  })

  test("the SDK's retry-exhaustion wrapper does not mask the underlying retryability", () => {
    // When the AI SDK exhausts its own retry budget it throws RetryError,
    // which is not an APICallError. The loud-class error underneath must
    // still classify as retryable.
    const wrapped = new RetryError({
      message: `Failed after 3 attempts. Last error: ${LOUD}`,
      reason: "maxRetriesExceeded",
      errors: [loudApiError(), loudApiError(), loudApiError()],
    })
    const request = MessageV2.fromError(wrapped, { providerID: ref.providerID })
    expect(SessionV1.APIError.isInstance(request)).toBe(true)
    expect(SessionRetry.retryable(request, "test")).toEqual({ message: LOUD })
  })

  test("the maxRetries default is non-zero", () => {
    expect(LLM.DEFAULT_MAX_RETRIES).toBeGreaterThan(0)
  })
})

itLoud.live("a retryable APIError mid-turn re-issues the turn", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        loudCalls = 0
        const { result, handle } = yield* process(dir, "loud death")

        // The failed turn was re-issued: two provider calls, and the turn
        // completed on the second one.
        expect(loudCalls).toBe(2)
        expect(result).toBe("continue")
        expect(handle.message.error).toBeUndefined()
      }),
    { config: cfg as any },
  ),
)

itNonRetryable.live("a non-retryable APIError does NOT re-issue the turn", () =>
  provideTmpdirInstance(
    (dir) =>
      Effect.gen(function* () {
        nonRetryableCalls = 0
        const { result, handle } = yield* process(dir, "auth death")

        // No re-issue: a rejected credential stays a rejected credential.
        expect(nonRetryableCalls).toBe(1)
        expect(result).toBe("stop")
        expect(handle.message.error?.name).toBe("APIError")
      }),
    { config: cfg as any },
  ),
)
