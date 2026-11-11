import { Config } from "@/config/config"
import { Provider } from "@/provider/provider"
import * as InstanceState from "@/effect/instance-state"
import { Effect } from "effect"
import { HttpApiBuilder } from "effect/unstable/httpapi"
import { InstanceHttpApi } from "../api"
import { markInstanceForReload } from "../lifecycle"

export const configHandlers = HttpApiBuilder.group(InstanceHttpApi, "config", (handlers) =>
  Effect.gen(function* () {
    const providerSvc = yield* Provider.Service
    const configSvc = yield* Config.Service

    const get = Effect.fn("ConfigHttpApi.get")(function* () {
      return yield* configSvc.get()
    })

    const update = Effect.fn("ConfigHttpApi.update")(function* (ctx) {
      yield* configSvc.update(ctx.payload)
      const instanceCtx = yield* InstanceState.context
      // Hot-reload: re-bootstrap the instance inline (before response) so the
      // next request sees updated agent defs, prompts, and tools immediately.
      // markInstanceForReload disposes + re-bootstraps in the pre-response
      // handler, so by the time the client reads the response the new config
      // is already live. Old instance is never left dangling.
      yield* markInstanceForReload(instanceCtx, {
        directory: instanceCtx.directory,
        worktree: instanceCtx.worktree,
        project: instanceCtx.project,
      })
      return ctx.payload
    })

    const providers = Effect.fn("ConfigHttpApi.providers")(function* () {
      const providers = yield* providerSvc.list()
      return {
        providers: Object.values(providers).map(Provider.toPublicInfo),
        default: Provider.defaultModelIDs(providers),
      }
    })

    return handlers.handle("get", get).handle("update", update).handle("providers", providers)
  }),
)
