# copilot-bridge

Node HTTP/SSE server that hosts a [`@github/copilot-sdk`](https://github.com/github/copilot-sdk) `CopilotClient`
and exposes it to the `benchmark-UI` browser app. The bridge forwards the agent's `ask_user`,
elicitation, and permission requests to the UI as SSE events, and the UI answers them via a
native modal — so the SKILL (`.apm/skills/benchmark-qed-setup/`) runs entirely in-app, no VS Code
needed.

## Run

```bash
cd copilot-bridge
npm install
npm run start          # listens on COPILOT_BRIDGE_PORT (default 8788)
```

The bridge expects `copilot login` to have been run at least once in a shell so that
`useLoggedInUser: true` finds credentials under `~/.copilot/`. The UI's auth badge calls
`/api/copilot/auth/status` to check.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness check |
| `GET`  | `/api/copilot/auth/status` | Shells `copilot auth status` |
| `POST` | `/api/copilot/sessions` | Body `{ initialPrompt?, model?, skillDirectories? }` → `{ sessionId }` |
| `GET`  | `/api/copilot/sessions/:id/events` | SSE stream (replays pending interactive requests on reconnect) |
| `POST` | `/api/copilot/sessions/:id/respond` | Body `{ requestId, kind, value }` |
| `POST` | `/api/copilot/sessions/:id/message` | Body `{ prompt }` |
| `DELETE` | `/api/copilot/sessions/:id` | Tear down the session |

## SSE event types

Forwarded directly from the SDK (`event:` is the type, `data:` is JSON):

- `assistant.message`, `assistant.message_delta`, `assistant.reasoning(_delta)`
- `tool.execution_start`, `tool.execution_complete`
- `user.message`, `session.idle`, `session.closed`, `session.error`

Bridge-owned events that block the agent until the browser POSTs `/respond`:

- `user_input.request` — `{ requestId, data: { question, choices?, allowFreeform } }`
- `elicitation.request` — `{ requestId, data: { message, requestedSchema, mode, elicitationSource } }`
- `permission.request` — `{ requestId, data: { kind, toolName, toolCallId, fileName?, fullCommandText? } }`

Answer shape per `kind`:

| kind | value |
|---|---|
| `user_input` | `{ answer: string, wasFreeform?: boolean }` |
| `elicitation` | `{ action: "accept"\|"decline"\|"cancel", content?: object }` |
| `permission` | `{ decision: "approve-once"\|"approve-for-session"\|"reject", feedback?: string }` |
