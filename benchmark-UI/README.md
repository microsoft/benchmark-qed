# benchmark-UI

Browser app for running and exploring `benchmark-qed` workspaces. It bundles three
processes:

| Process | What it does | Default URL |
|---|---|---|
| `ui` (Vite) | React frontend | `http://localhost:5173` |
| `init-runner` | Node sidecar — runs `benchmark-qed` CLI jobs (init / run-config / autoe / autoq), exposes a sandboxed FS API, and a native folder picker | `http://localhost:8787` |
| `copilot-bridge` | Node HTTP/SSE wrapper around `@github/copilot-sdk` so the in-app AI Assistant (skills under `.apm/skills/`) can drive the agent without VS Code | `http://localhost:8788` |

All three must be running for the UI to work end-to-end. The `dev:all` script
launches them together.

---

## 1. Prerequisites

- **Node.js 20+** (matches `node-pty`'s prebuilds). On Windows you also need
  the Visual Studio Build Tools (C++ workload) the first time `node-pty` builds
  — `npm install -g windows-build-tools` works on older systems, on newer ones
  install "Desktop development with C++" via the VS Installer.
- **Python 3.11** with the repo's virtualenv set up via [`uv`](https://docs.astral.sh/uv/),
  so the runner can find the `benchmark-qed` CLI. From the repo root:
  ```bash
  # macOS / Linux
  uv sync
  source .venv/bin/activate

  # Windows PowerShell
  uv sync
  .\.venv\Scripts\Activate.ps1

  # Windows cmd.exe
  uv sync
  .\.venv\Scripts\activate.bat
  ```
  The init-runner searches `PATH` for `benchmark-qed`; running `npm run init-runner`
  from a shell where the venv is active is the simplest way.
- **GitHub Copilot CLI** logged in on this machine — see [Copilot login](#2-copilot-login).
- **macOS / Linux / Windows 10+ (1809 or newer for ConPTY)** are supported.
  The init-runner uses PowerShell's `FolderBrowserDialog` for the native
  folder picker on Windows and `osascript` on macOS.

---

## 2. Copilot login

The `copilot-bridge` uses `useLoggedInUser: true`, which means it reuses
credentials produced by the standalone Copilot CLI. You need to log in **once**
in a terminal before the AI Assistant panel will work.

```bash
# Install the Copilot CLI if you don't have it
npm install -g @github/copilot

# Log in (opens a browser / device-code flow)
copilot

# Verify
copilot auth status
```

Credentials are stored under `~/.copilot/` and the bridge picks them up
automatically. The UI's auth badge hits `GET /api/copilot/auth/status` on the
bridge to confirm the session is valid; if it isn't, re-run `copilot` from a
shell.

> If `copilot auth status` reports logged-in but the UI still shows "not
> authenticated", confirm the bridge is running on `:8788` and that nothing else
> is bound to that port.

---

## 3. Install

```bash
cd benchmark-UI
npm install

cd ../copilot-bridge
npm install
```

(The init-runner has no extra deps beyond `benchmark-UI/node_modules`.)

---

## 4. Run

From `benchmark-UI/`, with the repo's Python venv active:

```bash
npm run dev:all
```

That starts all three processes in one terminal with prefixed output:

- `runner` → init-runner (`scripts/init-runner.mjs`) on `:8787`
- `bridge` → copilot-bridge (`../copilot-bridge/src/server.mjs`) on `:8788`
- `ui`     → Vite dev server on `:5173`

Open <http://localhost:5173>. Hot reload is enabled for the UI; the runner and
bridge restart on file save only if you launch them with `nodemon` (not
configured by default — just re-run `npm run dev:all`).

### Run individually

Use these when debugging one piece in isolation:

```bash
npm run dev              # UI only
npm run init-runner      # sidecar only
npm run copilot-bridge   # bridge only
```

### Configuration env vars

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `INIT_RUNNER_PORT` | init-runner | `8787` | Hard-coded in the UI; if you change it, also patch `src/App.tsx`, `src/copilot/pickers.ts`, and the dialog components that fetch `http://localhost:8787/...`. |
| `INIT_RUNNER_ORIGIN` | init-runner | `http://localhost:5173` | CORS allow-list. Set to the Vite preview origin (`http://localhost:4173`) if running `npm run preview`. |
| `COPILOT_BRIDGE_PORT` | copilot-bridge | `8788` | Mirror in `src/copilot/useCopilotSession.ts` if you change it. |
| `COPILOT_BRIDGE_ORIGIN` | copilot-bridge | `http://localhost:5173` | CORS allow-list. |

---

## 5. Project layout

```
benchmark-UI/
├── index.html
├── vite.config.ts
├── scripts/
│   └── init-runner.mjs        # Node sidecar: jobs + FS API + folder picker
├── src/
│   ├── App.tsx                # Top-level shell (workspaces, tabs, dialogs)
│   ├── copilot/
│   │   ├── useCopilotSession.ts   # Client for copilot-bridge SSE/REST
│   │   └── pickers.ts             # Folder/file pickers via init-runner
│   ├── components/
│   │   ├── CopilotPanel.tsx        # AI Assistant panel + skill picker
│   │   ├── JobLogViewer.tsx        # Live job output + Download log
│   │   ├── EvaluateQuestionsPickerDialog.tsx
│   │   └── ...
│   ├── recentReports.ts        # localStorage-backed report history
│   └── types.ts
└── public/

../copilot-bridge/              # Separate package — see its README.md
../.apm/skills/                 # Skills loaded by the AI Assistant
    benchmark-qed-setup/
    benchmark-qed-autoe/
    benchmark-qed-autoq/
    benchmark-qed-question-quality/
```

The AI Assistant in the UI is just a `CopilotPanel` that talks to the bridge.
Skills are picked from the in-panel skill grid; they live in `../.apm/skills/`
and are passed to the bridge as `skillDirectories` when a session starts.

---

## 6. Common tasks

### Build for production

```bash
npm run build       # tsc -b && vite build → dist/
npm run preview     # serves dist/ on :4173 (set INIT_RUNNER_ORIGIN/COPILOT_BRIDGE_ORIGIN to match)
```

### Lint

```bash
npm run lint
```

### Reset local UI state

Workspaces, recent reports, and a few flags are stored in `localStorage` under
keys prefixed `benchmark-qed:`. To wipe them, in the browser devtools console:

```js
Object.keys(localStorage)
  .filter((k) => k.startsWith("benchmark-qed:"))
  .forEach((k) => localStorage.removeItem(k));
```

---

## 7. Troubleshooting

- **"AI Assistant: not authenticated"** — run `copilot` in a shell, confirm with
  `copilot auth status`, then refresh the page. The bridge does not embed a
  login flow.
- **"`benchmark-qed` not found in PATH"** in job logs — your shell didn't have
  the repo venv active when you started `init-runner`. Stop it (`Ctrl+C` in the
  `dev:all` terminal), `source .venv/bin/activate`, then `npm run dev:all`
  again.
- **Folder picker does nothing** — only the init-runner's `/api/pick-folder`
  endpoint can open a native dialog. On macOS it shells out to `osascript`
  (grant Terminal automation permission the first time); on Windows it
  shells out to `powershell.exe`'s `FolderBrowserDialog`. Make sure the
  runner is up.
- **`EADDRINUSE` on `:8787` / `:8788`** — a previous runner/bridge is still
  running. `lsof -i :8787` / `lsof -i :8788` and kill the PID, or change the
  port via the env vars above (remember to patch the matching UI constants).
- **Reports don't auto-open after the agent finishes** — the picker writes
  `<destination>/<filename>` and the UI probes `init-runner` for the file on
  `session.idle`. If the runner is down, the report still exists on disk but
  won't open automatically; use the "Recent reports" list in the picker to
  reopen it.

---

## 8. Where to look in the code

- Bridge protocol & SSE events: `../copilot-bridge/README.md`.
- Sandboxed FS API (`/api/fs/list|read|write|create-file`) and job lifecycle:
  `scripts/init-runner.mjs`.
- Skill picker, transcript archiving, "Back to options" flow:
  `src/components/CopilotPanel.tsx`.
- Auto-open of generated reports + hidden destination workspaces:
  `src/App.tsx` (`openRecentReport`, `finalizePendingReport`,
  `addWorkspace({ hiddenFromSidebar: true })`).
