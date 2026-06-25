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

Activate the repo's Python venv first so the init-runner can find `uv` /
`benchmark-qed` on `PATH`:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows cmd.exe
.\.venv\Scripts\activate.bat
```

Then, from `benchmark-UI/`:

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

## 7. Azure Blob Storage workspaces

The UI can mount an Azure Blob container as a read-only workspace (the
"Add workspace → Azure Blob Container" tab). The browser talks to Azure
**directly** via `@azure/storage-blob` — there is no backend proxy — so the
storage account must explicitly allow your UI's origin via CORS.

### 7.1 Configure CORS on the storage account

Without this, every request fails with:

```
Access to fetch at 'https://<account>.blob.core.windows.net/...' from origin
'http://localhost:5173' has been blocked by CORS policy: Response to preflight
request doesn't pass access control check: No 'Access-Control-Allow-Origin'
header is present on the requested resource.
```

**Azure Portal**: Storage account → **Settings** → **Resource sharing (CORS)**
→ **Blob service** tab → add a row:

| Field | Value |
|---|---|
| Allowed origins | `http://localhost:5173` (one per line for multiple origins; `*` only for throw-away testing) |
| Allowed methods | `GET, HEAD, OPTIONS` (add `PUT, POST, DELETE` if the UI will ever write) |
| Allowed headers | `*` |
| Exposed headers | `*` |
| Max age (seconds) | `3600` |

Save. The rule is effective within ~30 s; hard-refresh the UI (Cmd/Ctrl+
Shift+R) before retrying.

**Azure CLI** equivalent:

```bash
az storage cors add \
  --services b \
  --methods GET HEAD OPTIONS \
  --origins http://localhost:5173 \
  --allowed-headers '*' \
  --exposed-headers '*' \
  --max-age 3600 \
  --account-name <your-storage-account>
```

For production, replace `http://localhost:5173` with the actual UI origin.

### 7.2 Enable editing blob files from the UI

By default a blob workspace is **read-only**: opening `settings.yaml` shows
the content but the Save button is disabled, and right-click → Delete fails
with *"This workspace source does not support deleting paths"*. The
`BlobFileSource` constructor decides the per-action capability by looking
at the SAS's `sp` parameter:

```ts
this.writable  = /w/i.test(sp) || /c/i.test(sp);   // BlobFileSource.ts
this.deletable = /d/i.test(sp);
```

To make the workspace writable end-to-end you need to change **three**
things — the SAS, the CORS rule, and the in-app workspace entry:

1. **Regenerate the SAS with the permissions you need** — use `sp=rcwl`
   for read + create + write + list, or `sp=rcwdl` if you also want to
   delete blobs and folders.

   ```bash
   az storage container generate-sas \
     --account-name <your-storage-account> \
     --name <container> \
     --permissions rcwdl \
     --expiry 2026-12-31T23:59:00Z \
     --https-only \
     --auth-mode key \
     --full-uri
   ```

   Portal equivalent: Container → **Shared access tokens** → tick **Read,
   Add, Create, Write, List** (and **Delete** for removal) → Generate.

2. **Extend the CORS rule** on the storage account (§7.1) so the browser
   can issue write requests. The Allowed methods column must include the
   verbs the SDK uses:

   | Operation | Method(s) used by `@azure/storage-blob` |
   |---|---|
   | List + read | `GET, HEAD, OPTIONS` |
   | Overwrite / upload (`writeFile`) | add `PUT` |
   | Create-via-block (large uploads) | add `POST` |
   | Delete | add `DELETE` |

   A workable set for full edit access is:
   `GET, HEAD, OPTIONS, PUT, POST, DELETE`.

3. **Re-add the workspace in the UI.** The dialog reads `sp` only when the
   workspace is created; the `writable` flag is cached for the session.
   Right-click the existing blob workspace in the sidebar → **Remove**,
   then add it again with the updated credentials. The editor's Save button is now
   enabled, and `BlobFileSource.writeFile` will `PUT` the file back to the
   container.

#### Caveats

- **Folders are virtual.** Azure has no real directories — they only
  exist as common prefixes. Deleting a "folder" via right-click iterates
  every blob whose name starts with that prefix and deletes them one by
  one (`BlobFileSource.deletePath`). For very large trees this can take
  a while and the UI does not show per-blob progress.
- **Last-writer-wins.** `writeFile` does a plain block-blob upload — no
  ETag / If-Match precondition. If two people share the SAS and edit the
  same file, the later save silently overwrites the earlier one.
- **Content-Type is forced** to `text/plain; charset=utf-8` on save,
  regardless of the original blob's type. Tools that switch behavior on
  `Content-Type` (e.g. `application/x-yaml`) may break after an edit.
- **Writeable SAS in the browser is sensitive.** Anyone who can read the
  page (devtools, screen recording, localStorage) can write to your
  container. Keep the CORS `Allowed origins` scoped to your real UI origin
  (never `*` for writeable SAS), keep `se=` short, and revoke via the
  portal's "Revoke all SAS" if it ever leaks.

### 7.3 Run jobs and download datasets against a blob workspace

The action-bar buttons next to a blob workspace — **Download predefined
inputs** (⬇) and **Run workspace** (▶) — work the same way as for local
workspaces, but the init-runner translates the click into a `benchmark-qed`
invocation that talks to the container directly:

- **Download** spawns:

  ```text
  benchmark-qed data download <dataset> <subdir> --accept-terms \
    --storage-type blob \
    --container-name <container> \
    --connection-string "BlobEndpoint=https://<account>.blob.core.windows.net;SharedAccessSignature=<sas-token>" \
    [--base-dir <workspace-prefix>]
  ```

  The dataset archive is extracted directly into
  `<workspace-prefix>/<subdir>/` in the container — no local staging. The
  dialog's "Pick Folder" button is hidden for blob workspaces because the
  destination is interpreted as a sub-prefix of the container, not a host
  filesystem path.

- **Run workspace** spawns one of:

  ```text
  benchmark-qed autoq      blob://<container>/<prefix>/settings.yaml output --connection-string "<conn>"
  benchmark-qed autoe pairwise-scores  blob://… output --connection-string "<conn>"
  benchmark-qed autoe reference-scores blob://… output --connection-string "<conn>"
  benchmark-qed autoe assertion-scores blob://… output --connection-string "<conn>"
  ```

  The Python CLI's `resolve_config_path` downloads the entire prefix into
  a temp directory so prompt-template `!include` references resolve. The
  second positional argument (`output`) is treated as a child path inside
  the `output_storage` block configured in `settings.yaml`, so **the
  settings.yaml in a blob workspace must declare `output_storage` (and
  `input.storage` if reading data from blob) as blob** — otherwise the
  CLI falls back to local `FileStorage` and writes to `./output` in the
  init-runner's working directory.

  `benchmark-qed config init … --storage-type blob …` (already wired into
  the workspace-creation flow) emits the correct `output_storage` /
  `input.storage` blocks; if you authored the settings.yaml by hand,
  make sure those are present.

Requirements for the buttons to succeed:

1. The SAS must allow both read and write (`sp=rcwl` minimum,
   `sp=rcwdl` if you also want to delete intermediate outputs).
2. CORS on the storage account must allow `GET, HEAD, OPTIONS, PUT,
   POST` (and `DELETE` if applicable) — the init-runner uses the SAS
   from the browser; the Azure SDK in Python issues the actual data
   requests from the runner process.
3. The benchmark-qed Python venv must be active when you started
   `npm run dev:all` (the runner can't find the CLI otherwise — see §8
   for that error).

### 7.4 Diagnose blob errors

Open browser devtools → Network tab → retry → click the failing request:

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to fetch` + red "blocked by CORS policy" message | Storage account CORS doesn't include the UI origin | 7.1 above |
| `403 AuthorizationFailure` / `AuthenticationFailed` | SAS lacks required permission or has expired | Regenerate with at least `sp=rl` and a future `se=` |
| `403 AuthorizationPermissionMismatch` | SAS scoped to the wrong resource (`sr` mismatch) | Use a container SAS (`sr=c`), not an account or blob SAS |
| `403 This request is not authorized to perform this operation using this permission` on list | SAS missing `l` | Regenerate with `sp=rl` (or `rcwl` for editing) |
| `403 …using this permission` on save (`PUT`) | SAS missing `w` (and/or `c`) | Regenerate with `sp=rcwl` and re-add the workspace |
| Save button stays disabled even with a writeable SAS | Old (read-only) workspace entry still active | Remove the workspace and re-add it with the new SAS — `writable` is cached on connect |
| `"This workspace source does not support deleting paths"` banner on blob workspace | SAS lacks `d` permission (workspace was added with a non-deletable SAS) | Regenerate with `sp=rcwdl` and re-add the workspace |
| `405 Method Not Allowed` on `PUT`/`DELETE` | CORS allows only `GET/HEAD/OPTIONS` | Add `PUT` (and `POST`/`DELETE` as needed) to §7.1 |
| `404 ContainerNotFound` | Wrong container name | Double-check the path segment after the account |
| `400 InvalidQueryParameter` on `restype=container&comp=list` | SAS not allowed for listing | Add `l` to `sp` |

---

## 8. Troubleshooting

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
- **"Failed to load `<workspace>`: RestError: Error sending request: Failed to
  fetch"** when adding a blob workspace — CORS on the storage account; see
  [§7.1](#71-configure-cors-on-the-storage-account).

---

## 9. Where to look in the code

- Bridge protocol & SSE events: `../copilot-bridge/README.md`.
- Sandboxed FS API (`/api/fs/list|read|write|create-file`) and job lifecycle:
  `scripts/init-runner.mjs`.
- Skill picker, transcript archiving, "Back to options" flow:
  `src/components/CopilotPanel.tsx`.
- Auto-open of generated reports + hidden destination workspaces:
  `src/App.tsx` (`openRecentReport`, `finalizePendingReport`,
  `addWorkspace({ hiddenFromSidebar: true })`).
- Blob workspace integration: `src/sources/BlobFileSource.ts`,
  `src/components/BlobConnectDialog.tsx`, `AddWorkspaceTabsDialog.tsx`.
