import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { promisify } from "node:util";
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";
import { DefaultAzureCredential } from "@azure/identity";
import { BlobServiceClient } from "@azure/storage-blob";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PORT = Number(process.env.INIT_RUNNER_PORT || 8787);
const ALLOWED_ORIGIN = process.env.INIT_RUNNER_ORIGIN || "http://localhost:5173";

// node-pty's prebuilt `spawn-helper` ships without the executable bit set on
// some npm versions, which causes `pty.spawn` to fail with "posix_spawnp
// failed.". Ensure it's executable before we import the module.
try {
  const nodePtyRoot = path.resolve(__dirname, "..", "node_modules", "node-pty", "prebuilds");
  for (const platform of ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64"]) {
    const helper = path.join(nodePtyRoot, platform, "spawn-helper");
    if (fs.existsSync(helper)) {
      try {
        fs.chmodSync(helper, 0o755);
      } catch {
        /* ignore */
      }
    }
  }
} catch {
  /* ignore */
}
const { default: pty } = await import("node-pty");
const execFileAsync = promisify(execFile);

const initJobs = new Map();
const runJobs = new Map();
const jobProcesses = new Map(); // jobId -> child process
const cancelledJobs = new Set(); // jobIds that were user-cancelled

// ---------------------------------------------------------------------------
// Persistent job store. Survives runner restarts.
// ---------------------------------------------------------------------------
const JOBS_STORE_DIR = path.join(REPO_ROOT, ".benchmark-qed-runner");
const JOBS_STORE_FILE = path.join(JOBS_STORE_DIR, "jobs.json");
// Cap stored output per job so a long-running job's log doesn't blow up the
// store file. The live in-memory copy is unbounded; we only truncate what
// gets serialized to disk.
const PERSISTED_OUTPUT_LIMIT = 200_000; // ~200 KB

function truncateForPersist(s) {
  if (!s) return "";
  if (s.length <= PERSISTED_OUTPUT_LIMIT) return s;
  const head = s.slice(0, 2000);
  const tail = s.slice(-PERSISTED_OUTPUT_LIMIT + 2000);
  return `${head}\n\n... [truncated ${s.length - PERSISTED_OUTPUT_LIMIT} chars] ...\n\n${tail}`;
}

let persistTimer = null;
function persistJobsSoon() {
  if (persistTimer) return;
  persistTimer = setTimeout(() => {
    persistTimer = null;
    try {
      if (!fs.existsSync(JOBS_STORE_DIR)) {
        fs.mkdirSync(JOBS_STORE_DIR, { recursive: true });
      }
      const payload = {
        version: 1,
        initJobs: Array.from(initJobs.values()).map((j) => ({
          ...j,
          output: truncateForPersist(j.output),
        })),
        runJobs: Array.from(runJobs.values()).map((j) => ({
          ...j,
          output: truncateForPersist(j.output),
        })),
      };
      const tmp = `${JOBS_STORE_FILE}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify(payload), "utf-8");
      fs.renameSync(tmp, JOBS_STORE_FILE);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[init-runner] Failed to persist jobs:", err);
    }
  }, 500);
}

function loadPersistedJobs() {
  try {
    if (!fs.existsSync(JOBS_STORE_FILE)) return;
    const raw = fs.readFileSync(JOBS_STORE_FILE, "utf-8");
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return;
    const restore = (arr, map) => {
      if (!Array.isArray(arr)) return;
      for (const j of arr) {
        if (!j || typeof j !== "object" || typeof j.id !== "string") continue;
        // Any job that was 'running' when the runner died is now orphaned.
        if (j.status === "running") {
          j.status = "failed";
          j.endedAt = j.endedAt || new Date().toISOString();
          j.exitCode = j.exitCode ?? -1;
          j.output = `${j.output || ""}\n[Runner restarted while job was running; status forced to failed.]\n`;
        }
        map.set(j.id, j);
      }
    };
    restore(data.initJobs, initJobs);
    restore(data.runJobs, runJobs);
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[init-runner] Failed to load persisted jobs:", err);
  }
}

loadPersistedJobs();

function flushPersistedJobs() {
  if (persistTimer) {
    clearTimeout(persistTimer);
    persistTimer = null;
  }
  try {
    if (!fs.existsSync(JOBS_STORE_DIR)) {
      fs.mkdirSync(JOBS_STORE_DIR, { recursive: true });
    }
    const payload = {
      version: 1,
      initJobs: Array.from(initJobs.values()).map((j) => ({
        ...j,
        output: truncateForPersist(j.output),
      })),
      runJobs: Array.from(runJobs.values()).map((j) => ({
        ...j,
        output: truncateForPersist(j.output),
      })),
    };
    fs.writeFileSync(JOBS_STORE_FILE, JSON.stringify(payload), "utf-8");
  } catch {
    /* ignore */
  }
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => {
    flushPersistedJobs();
    process.exit(0);
  });
}
process.on("beforeExit", flushPersistedJobs);


function resolveCorsOrigin(req) {
  const origin = req.headers.origin;
  if (!origin) return ALLOWED_ORIGIN;
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) {
    return origin;
  }
  return ALLOWED_ORIGIN;
}

function json(req, res, status, payload) {
  const corsOrigin = resolveCorsOrigin(req);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": corsOrigin,
    "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(JSON.stringify(payload));
}

function resolveFsPath(rootPath, relPath = "") {
  const root = path.resolve(rootPath);
  const target = path.resolve(root, relPath || ".");
  if (target !== root && !target.startsWith(`${root}${path.sep}`)) {
    throw new Error("Path escapes root folder.");
  }
  return { root, target };
}

function normalizeBlobPrefix(prefix = "") {
  return String(prefix).trim().replace(/^\/+|\/+$/g, "");
}

function resolveBlobConfig(blob) {
  if (!blob || typeof blob !== "object") {
    throw new Error("blob configuration is required.");
  }

  if (typeof blob.accountUrl === "string" && typeof blob.containerName === "string") {
    return {
      accountUrl: blob.accountUrl.replace(/\/+$/, ""),
      containerName: blob.containerName.trim(),
      prefix: normalizeBlobPrefix(blob.prefix),
    };
  }

  throw new Error("blob.accountUrl and blob.containerName are required.");
}

function blobFullPath(prefix = "", relPath = "") {
  const normalizedPrefix = normalizeBlobPrefix(prefix);
  const normalizedPath = String(relPath).replace(/^\/+|\/+$/g, "");
  return [normalizedPrefix, normalizedPath].filter(Boolean).join("/");
}

function getBlobContainerClient(accountUrl, containerName) {
  const credential = new DefaultAzureCredential();
  const service = new BlobServiceClient(accountUrl, credential);
  return service.getContainerClient(containerName);
}

async function readBlobText(blobClient) {
  const resp = await blobClient.download();
  if (!resp.readableStreamBody) return "";
  const chunks = [];
  for await (const chunk of resp.readableStreamBody) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf-8");
}

// Minimal terminal-style processor so in-place repaints (rich progress bars,
// spinners, etc.) overwrite the previous frame instead of accumulating as
// duplicated text. Handles \r, CSI cursor-up (\x1b[<n>A), and CSI erase-line
// (\x1b[K / \x1b[2K). All other ANSI escapes are stripped.
//
// Pure: takes the current cumulative buffer and a chunk of raw output,
// returns the new buffer. Used by the live in-memory job.output buffer
// (which gets size-capped by the caller).
function processTerminalChunk(buf, chunk) {
  const txt = String(chunk ?? "");

  const eraseCurrentLine = () => {
    const nl = buf.lastIndexOf("\n");
    buf = nl === -1 ? "" : buf.slice(0, nl + 1);
  };

  let i = 0;
  while (i < txt.length) {
    const ch = txt[i];
    if (ch === "\r") {
      if (txt[i + 1] === "\n") {
        buf += "\n";
        i += 2;
      } else {
        eraseCurrentLine();
        i += 1;
      }
    } else if (ch === "\x1b" && txt[i + 1] === "[") {
      let j = i + 2;
      while (j < txt.length) {
        const code = txt.charCodeAt(j);
        if (code >= 0x40 && code <= 0x7e) break;
        j += 1;
      }
      if (j >= txt.length) break;
      const final = txt[j];
      const params = txt.slice(i + 2, j).replace(/[^0-9;]/g, "");
      if (final === "A") {
        const n = Math.max(1, parseInt(params || "1", 10) || 1);
        eraseCurrentLine();
        for (let k = 0; k < n; k += 1) {
          if (buf.endsWith("\n")) buf = buf.slice(0, -1);
          eraseCurrentLine();
        }
      } else if (final === "K" || final === "J") {
        eraseCurrentLine();
      } else if (final === "G" || final === "H" || final === "f") {
        eraseCurrentLine();
      }
      i = j + 1;
    } else if (ch === "\x1b" && txt[i + 1] === "]") {
      let j = i + 2;
      while (j < txt.length) {
        if (txt[j] === "\x07") {
          j += 1;
          break;
        }
        if (txt[j] === "\x1b" && txt[j + 1] === "\\") {
          j += 2;
          break;
        }
        j += 1;
      }
      i = j;
    } else if (ch === "\x1b") {
      i += txt[i + 1] ? 2 : 1;
    } else {
      buf += ch;
      i += 1;
    }
  }

  // Collapse consecutive progress-bar style lines (rich / tqdm) by label so
  // we don't accumulate hundreds of redraws when stdout isn't a TTY.
  return collapseProgressLines(buf);
}

function appendOutput(job, chunk) {
  let buf = processTerminalChunk(job.output, chunk);
  if (buf.length > 120_000) {
    buf = buf.slice(-120_000);
  }
  job.output = buf;
}

// Matches lines that look like a progress bar — they contain a run of the
// block-drawing characters rich/tqdm use to render bars.
const PROGRESS_LINE_RE = /[\u2501\u2578\u257A\u2588\u2591\u2592\u2593#=\-█▉▊▋▌▍▎▏]{3,}/;

function progressKey(line) {
  // Use the text before the first progress-bar character run as a stable
  // identifier for that bar (e.g. "Scoring empowerment...").
  const m = line.match(PROGRESS_LINE_RE);
  if (!m) return null;
  return line.slice(0, m.index).trimEnd();
}

function collapseProgressLines(text) {
  const lines = text.split("\n");
  const keep = new Array(lines.length).fill(true);

  // Pass 1: drop any line that is byte-identical to the previous kept line.
  // This catches every progress format (rich, tqdm, custom) when stdout
  // isn't a TTY and frames are re-emitted as fresh lines.
  let prevKeptIdx = -1;
  for (let idx = 0; idx < lines.length; idx += 1) {
    if (prevKeptIdx !== -1 && lines[idx] === lines[prevKeptIdx]) {
      keep[idx] = false;
    } else {
      prevKeptIdx = idx;
    }
  }

  // Pass 2: collapse progress-bar lines by their text-before-the-bar key,
  // keeping only the most recent occurrence per label. This handles bars
  // that change every frame (percentages, ETA) so they don't accumulate.
  const seen = new Set();
  for (let idx = lines.length - 1; idx >= 0; idx -= 1) {
    if (!keep[idx]) continue;
    const key = progressKey(lines[idx]);
    if (key === null) continue;
    if (seen.has(key)) {
      keep[idx] = false;
    } else {
      seen.add(key);
    }
  }

  const out = [];
  for (let idx = 0; idx < lines.length; idx += 1) {
    if (keep[idx]) out.push(lines[idx]);
  }
  return out.join("\n");
}

// Environment passed to spawned children. We deliberately leave TERM /
// FORCE_COLOR alone so rich/tqdm keep emitting progress frames at their normal
// cadence — the per-label dedupe in collapseProgressLines() keeps the buffer
// to a single row per progress bar while still letting the latest frame (with
// updated percentage / ETA) replace the previous one on each poll.
const CHILD_ENV = {
  ...process.env,
  PYTHONUNBUFFERED: "1",
  PYTHONIOENCODING: "utf-8",
};

const IS_WINDOWS = process.platform === "win32";

// Quote a single argument for cmd.exe's `/c` line. cmd.exe parses its command
// line itself (rather than handing it to CommandLineToArgvW), so we wrap any
// token containing whitespace or cmd metacharacters in double quotes and
// escape embedded quotes by doubling them — the convention `cmd.exe` uses
// for `/c "..."`.
function quoteForCmd(arg) {
  const s = String(arg);
  if (s.length === 0) return '""';
  if (!/[\s"&|<>^()%!]/.test(s)) return s;
  return `"${s.replace(/"/g, '""')}"`;
}

// On Windows, route spawns through `cmd.exe /d /s /c` so that PATHEXT
// resolution kicks in and `.cmd`/`.bat` console-script shims (how pip / uv /
// pipx install `benchmark-qed` on Windows) can actually execute. Both
// node-pty's ConPTY backend and Node's child_process.spawn refuse to launch
// .cmd files directly, producing a generic "File not found:" error.
function toWindowsCmdInvocation(cmd, cmdArgs) {
  const line = [cmd, ...cmdArgs].map(quoteForCmd).join(" ");
  // `/s` makes cmd.exe strip exactly the first and last quote on the command
  // line, so we wrap the whole inner command in outer quotes that get peeled
  // back off, leaving our per-token quoting intact.
  return {
    cmd: process.env.ComSpec || "cmd.exe",
    args: ["/d", "/s", "/c", `"${line}"`],
  };
}

// node-pty performs its OWN argv -> command-line quoting (it does not support
// `windowsVerbatimArguments`). So the pre-quoted, outer-wrapped form produced
// by toWindowsCmdInvocation gets re-escaped and mangled, which makes cmd.exe
// report "The filename, directory name, or volume label syntax is incorrect."
// For the pty path we therefore pass plain, un-pre-quoted tokens and let
// node-pty quote them. A real .exe (e.g. uv.exe) can be launched directly;
// only console-script *.cmd/*.bat shims need to go through cmd.exe.
function toWindowsPtyInvocation(cmd, cmdArgs) {
  if (/\.(cmd|bat)$/i.test(cmd)) {
    return {
      cmd: process.env.ComSpec || "cmd.exe",
      args: ["/d", "/c", cmd, ...cmdArgs],
    };
  }
  return { cmd, args: cmdArgs };
}

function spawnChild(cmd, cmdArgs) {
  const target = IS_WINDOWS ? toWindowsCmdInvocation(cmd, cmdArgs) : { cmd, args: cmdArgs };
  return spawn(target.cmd, target.args, {
    cwd: REPO_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    env: CHILD_ENV,
    windowsVerbatimArguments: IS_WINDOWS,
  });
}

function spawnChildInCwd(cmd, cmdArgs, cwd = REPO_ROOT) {
  const target = IS_WINDOWS ? toWindowsCmdInvocation(cmd, cmdArgs) : { cmd, args: cmdArgs };
  return spawn(target.cmd, target.args, {
    cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: CHILD_ENV,
    windowsVerbatimArguments: IS_WINDOWS,
  });
}

// Detect once which invocation of the benchmark-qed CLI works on this machine.
// Avoids spamming the job log with "'benchmark-qed' not found in PATH" on every
// run. Result is cached for the lifetime of the runner process.
let _resolvedBenchmarkInvocation = null;
function which(cmd) {
  // Cross-platform: use `where` on Windows, `command -v` on POSIX. Both print
  // the resolved absolute path to stdout (one per line) when the binary is
  // discoverable on PATH and exit 0; we keep the first match.
  try {
    const isWin = process.platform === "win32";
    // Node 20+ deprecates passing an args array together with `shell: true`
    // (DEP0190). When we need the shell (Windows PATHEXT resolution), pass
    // the command as a single pre-quoted string instead.
    const res = isWin
      ? spawnSync(`where ${quoteForCmd(cmd)}`, {
          env: CHILD_ENV,
          encoding: "utf8",
          timeout: 5000,
          shell: true,
        })
      : spawnSync("/bin/sh", ["-c", `command -v ${cmd}`], {
          env: CHILD_ENV,
          encoding: "utf8",
          timeout: 5000,
        });
    if (res.status === 0) {
      const out = (res.stdout || "").trim();
      if (out) return out.split(/\r?\n/)[0];
    }
  } catch {
    /* ignore */
  }
  return null;
}
function resolveBenchmarkInvocation() {
  if (_resolvedBenchmarkInvocation) return _resolvedBenchmarkInvocation;
  const defaultPython = process.env.BENCHMARK_QED_PYTHON || "python";
  const probe = (cmd, args) => {
    try {
      // On Windows, console-script entry points (uv, benchmark-qed) are
      // installed as `.cmd` shims that Node's `spawn` can't execute without
      // a shell. Node 20+ deprecates passing an args array with `shell: true`
      // (DEP0190), so we hand the shell a single pre-quoted command line.
      const isWin = process.platform === "win32";
      const res = isWin
        ? spawnSync(
            [cmd, ...args].map(quoteForCmd).join(" "),
            {
              cwd: REPO_ROOT,
              env: CHILD_ENV,
              stdio: "ignore",
              timeout: 90000,
              shell: true,
            },
          )
        : spawnSync(cmd, args, {
            cwd: REPO_ROOT,
            env: CHILD_ENV,
            stdio: "ignore",
            // First-run `uv run` may sync/install the project venv, which
            // can easily exceed a tight timeout. Give probes headroom.
            timeout: 90000,
          });
      return res.status === 0;
    } catch {
      return false;
    }
  };

  // On Windows, prefer `uv run benchmark-qed` when uv is available: a bare
  // `benchmark-qed.cmd` shim often isn't on the PATH the runner inherits
  // (especially when launched from VS Code outside an activated venv), so
  // probing it first wastes time and the failure was the original Windows
  // bug. On macOS / Linux the existing order (bare CLI first, uv second)
  // already worked when the venv is activated, so we leave it untouched.
  const uvAvailable = IS_WINDOWS && Boolean(which("uv"));
  const uvCandidate = {
    cmd: "uv",
    prefix: ["run", "benchmark-qed"],
    label: "uv run benchmark-qed",
    probe: ["run", "benchmark-qed", "--help"],
  };
  const benchmarkCandidate = {
    cmd: "benchmark-qed",
    prefix: [],
    label: "benchmark-qed",
    probe: ["--help"],
  };
  const pythonCandidate = {
    cmd: defaultPython,
    prefix: ["-m", "benchmark_qed"],
    label: `${defaultPython} -m benchmark_qed`,
    probe: ["-m", "benchmark_qed", "--help"],
  };
  const candidates = uvAvailable
    ? [uvCandidate, benchmarkCandidate, pythonCandidate]
    : [benchmarkCandidate, uvCandidate, pythonCandidate];

  for (const c of candidates) {
    if (probe(c.cmd, c.probe)) {
      const absPath = which(c.cmd) || c.cmd;
      _resolvedBenchmarkInvocation = { cmd: c.cmd, absPath, prefix: c.prefix, label: c.label };
      // eslint-disable-next-line no-console
      console.log(`[init-runner] Using '${c.label}' (${absPath}) for benchmark-qed CLI.`);
      return _resolvedBenchmarkInvocation;
    }
    // eslint-disable-next-line no-console
    console.warn(`[init-runner] Probe failed for '${c.label}'.`);
  }

  // Nothing worked at probe time (e.g. a probe timed out). Rather than fall
  // back to a bare `benchmark-qed` that almost certainly isn't on PATH, prefer
  // `uv run benchmark-qed` whenever `uv` is discoverable — that's the
  // documented setup and matches how the CLI is launched on Linux.
  const uvPath = which("uv");
  if (uvPath) {
    _resolvedBenchmarkInvocation = {
      cmd: "uv",
      absPath: uvPath,
      prefix: ["run", "benchmark-qed"],
      label: "uv run benchmark-qed",
    };
    // eslint-disable-next-line no-console
    console.log(
      `[init-runner] No probe succeeded; falling back to 'uv run benchmark-qed' (${uvPath}).`,
    );
    return _resolvedBenchmarkInvocation;
  }

  // No uv either; fall back to plain `benchmark-qed` and let the spawn error
  // surface to the user.
  _resolvedBenchmarkInvocation = { cmd: "benchmark-qed", absPath: "benchmark-qed", prefix: [], label: "benchmark-qed" };
  return _resolvedBenchmarkInvocation;
}

function runCommandWithFallback(job, args, onCompleted) {
  const { absPath, prefix, label } = resolveBenchmarkInvocation();
  const cmdArgs = [...prefix, ...args];
  const commandLabel = `${label} ${args.join(" ")}`;
  job.command = commandLabel;

  const markCompleted = (code) => {
    job.endedAt = new Date().toISOString();
    job.exitCode = code ?? -1;
    const wasCancelled = cancelledJobs.has(job.id);
    if (wasCancelled) {
      job.status = "cancelled";
      cancelledJobs.delete(job.id);
    } else {
      job.status = code === 0 ? "succeeded" : "failed";
    }
    if (jobProcesses.has(job.id)) {
      jobProcesses.delete(job.id);
    }
    if (typeof onCompleted === "function") {
      try {
        onCompleted(job);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[init-runner] onCompleted handler failed:", err);
      }
    }
    persistJobsSoon();
  };

  // Spawn through a pseudo-terminal so that rich/tqdm detect a real TTY and
  // emit their proper redrawing progress bars (with cursor-up ANSI codes,
  // which appendOutput already interprets).
  let proc;
  try {
    const ptyTarget = IS_WINDOWS
      ? toWindowsPtyInvocation(absPath, cmdArgs)
      : { cmd: absPath, args: cmdArgs };
    // On Windows, pass args as a pre-joined string so node-pty (ConPTY) does
    // not apply its own CommandLineToArgvW escaping on top of our cmd.exe
    // quoting. Without this, paths containing spaces get their inner quotes
    // double-escaped, causing "The filename, directory name, or volume label
    // syntax is incorrect" errors on Windows.
    const ptyArgs = IS_WINDOWS ? ptyTarget.args.join(" ") : ptyTarget.args;
    proc = pty.spawn(ptyTarget.cmd, ptyArgs, {
      name: "xterm-256color",
      cols: 120,
      rows: 30,
      cwd: REPO_ROOT,
      env: { ...CHILD_ENV, TERM: "xterm-256color", FORCE_COLOR: "1" },
    });
  } catch (err) {
    job.status = "failed";
    job.endedAt = new Date().toISOString();
    job.exitCode = -1;
    appendOutput(job, `\nFailed to start command: ${err.message}\n`);
    return;
  }

  // Wrap the IPty handle so cancelJob's `.kill("SIGTERM")` still works.
  const handle = {
    kill: (signal) => {
      try {
        proc.kill(signal || "SIGTERM");
      } catch {
        /* ignore */
      }
    },
  };
  jobProcesses.set(job.id, handle);

  proc.onData((data) => {
    appendOutput(job, data);
    persistJobsSoon();
  });
  proc.onExit(({ exitCode }) => markCompleted(exitCode));
}
// Cancel a running job by id
function cancelJob(jobId) {
  const proc = jobProcesses.get(jobId);
  if (proc && typeof proc.kill === "function") {
    cancelledJobs.add(jobId);
    proc.kill("SIGTERM");
    jobProcesses.delete(jobId);
    return true;
  }
  return false;
}

async function runCommandWithFallbackAndCapture(args, options = {}) {
  const cwd = options.cwd || REPO_ROOT;
  const { absPath, prefix, label } = resolveBenchmarkInvocation();
  const capture = { output: "" };
  const attempts = [
    {
      cmd: absPath,
      cmdArgs: [...prefix, ...args],
      label: `${label} ${args.join(" ")}`,
      fallbackMessage: "",
    },
  ];

  for (const attempt of attempts) {
    const result = await new Promise((resolve) => {
      const proc = spawnChildInCwd(attempt.cmd, attempt.cmdArgs, cwd);
      let settled = false;

      proc.stdout.on("data", (chunk) => appendOutput(capture, chunk));
      proc.stderr.on("data", (chunk) => appendOutput(capture, chunk));

      proc.on("error", (err) => {
        if (settled) return;
        settled = true;
        if (err && typeof err === "object" && "code" in err && err.code === "ENOENT") {
          resolve({ kind: "enoent", attempt });
          return;
        }
        resolve({ kind: "error", attempt, err });
      });

      proc.on("close", (code) => {
        if (settled) return;
        settled = true;
        resolve({ kind: "close", attempt, code: code ?? -1 });
      });
    });

    if (result.kind === "enoent") {
      if (result.attempt.fallbackMessage) {
        appendOutput(capture, result.attempt.fallbackMessage);
      }
      continue;
    }

    if (result.kind === "error") {
      appendOutput(capture, `\nFailed to start command: ${result.err.message}\n`);
      return {
        ok: false,
        command: result.attempt.label,
        exitCode: -1,
        output: capture.output,
      };
    }

    return {
      ok: result.code === 0,
      command: result.attempt.label,
      exitCode: result.code,
      output: capture.output,
    };
  }

  return {
    ok: false,
    command: `benchmark-qed ${args.join(" ")}`,
    exitCode: -1,
    output: capture.output,
  };
}

function startInitJob(body) {
  const allowedConfigTypes = [
    "autoq",
    "autoe_pairwise",
    "autoe_reference",
    "autoe_assertion",
  ];

  if (!allowedConfigTypes.includes(body.configType)) {
    throw new Error(`Invalid configType: ${body.configType}`);
  }
  if (!body.rootPath || typeof body.rootPath !== "string") {
    throw new Error("rootPath is required.");
  }

  const args = ["config", "init", body.configType, body.rootPath];
  if (body.storageType === "blob") {
    args.push("--storage-type", "blob");
    if (body.containerName) args.push("--container-name", body.containerName);
    if (body.accountUrl) args.push("--account-url", body.accountUrl);
    if (body.connectionString) {
      args.push("--connection-string", body.connectionString);
    }
    if (body.baseDir) args.push("--base-dir", body.baseDir);
  }

  const id = randomUUID();
  const job = {
    id,
    status: "running",
    startedAt: new Date().toISOString(),
    endedAt: null,
    rootPath: body.rootPath,
    configType: body.configType,
    storageType: body.storageType === "blob" ? "blob" : "local",
    command: `benchmark-qed ${args.join(" ")}`,
    output: "",
    exitCode: null,
  };
  initJobs.set(id, job);
  persistJobsSoon();
  runCommandWithFallback(job, args);

  return job;
}

function startRunJob(body) {
  const allowedConfigTypes = [
    "autoq",
    "autoe_pairwise",
    "autoe_reference",
    "autoe_assertion",
  ];

  if (!allowedConfigTypes.includes(body.configType)) {
    throw new Error(`Invalid configType: ${body.configType}`);
  }

  let settingsPath;
  let outputPath;
  let extraArgs = [];
  let displayRoot;

  if (body.blob && typeof body.blob === "object") {
    // Blob workspace: pass blob:// URIs to the CLI and authenticate via
    // --account-url (DefaultAzureCredential / az CLI auth) on the runner.
    const { accountUrl, containerName, prefix: basePrefix } = resolveBlobConfig(
      body.blob,
    );
    const settingsBlob = basePrefix
      ? `blob://${containerName}/${basePrefix}/settings.yaml`
      : `blob://${containerName}/settings.yaml`;
    settingsPath = settingsBlob;
    // output_data_path is treated as a child-path inside the configured
    // output_storage by autoq/autoe; "output" matches local convention.
    outputPath = "output";
    extraArgs = ["--account-url", accountUrl];
    displayRoot = basePrefix
      ? `blob://${containerName}/${basePrefix}`
      : `blob://${containerName}`;
  } else {
    if (!body.rootPath || typeof body.rootPath !== "string") {
      throw new Error("rootPath is required.");
    }
    settingsPath = path.join(body.rootPath, "settings.yaml");
    outputPath = path.join(body.rootPath, "output");
    displayRoot = body.rootPath;
  }

  const runArgsByType = {
    autoq: ["autoq", settingsPath, outputPath, ...extraArgs],
    autoe_pairwise: ["autoe", "pairwise-scores", settingsPath, outputPath, ...extraArgs],
    autoe_reference: ["autoe", "reference-scores", settingsPath, outputPath, ...extraArgs],
    autoe_assertion: ["autoe", "assertion-scores", settingsPath, outputPath, ...extraArgs],
  };

  const args = runArgsByType[body.configType];

  const id = randomUUID();
  const job = {
    id,
    status: "running",
    startedAt: new Date().toISOString(),
    endedAt: null,
    rootPath: displayRoot,
    configType: body.configType,
    command: `benchmark-qed ${args.join(" ")}`,
    output: "",
    exitCode: null,
  };
  runJobs.set(id, job);
  persistJobsSoon();
  runCommandWithFallback(job, args);
  return job;
}

async function parseBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf-8").trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

async function pickFolderPath() {
  if (process.platform === "darwin") {
    const { stdout } = await execFileAsync("osascript", [
      "-e",
      'POSIX path of (choose folder with prompt "Select benchmark root folder")',
    ]);
    const selected = stdout.trim();
    if (!selected) {
      throw new Error("No folder selected.");
    }
    return selected.replace(/\/$/, "");
  }

  if (process.platform === "win32") {
    // STA + WinForms FolderBrowserDialog. `__CANCELLED__` is our sentinel so
    // we can distinguish a user cancel from an empty selection.
    const script =
      "Add-Type -AssemblyName System.Windows.Forms;" +
      "$f = New-Object System.Windows.Forms.FolderBrowserDialog;" +
      "$f.Description = 'Select benchmark root folder';" +
      "if ($f.ShowDialog() -eq 'OK') { Write-Output $f.SelectedPath } else { Write-Output '__CANCELLED__' }";
    const { stdout } = await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-STA", "-Command", script],
      { timeout: 5 * 60 * 1000 },
    );
    const selected = stdout.trim();
    if (!selected || selected === "__CANCELLED__") {
      throw new Error("No folder selected.");
    }
    return selected.replace(/[\\/]+$/, "");
  }

  throw new Error(
    `Native folder picker is not supported on platform '${process.platform}'.`,
  );
}

async function pickFilesPaths() {
  if (process.platform === "darwin") {
    const { stdout } = await execFileAsync("osascript", [
      "-e",
      'set theFiles to choose file with prompt "Select dataset file(s) to import" with multiple selections allowed',
      "-e",
      'set out to ""',
      "-e",
      "repeat with f in theFiles",
      "-e",
      'set out to out & (POSIX path of f) & "\\n"',
      "-e",
      "end repeat",
      "-e",
      "return out",
    ]);
    const lines = stdout
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      throw new Error("No files selected.");
    }
    return lines;
  }

  if (process.platform === "win32") {
    const script =
      "Add-Type -AssemblyName System.Windows.Forms;" +
      "$o = New-Object System.Windows.Forms.OpenFileDialog;" +
      "$o.Title = 'Select dataset file(s) to import';" +
      "$o.Multiselect = $true;" +
      "if ($o.ShowDialog() -eq 'OK') { $o.FileNames | ForEach-Object { Write-Output $_ } } else { Write-Output '__CANCELLED__' }";
    const { stdout } = await execFileAsync(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-STA", "-Command", script],
      { timeout: 5 * 60 * 1000 },
    );
    const lines = stdout
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (lines.length === 0 || lines[0] === "__CANCELLED__") {
      throw new Error("No files selected.");
    }
    return lines;
  }

  throw new Error(
    `Native file picker is not supported on platform '${process.platform}'.`,
  );
}

const server = createServer(async (req, res) => {
  if (!req.url || !req.method) {
    json(req, res, 400, { error: "Bad request" });
    return;
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const pathname = url.pathname;

  if (req.method === "OPTIONS") {
    const corsOrigin = resolveCorsOrigin(req);
    res.writeHead(204, {
      "Access-Control-Allow-Origin": corsOrigin,
      "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  if (req.method === "GET" && pathname === "/health") {
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "POST" && pathname === "/api/init-jobs") {
    try {
      const body = await parseBody(req);
      const job = startInitJob(body);
      json(req, res, 202, job);
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/pick-folder") {
    try {
      const folderPath = await pickFolderPath();
      json(req, res, 200, { path: folderPath });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (
        /User canceled|\(-128\)|execution error: User canceled/i.test(message)
      ) {
        json(req, res, 200, { cancelled: true });
        return;
      }
      json(req, res, 400, {
        error: message,
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/pick-files") {
    try {
      const paths = await pickFilesPaths();
      json(req, res, 200, { paths });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (
        /User canceled|\(-128\)|execution error: User canceled/i.test(message)
      ) {
        json(req, res, 200, { cancelled: true });
        return;
      }
      json(req, res, 400, { error: message });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/init-jobs") {
    const data = Array.from(initJobs.values()).sort(
      (a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt),
    );
    json(req, res, 200, data);
    return;
  }

  if (req.method === "POST" && pathname === "/api/run-jobs") {
    try {
      const body = await parseBody(req);
      // For local workspaces, fail fast if settings.yaml is missing.
      // Blob workspaces skip this check; the Python CLI will surface a
      // clear error if the blob doesn't exist.
      if (!body.blob) {
        const settingsPath = path.join(body.rootPath ?? "", "settings.yaml");
        await fsp.access(settingsPath);
      }
      const job = startRunJob(body);
      json(req, res, 202, job);
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/datasets/download") {
    try {
      const body = await parseBody(req);
      const dataset = body.dataset;
      const outputPathInput = body.outputPath;
      const allowedDatasets = ["AP_news", "podcast", "example_answers"];

      if (!allowedDatasets.includes(dataset)) {
        throw new Error(`Invalid dataset: ${dataset}`);
      }

      const outputRelPath =
        typeof outputPathInput === "string" && outputPathInput.trim().length > 0
          ? outputPathInput.trim()
          : "input";

      let args;
      let outputPath; // for the success payload
      if (body.blob && typeof body.blob === "object") {
        const { accountUrl, containerName, prefix: basePrefix } = resolveBlobConfig(
          body.blob,
        );
        // Use --account-url (DefaultAzureCredential / az CLI auth) on the
        // runner so the storage backend can perform account-level operations
        // when needed.
        args = [
          "data",
          "download",
          dataset,
          outputRelPath,
          "--accept-terms",
          "--storage-type",
          "blob",
          "--container-name",
          containerName,
        ];
        args.push("--account-url", accountUrl);
        // The CLI joins base_dir + output_dir, so set base_dir to the
        // workspace prefix and pass the user-chosen subdir (e.g. "input")
        // as the second positional argument. Blobs land under
        // <prefix>/<outputRelPath>/...
        if (basePrefix) {
          args.push("--base-dir", basePrefix);
        }
        outputPath = basePrefix
          ? `blob://${containerName}/${basePrefix}/${outputRelPath}`
          : `blob://${containerName}/${outputRelPath}`;
      } else {
        const rootPath = body.rootPath;
        if (!rootPath || typeof rootPath !== "string") {
          throw new Error("rootPath is required.");
        }
        const { target } = resolveFsPath(rootPath, outputRelPath);
        outputPath = target;
        await fsp.mkdir(outputPath, { recursive: true });
        args = [
          "data",
          "download",
          dataset,
          outputPath,
          "--accept-terms",
        ];
      }

      const result = await runCommandWithFallbackAndCapture(args, {
        cwd: REPO_ROOT,
      });
      if (!result.ok) {
        const friendly = `Dataset download failed (exit code: ${result.exitCode}).`;
        json(req, res, 400, {
          error: friendly,
          command: result.command,
          output: result.output,
        });
        return;
      }

      json(req, res, 200, {
        ok: true,
        dataset,
        outputPath,
        command: result.command,
        output: result.output,
      });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/datasets/import") {
    try {
      const body = await parseBody(req);
      const destinationFolder = body.destinationFolder;
      const sourcePaths = body.sourcePaths;
      const flatten = body.flatten === true;
      const subdir =
        typeof body.subdir === "string" && body.subdir.trim().length > 0
          ? body.subdir.trim()
          : "input";

      if (!destinationFolder || typeof destinationFolder !== "string") {
        throw new Error("destinationFolder is required.");
      }
      if (!Array.isArray(sourcePaths) || sourcePaths.length === 0) {
        throw new Error("sourcePaths must be a non-empty array.");
      }
      for (const p of sourcePaths) {
        if (typeof p !== "string" || !path.isAbsolute(p)) {
          throw new Error(`sourcePaths entries must be absolute paths. Got: ${p}`);
        }
      }

      const destRoot = path.resolve(destinationFolder);
      const { target: destInput } = resolveFsPath(destRoot, subdir);
      await fsp.mkdir(destInput, { recursive: true });

      // Expand sourcePaths: when flatten is true, replace any directory entry
      // with its immediate children so they land directly under destInput.
      const expanded = [];
      for (const src of sourcePaths) {
        if (flatten) {
          try {
            const stat = await fsp.stat(src);
            if (stat.isDirectory()) {
              const entries = await fsp.readdir(src);
              for (const name of entries) {
                expanded.push(path.join(src, name));
              }
              continue;
            }
          } catch {
            // fall through; the copy loop will record the error
          }
        }
        expanded.push(src);
      }

      const imported = [];
      const skipped = [];
      for (const src of expanded) {
        try {
          const stat = await fsp.stat(src);
          const base = path.basename(src);
          const target = path.join(destInput, base);
          if (stat.isDirectory()) {
            await fsp.cp(src, target, { recursive: true, force: true });
          } else {
            await fsp.copyFile(src, target);
          }
          imported.push({ source: src, target });
        } catch (e) {
          skipped.push({
            source: src,
            error: e instanceof Error ? e.message : String(e),
          });
        }
      }

      if (imported.length === 0) {
        json(req, res, 400, {
          error: "No files were imported.",
          skipped,
        });
        return;
      }

      json(req, res, 200, {
        ok: true,
        destinationFolder: destRoot,
        inputFolder: destInput,
        imported,
        skipped,
      });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/run-jobs") {
    const data = Array.from(runJobs.values()).sort(
      (a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt),
    );
    json(req, res, 200, data);
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/init-jobs") {
    initJobs.clear();
    persistJobsSoon();
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/run-jobs") {
    runJobs.clear();
    persistJobsSoon();
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "DELETE" && pathname.startsWith("/api/init-jobs/")) {
    const id = pathname.split("/").pop();
    const job = id ? initJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    if (job.status === "running") {
      json(req, res, 400, { error: "Cannot delete a running job." });
      return;
    }
    initJobs.delete(id);
    persistJobsSoon();
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "DELETE" && pathname.startsWith("/api/run-jobs/")) {
    const id = pathname.split("/").pop();
    const job = id ? runJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    if (job.status === "running") {
      json(req, res, 400, { error: "Cannot delete a running job. Cancel it first." });
      return;
    }
    runJobs.delete(id);
    persistJobsSoon();
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "GET" && pathname.startsWith("/api/init-jobs/")) {
    const id = pathname.split("/").pop();
    const job = id ? initJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    json(req, res, 200, job);
    return;
  }

  if (req.method === "GET" && pathname.startsWith("/api/run-jobs/")) {
    const id = pathname.split("/").pop();
    const job = id ? runJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    json(req, res, 200, job);
    return;
  }

  if (req.method === "POST" && pathname.startsWith("/api/run-jobs/") && pathname.endsWith("/cancel")) {
    const id = pathname.split("/").slice(-2, -1)[0];
    const job = id ? runJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    if (job.status !== "running") {
      json(req, res, 400, { error: "Job is not running" });
      return;
    }
    const cancelled = cancelJob(id);
    if (cancelled) {
      job.status = "cancelled";
      job.endedAt = new Date().toISOString();
      job.exitCode = -1;
      appendOutput(job, "\nJob cancelled by user.\n");
      persistJobsSoon();
      json(req, res, 200, { ok: true });
    } else {
      json(req, res, 400, { error: "Failed to cancel job or already finished." });
    }
    return;
  }

  if (req.method === "POST" && pathname.startsWith("/api/run-jobs/") && pathname.endsWith("/rerun")) {
    const id = pathname.split("/").slice(-2, -1)[0];
    const job = id ? runJobs.get(id) : undefined;
    if (!job) {
      json(req, res, 404, { error: "Job not found" });
      return;
    }
    if (job.status === "running") {
      json(req, res, 400, { error: "Job is already running." });
      return;
    }
    if (!job.rootPath || !job.configType) {
      json(req, res, 400, {
        error: "Job is missing rootPath or configType; cannot re-run.",
      });
      return;
    }

    const settingsPath = path.join(job.rootPath, "settings.yaml");
    const outputPath = path.join(job.rootPath, "output");
    const runArgsByType = {
      autoq: ["autoq", settingsPath, outputPath],
      autoe_pairwise: ["autoe", "pairwise-scores", settingsPath, outputPath],
      autoe_reference: ["autoe", "reference-scores", settingsPath, outputPath],
      autoe_assertion: ["autoe", "assertion-scores", settingsPath, outputPath],
    };
    const args = runArgsByType[job.configType];
    if (!args) {
      json(req, res, 400, { error: `Unknown configType: ${job.configType}` });
      return;
    }

    // Reset the existing job in place so the UI sees the same id with a
    // fresh run rather than a new history entry.
    job.status = "running";
    job.startedAt = new Date().toISOString();
    job.endedAt = null;
    job.exitCode = null;
    job.output = "";
    persistJobsSoon();
    runCommandWithFallback(job, args);
    json(req, res, 202, job);
    return;
  }

  if (req.method === "GET" && pathname === "/api/fs/list") {
    try {
      const rootPath = url.searchParams.get("root") ?? "";
      const relPath = url.searchParams.get("path") ?? "";
      const { root, target } = resolveFsPath(rootPath, relPath);
      const entries = await fsp.readdir(target, { withFileTypes: true });
      const nodes = entries
        .filter((entry) => entry.isDirectory() || entry.isFile())
        .map((entry) => ({
          name: entry.name,
          path: relPath ? `${relPath}/${entry.name}` : entry.name,
          kind: entry.isDirectory() ? "directory" : "file",
        }));
      json(req, res, 200, { root, path: relPath, nodes });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/blob/list") {
    try {
      const accountUrl = url.searchParams.get("accountUrl") ?? "";
      const containerName = url.searchParams.get("containerName") ?? "";
      const prefix = url.searchParams.get("prefix") ?? "";
      const relPath = url.searchParams.get("path") ?? "";
      if (!accountUrl || !containerName) {
        throw new Error("accountUrl and containerName are required.");
      }
      const client = getBlobContainerClient(accountUrl, containerName);
      const fullPrefix = blobFullPath(prefix, relPath);
      const iter = client.listBlobsByHierarchy("/", {
        prefix: fullPrefix ? `${fullPrefix}/` : "",
      });
      const basePath = relPath.replace(/^\/+|\/+$/g, "");
      const nodes = [];
      for await (const item of iter) {
        if (item.kind === "prefix") {
          const full = item.name.endsWith("/") ? item.name.slice(0, -1) : item.name;
          const name = fullPrefix ? full.slice(fullPrefix.length + 1) : full;
          if (!name) continue;
          nodes.push({
            name,
            path: basePath ? `${basePath}/${name}` : name,
            kind: "directory",
          });
        } else {
          const full = item.name;
          const name = fullPrefix ? full.slice(fullPrefix.length + 1) : full;
          if (!name) continue;
          nodes.push({
            name,
            path: basePath ? `${basePath}/${name}` : name,
            kind: "file",
          });
        }
      }
      json(req, res, 200, { nodes });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/fs/read") {
    try {
      const rootPath = url.searchParams.get("root") ?? "";
      const relPath = url.searchParams.get("path") ?? "";
      if (!relPath) throw new Error("path is required.");
      const { target } = resolveFsPath(rootPath, relPath);
      const content = await fsp.readFile(target, "utf-8");
      json(req, res, 200, { content });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "GET" && pathname === "/api/blob/read") {
    try {
      const accountUrl = url.searchParams.get("accountUrl") ?? "";
      const containerName = url.searchParams.get("containerName") ?? "";
      const prefix = url.searchParams.get("prefix") ?? "";
      const relPath = url.searchParams.get("path") ?? "";
      if (!accountUrl || !containerName || !relPath) {
        throw new Error("accountUrl, containerName, and path are required.");
      }
      const client = getBlobContainerClient(accountUrl, containerName);
      const blobClient = client.getBlobClient(blobFullPath(prefix, relPath));
      const content = await readBlobText(blobClient);
      json(req, res, 200, { content });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "PUT" && pathname === "/api/fs/write") {
    try {
      const body = await parseBody(req);
      const rootPath = body.root;
      const relPath = body.path;
      const content = String(body.content ?? "");
      if (!rootPath || !relPath) {
        throw new Error("root and path are required.");
      }
      const { target } = resolveFsPath(rootPath, relPath);
      await fsp.mkdir(path.dirname(target), { recursive: true });
      await fsp.writeFile(target, content, "utf-8");
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "PUT" && pathname === "/api/blob/write") {
    try {
      const body = await parseBody(req);
      const { accountUrl, containerName, prefix, path: relPath } = body;
      const content = String(body.content ?? "");
      if (!accountUrl || !containerName || !relPath) {
        throw new Error("accountUrl, containerName, and path are required.");
      }
      const client = getBlobContainerClient(accountUrl, containerName);
      const blob = client.getBlockBlobClient(blobFullPath(prefix, relPath));
      await blob.upload(content, Buffer.byteLength(content, "utf-8"), {
        blobHTTPHeaders: { blobContentType: "text/plain; charset=utf-8" },
      });
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/fs/create-file") {
    try {
      const body = await parseBody(req);
      const rootPath = body.root;
      const relPath = body.path;
      const content = String(body.content ?? "");
      if (!rootPath || !relPath) {
        throw new Error("root and path are required.");
      }
      const { target } = resolveFsPath(rootPath, relPath);
      await fsp.mkdir(path.dirname(target), { recursive: true });
      await fsp.writeFile(target, content, "utf-8");
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/blob/create-file") {
    try {
      const body = await parseBody(req);
      const { accountUrl, containerName, prefix, path: relPath } = body;
      const content = String(body.content ?? "");
      if (!accountUrl || !containerName || !relPath) {
        throw new Error("accountUrl, containerName, and path are required.");
      }
      const client = getBlobContainerClient(accountUrl, containerName);
      const blob = client.getBlockBlobClient(blobFullPath(prefix, relPath));
      await blob.upload(content, Buffer.byteLength(content, "utf-8"), {
        blobHTTPHeaders: { blobContentType: "text/plain; charset=utf-8" },
      });
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/fs/create-folder") {
    try {
      const body = await parseBody(req);
      const rootPath = body.root;
      const relPath = body.path;
      if (!rootPath || !relPath) {
        throw new Error("root and path are required.");
      }
      const { target } = resolveFsPath(rootPath, relPath);
      await fsp.mkdir(target, { recursive: true });
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/fs/delete") {
    try {
      const body = await parseBody(req);
      const rootPath = body.root;
      const relPath = body.path;
      if (!rootPath || !relPath) {
        throw new Error("root and path are required.");
      }
      const { target } = resolveFsPath(rootPath, relPath);
      await fsp.rm(target, { recursive: true, force: false });
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  // Rename (or move) a file or folder inside a workspace. The new name may
  // include forward slashes to move within the workspace; both source and
  // destination must stay inside the workspace root.
  if (req.method === "POST" && pathname === "/api/fs/rename") {
    try {
      const body = await parseBody(req);
      const rootPath = body.root;
      const relPath = body.path;
      const newName = body.newName;
      if (!rootPath || !relPath || !newName) {
        throw new Error("root, path, and newName are required.");
      }
      const trimmedName = String(newName)
        .trim()
        .replace(/^[/\\]+|[/\\]+$/g, "");
      if (!trimmedName) {
        throw new Error("newName cannot be empty.");
      }
      const { root, target: source } = resolveFsPath(rootPath, relPath);
      const parentRel = path.posix.dirname(relPath.replace(/\\/g, "/"));
      const destRel =
        parentRel && parentRel !== "."
          ? `${parentRel}/${trimmedName}`
          : trimmedName;
      const { target: dest } = resolveFsPath(rootPath, destRel);
      if (source === dest) {
        json(req, res, 200, { ok: true, path: destRel });
        return;
      }
      const srcStat = await fsp.stat(source).catch(() => null);
      if (!srcStat) {
        throw new Error(`Source not found: ${relPath}`);
      }
      const destStat = await fsp.stat(dest).catch(() => null);
      if (destStat) {
        throw new Error(
          `Destination already exists: ${destRel}. Choose a different name.`,
        );
      }
      // Disallow renaming the workspace root itself.
      if (source === root) {
        throw new Error("Cannot rename the workspace root folder.");
      }
      await fsp.mkdir(path.dirname(dest), { recursive: true });
      await fsp.rename(source, dest);
      json(req, res, 200, { ok: true, path: destRel });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/blob/delete") {
    try {
      const body = await parseBody(req);
      const { accountUrl, containerName, prefix, path: relPath } = body;
      if (!accountUrl || !containerName || !relPath) {
        throw new Error("accountUrl, containerName, and path are required.");
      }
      const client = getBlobContainerClient(accountUrl, containerName);
      const blobName = blobFullPath(prefix, relPath);
      const single = client.getBlobClient(blobName);
      const resDelete = await single.deleteIfExists();
      if (!resDelete.succeeded) {
        const folderPrefix = blobName.endsWith("/") ? blobName : `${blobName}/`;
        for await (const item of client.listBlobsFlat({ prefix: folderPrefix })) {
          await client.getBlobClient(item.name).deleteIfExists();
        }
      }
      json(req, res, 200, { ok: true });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  // Recursively copy an entire workspace folder (or any absolute folder) to a
  // new absolute destination path. Used by the "Copy workspace" action in the
  // UI so users can fork a workspace and load the copy as a new tab.
  if (req.method === "POST" && pathname === "/api/fs/copy-path") {
    try {
      const body = await parseBody(req);
      const sourcePath = body.sourcePath;
      const destPath = body.destPath;
      const overwrite = Boolean(body.overwrite);
      if (!sourcePath || typeof sourcePath !== "string") {
        throw new Error("sourcePath is required.");
      }
      if (!destPath || typeof destPath !== "string") {
        throw new Error("destPath is required.");
      }
      const absSource = path.resolve(sourcePath);
      const absDest = path.resolve(destPath);
      const srcStat = await fsp.stat(absSource).catch(() => null);
      if (!srcStat || !srcStat.isDirectory()) {
        throw new Error(`Source folder not found: ${absSource}`);
      }
      if (absDest === absSource) {
        throw new Error("Destination must differ from source.");
      }
      if (
        absDest.startsWith(`${absSource}${path.sep}`) ||
        absDest === absSource
      ) {
        throw new Error("Destination cannot be inside the source folder.");
      }
      const destStat = await fsp.stat(absDest).catch(() => null);
      if (destStat && !overwrite) {
        throw new Error(
          `Destination already exists: ${absDest}. Choose a different path.`,
        );
      }
      await fsp.mkdir(path.dirname(absDest), { recursive: true });
      await fsp.cp(absSource, absDest, {
        recursive: true,
        force: overwrite,
        errorOnExist: !overwrite,
      });
      json(req, res, 200, { ok: true, path: absDest });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/detect-configs") {
    try {
      const body = await parseBody(req);
      const rootPath = body.rootPath;
      if (!rootPath || typeof rootPath !== "string") {
        throw new Error("rootPath is required.");
      }

      const detectedConfigs = [];
      const root = path.resolve(rootPath);
      const entries = await fsp.readdir(root, { withFileTypes: true });

      // Check if root itself has a settings.yaml
      try {
        await fsp.access(path.join(root, "settings.yaml"));
        const settingsContent = await fsp.readFile(path.join(root, "settings.yaml"), "utf-8");
        const configType = detectConfigType(settingsContent);
        if (configType) {
          detectedConfigs.push({
            path: rootPath,
            name: path.basename(rootPath),
            configType,
            isRoot: true,
          });
        }
      } catch {
        // Root doesn't have settings.yaml, continue
      }

      // Check subfolders for settings.yaml
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const subfolderPath = path.join(root, entry.name);
        const settingsPath = path.join(subfolderPath, "settings.yaml");
        try {
          await fsp.access(settingsPath);
          const settingsContent = await fsp.readFile(settingsPath, "utf-8");
          const configType = detectConfigType(settingsContent);
          if (configType) {
            detectedConfigs.push({
              path: subfolderPath,
              name: entry.name,
              configType,
              isRoot: false,
            });
          }
        } catch {
          // Subfolder doesn't have settings.yaml, skip
        }
      }

      json(req, res, 200, { configs: detectedConfigs });
    } catch (err) {
      json(req, res, 400, {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  json(req, res, 404, { error: "Not found" });
});

function detectConfigType(yamlContent) {
  // Try to detect config type from YAML content
  // Look for key indicators of each config type
  if (/^\s*base:\s*$|^\s*others:\s*[\[\-]|score_min:|score_max:|pairwise/mi.test(yamlContent)) {
    return "autoe_pairwise";
  }
  if (/^\s*reference:\s*$|^\s*generated:\s*[\[\-]|^\s*- name:|reference_base_path/mi.test(yamlContent)) {
    return "autoe_reference";
  }
  if (/^\s*generated:\s*$|^\s*assertions:\s*$|assertions_path:|pass_threshold:|detect_discovery/mi.test(yamlContent)) {
    return "autoe_assertion";
  }
  if (/^\s*input:\s*$|^\s*output:/mi.test(yamlContent)) {
    return "autoq";
  }
  // Default fallback based on presence of common fields
  if (/input|output|questions/mi.test(yamlContent)) {
    return "autoq";
  }
  return null;
}

server.on("error", (err) => {
  if (err && typeof err === "object" && "code" in err && err.code === "EADDRINUSE") {
    console.log(`Init runner already running on http://localhost:${PORT}`);
    process.exit(0);
  }
  console.error(err);
  process.exit(1);
});

server.listen(PORT, () => {
  console.log(`Init runner listening on http://localhost:${PORT}`);
  console.log(`Repo root: ${REPO_ROOT}`);
});
