import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { promisify } from "node:util";
import * as fs from "node:fs";
import * as fsp from "node:fs/promises";

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

// Minimal terminal-style processor so in-place repaints (rich progress bars,
// spinners, etc.) overwrite the previous frame instead of accumulating as
// duplicated text. Handles \r, CSI cursor-up (\x1b[<n>A), and CSI erase-line
// (\x1b[K / \x1b[2K). All other ANSI escapes are stripped.
function appendOutput(job, chunk) {
  const txt = String(chunk ?? "");
  let buf = job.output;

  const eraseCurrentLine = () => {
    const nl = buf.lastIndexOf("\n");
    buf = nl === -1 ? "" : buf.slice(0, nl + 1);
  };

  let i = 0;
  while (i < txt.length) {
    const ch = txt[i];
    if (ch === "\r") {
      // Carriage return: rewind to start of current line.
      // Treat \r\n as a regular newline.
      if (txt[i + 1] === "\n") {
        buf += "\n";
        i += 2;
      } else {
        eraseCurrentLine();
        i += 1;
      }
    } else if (ch === "\x1b" && txt[i + 1] === "[") {
      // CSI sequence: ESC [ params final
      let j = i + 2;
      let params = "";
      while (j < txt.length && /[0-9;]/.test(txt[j])) {
        params += txt[j];
        j += 1;
      }
      const final = txt[j];
      if (final === undefined) {
        // Incomplete escape at chunk boundary — drop it; subsequent chunk
        // would otherwise resync anyway.
        break;
      }
      if (final === "A") {
        // Cursor up N (default 1): clear current line, then drop N prior lines.
        const n = Math.max(1, parseInt(params || "1", 10) || 1);
        eraseCurrentLine();
        for (let k = 0; k < n; k += 1) {
          if (buf.endsWith("\n")) buf = buf.slice(0, -1);
          eraseCurrentLine();
        }
      } else if (final === "K" || final === "J") {
        // Erase in line / display — collapse to clearing current line.
        eraseCurrentLine();
      }
      // All other CSI sequences (colors, cursor moves, etc.) are stripped.
      i = j + 1;
    } else if (ch === "\x1b") {
      // Other ESC sequence (e.g. ESC ] OSC, ESC ( charset). Skip ESC + next.
      i += txt[i + 1] ? 2 : 1;
    } else {
      buf += ch;
      i += 1;
    }
  }

  // Collapse consecutive progress-bar style lines (rich / tqdm) by label so
  // we don't accumulate hundreds of redraws when stdout isn't a TTY.
  buf = collapseProgressLines(buf);

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

function spawnChild(cmd, cmdArgs) {
  return spawn(cmd, cmdArgs, {
    cwd: REPO_ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    env: CHILD_ENV,
  });
}

function spawnChildInCwd(cmd, cmdArgs, cwd = REPO_ROOT) {
  return spawn(cmd, cmdArgs, {
    cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: CHILD_ENV,
  });
}

// Detect once which invocation of the benchmark-qed CLI works on this machine.
// Avoids spamming the job log with "'benchmark-qed' not found in PATH" on every
// run. Result is cached for the lifetime of the runner process.
let _resolvedBenchmarkInvocation = null;
function which(cmd) {
  try {
    const res = spawnSync("/bin/sh", ["-c", `command -v ${cmd}`], {
      env: CHILD_ENV,
      encoding: "utf8",
      timeout: 5000,
    });
    if (res.status === 0) {
      const out = (res.stdout || "").trim();
      if (out) return out.split("\n")[0];
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
      const res = spawnSync(cmd, args, {
        cwd: REPO_ROOT,
        env: CHILD_ENV,
        stdio: "ignore",
        timeout: 15000,
      });
      return res.status === 0;
    } catch {
      return false;
    }
  };

  const candidates = [
    { cmd: "benchmark-qed", prefix: [], label: "benchmark-qed", probe: ["--help"] },
    { cmd: "uv", prefix: ["run", "benchmark-qed"], label: "uv run benchmark-qed", probe: ["run", "benchmark-qed", "--help"] },
    { cmd: defaultPython, prefix: ["-m", "benchmark_qed"], label: `${defaultPython} -m benchmark_qed`, probe: ["-m", "benchmark_qed", "--help"] },
  ];

  for (const c of candidates) {
    if (probe(c.cmd, c.probe)) {
      const absPath = which(c.cmd) || c.cmd;
      _resolvedBenchmarkInvocation = { cmd: c.cmd, absPath, prefix: c.prefix, label: c.label };
      // eslint-disable-next-line no-console
      console.log(`[init-runner] Using '${c.label}' (${absPath}) for benchmark-qed CLI.`);
      return _resolvedBenchmarkInvocation;
    }
  }

  // Nothing worked at probe time; fall back to plain `benchmark-qed` and let
  // the spawn error surface to the user.
  _resolvedBenchmarkInvocation = { cmd: "benchmark-qed", absPath: "benchmark-qed", prefix: [], label: "benchmark-qed" };
  return _resolvedBenchmarkInvocation;
}

function runCommandWithFallback(job, args) {
  const { absPath, prefix, label } = resolveBenchmarkInvocation();
  const cmdArgs = [...prefix, ...args];
  const commandLabel = `${label} ${args.join(" ")}`;
  job.command = commandLabel;

  const markCompleted = (code) => {
    job.endedAt = new Date().toISOString();
    job.exitCode = code ?? -1;
    job.status = code === 0 ? "succeeded" : "failed";
    if (jobProcesses.has(job.id)) {
      jobProcesses.delete(job.id);
    }
  };

  // Spawn through a pseudo-terminal so that rich/tqdm detect a real TTY and
  // emit their proper redrawing progress bars (with cursor-up ANSI codes,
  // which appendOutput already interprets).
  let proc;
  try {
    proc = pty.spawn(absPath, cmdArgs, {
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

  proc.onData((data) => appendOutput(job, data));
  proc.onExit(({ exitCode }) => markCompleted(exitCode));
}
// Cancel a running job by id
function cancelJob(jobId) {
  const proc = jobProcesses.get(jobId);
  if (proc && typeof proc.kill === "function") {
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
    command: `benchmark-qed ${args.join(" ")}`,
    output: "",
    exitCode: null,
  };
  initJobs.set(id, job);
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
  if (!body.rootPath || typeof body.rootPath !== "string") {
    throw new Error("rootPath is required.");
  }

  const settingsPath = path.join(body.rootPath, "settings.yaml");
  const outputPath = path.join(body.rootPath, "output");

  const runArgsByType = {
    autoq: ["autoq", settingsPath, outputPath],
    autoe_pairwise: ["autoe", "pairwise-scores", settingsPath, outputPath],
    autoe_reference: ["autoe", "reference-scores", settingsPath, outputPath],
    autoe_assertion: ["autoe", "assertion-scores", settingsPath, outputPath],
  };

  const args = runArgsByType[body.configType];

  const id = randomUUID();
  const job = {
    id,
    status: "running",
    startedAt: new Date().toISOString(),
    endedAt: null,
    rootPath: body.rootPath,
    configType: body.configType,
    command: `benchmark-qed ${args.join(" ")}`,
    output: "",
    exitCode: null,
  };
  runJobs.set(id, job);
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
  if (process.platform !== "darwin") {
    throw new Error("Native folder picker is currently supported on macOS only.");
  }

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

async function pickFilesPaths() {
  if (process.platform !== "darwin") {
    throw new Error("Native file picker is currently supported on macOS only.");
  }

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
      const settingsPath = path.join(body.rootPath ?? "", "settings.yaml");
      await fsp.access(settingsPath);
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
      const rootPath = body.rootPath;
      const dataset = body.dataset;
      const outputPathInput = body.outputPath;
      const allowedDatasets = ["AP_news", "podcast", "example_answers"];

      if (!rootPath || typeof rootPath !== "string") {
        throw new Error("rootPath is required.");
      }
      if (!allowedDatasets.includes(dataset)) {
        throw new Error(`Invalid dataset: ${dataset}`);
      }

      const outputRelPath =
        typeof outputPathInput === "string" && outputPathInput.trim().length > 0
          ? outputPathInput.trim()
          : "input";
      const { target: outputPath } = resolveFsPath(rootPath, outputRelPath);
      const outputPathArg = path.relative(rootPath, outputPath) || ".";
      await fsp.mkdir(outputPath, { recursive: true });

      const args = [
        "data",
        "download",
        dataset,
        outputPath,
        "--accept-terms",
      ];
      const result = await runCommandWithFallbackAndCapture(args, {
        cwd: REPO_ROOT,
      });
      if (!result.ok) {
        json(req, res, 400, {
          error: `Dataset download failed (exit code: ${result.exitCode}).`,
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
    json(req, res, 200, { ok: true });
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/run-jobs") {
    runJobs.clear();
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
      json(req, res, 200, { ok: true });
    } else {
      json(req, res, 400, { error: "Failed to cancel job or already finished." });
    }
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
