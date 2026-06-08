// HTTP + SSE server that fronts the CopilotManager. Designed to run alongside
// init-runner (default port 8788, init-runner is 8787).
//
// Endpoints:
//   GET  /health
//   GET  /api/copilot/auth/status
//   POST /api/copilot/sessions             -> { sessionId }
//   GET  /api/copilot/sessions/:id/events  -> SSE stream
//   POST /api/copilot/sessions/:id/respond -> 204
//   POST /api/copilot/sessions/:id/message -> 204
//   DELETE /api/copilot/sessions/:id       -> 204

import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";

import { SseBus } from "./sseBus.mjs";
import { PendingRequests } from "./pendingRequests.mjs";
import { CopilotManager } from "./copilotManager.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const PORT = Number(process.env.COPILOT_BRIDGE_PORT || 8788);
const ALLOWED_ORIGIN =
  process.env.COPILOT_BRIDGE_ORIGIN || "http://localhost:5173";

const execFileAsync = promisify(execFile);

const bus = new SseBus();
const pending = new PendingRequests();
const manager = new CopilotManager({ repoRoot: REPO_ROOT, bus, pending });

function resolveCors(req) {
  const origin = req.headers.origin;
  if (!origin) return ALLOWED_ORIGIN;
  if (/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return origin;
  return ALLOWED_ORIGIN;
}

function corsHeaders(req) {
  return {
    "Access-Control-Allow-Origin": resolveCors(req),
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(req, res, status, payload) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    ...corsHeaders(req),
  });
  res.end(JSON.stringify(payload));
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new Error("Invalid JSON body");
  }
}

async function checkAuthStatus() {
  // Copilot CLI syntax differs across versions:
  // - older: `copilot auth status`
  // - newer: `copilot -p "auth status"` (or `-i` for interactive mode)
  // Try both non-interactive command forms so the UI can detect auth state
  // regardless of installed CLI version.
  // On Windows the globally installed binary is `copilot.cmd`, which Node's
  // execFile can only invoke through the shell — hence the `shell` flag.
  const opts = {
    timeout: 5000,
    shell: process.platform === "win32",
  };
  const attempts = [
    ["auth", "status"],
    ["-p", "auth status"],
  ];

  let lastErr;
  for (const args of attempts) {
    try {
      const { stdout, stderr } = await execFileAsync("copilot", args, opts);
      const detail = `${stdout || ""}${stderr || ""}`.trim();
      if (!detail) {
        return { ok: true, detail: "Copilot CLI responded." };
      }
      const lower = detail.toLowerCase();
      const unauthenticated =
        lower.includes("not authenticated") ||
        lower.includes("run 'copilot login'") ||
        lower.includes("run \"copilot login\"");
      return { ok: !unauthenticated, detail };
    } catch (err) {
      lastErr = err;
    }
  }

  return {
    ok: false,
    detail:
      lastErr?.stderr?.toString().trim() ||
      lastErr?.message ||
      "Copilot CLI not found or not logged in. Run `copilot login`.",
  };
}

function openSseStream(req, res, sessionId) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    ...corsHeaders(req),
  });
  res.write(`retry: 2000\n\n`);

  // Replay any still-pending requests so a reconnecting tab sees them too.
  for (const entry of manager.pendingFor(sessionId)) {
    res.write(
      `event: ${entry.kind}.request\ndata: ${JSON.stringify({
        type: `${entry.kind}.request`,
        requestId: entry.requestId,
        data: entry.payload,
        replay: true,
      })}\n\n`,
    );
  }

  const unsubscribe = bus.subscribe(sessionId, (event) => {
    const eventName = (event.type || "message").replace(/[^a-zA-Z0-9._-]/g, "_");
    res.write(`event: ${eventName}\ndata: ${JSON.stringify(event)}\n\n`);
  });

  const keepAlive = setInterval(() => res.write(`: keep-alive\n\n`), 25_000);

  const cleanup = () => {
    clearInterval(keepAlive);
    unsubscribe();
    try {
      res.end();
    } catch {
      /* ignore */
    }
  };
  req.on("close", cleanup);
  req.on("aborted", cleanup);
}

const server = createServer(async (req, res) => {
  if (!req.url || !req.method) return json(req, res, 400, { error: "Bad request" });

  if (req.method === "OPTIONS") {
    res.writeHead(204, corsHeaders(req));
    return res.end();
  }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const { pathname } = url;

  try {
    if (req.method === "GET" && pathname === "/health") {
      return json(req, res, 200, { ok: true });
    }

    if (req.method === "GET" && pathname === "/api/copilot/auth/status") {
      const status = await checkAuthStatus();
      return json(req, res, 200, status);
    }

    if (req.method === "POST" && pathname === "/api/copilot/sessions") {
      const body = await readBody(req);
      const { sessionId } = await manager.createSession({
        initialPrompt: body.initialPrompt,
        silentInitialPrompt: body.silentInitialPrompt,
        model: body.model,
        skillDirectories: body.skillDirectories,
      });
      return json(req, res, 201, { sessionId });
    }

    const sessionMatch = pathname.match(
      /^\/api\/copilot\/sessions\/([^/]+)(?:\/(events|respond|message))?$/,
    );
    if (sessionMatch) {
      const sessionId = sessionMatch[1];
      const action = sessionMatch[2];

      if (req.method === "GET" && action === "events") {
        return openSseStream(req, res, sessionId);
      }
      if (req.method === "POST" && action === "respond") {
        const body = await readBody(req);
        if (!body.requestId || !body.kind) {
          return json(req, res, 400, {
            error: "requestId and kind are required",
          });
        }
        manager.respond(sessionId, body.requestId, body.value ?? {});
        res.writeHead(204, corsHeaders(req));
        return res.end();
      }
      if (req.method === "POST" && action === "message") {
        const body = await readBody(req);
        if (!body.prompt) {
          return json(req, res, 400, { error: "prompt is required" });
        }
        await manager.sendMessage(sessionId, body.prompt);
        res.writeHead(204, corsHeaders(req));
        return res.end();
      }
      if (req.method === "DELETE" && !action) {
        await manager.disconnect(sessionId);
        res.writeHead(204, corsHeaders(req));
        return res.end();
      }
    }

    return json(req, res, 404, { error: "Not found" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return json(req, res, 400, { error: message });
  }
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(
    `[copilot-bridge] listening on http://localhost:${PORT} (origin ${ALLOWED_ORIGIN})`,
  );
});

async function shutdown(signal) {
  // eslint-disable-next-line no-console
  console.log(`[copilot-bridge] received ${signal}, shutting down...`);
  try {
    await manager.shutdown();
  } finally {
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 5000).unref();
  }
}
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
