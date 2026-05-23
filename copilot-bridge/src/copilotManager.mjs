// Wraps @github/copilot-sdk to host one CopilotClient and many sessions, with
// every SDK callback (assistant deltas, ask_user, elicitation, permission)
// pushed onto the per-session SSE bus and parked as a pending request when the
// agent expects an answer.

import path from "node:path";

let sdkPromise;
async function loadSdk() {
  if (!sdkPromise) sdkPromise = import("@github/copilot-sdk");
  return sdkPromise;
}

export class CopilotManager {
  /**
   * @param {object} opts
   * @param {string} opts.repoRoot   Workspace root (for default skill directory).
   * @param {import('./sseBus.mjs').SseBus} opts.bus
   * @param {import('./pendingRequests.mjs').PendingRequests} opts.pending
   */
  constructor({ repoRoot, bus, pending }) {
    this.repoRoot = repoRoot;
    this.bus = bus;
    this.pending = pending;
    /** @type {import('@github/copilot-sdk').CopilotClient | null} */
    this.client = null;
    /** @type {Map<string, import('@github/copilot-sdk').CopilotSession>} */
    this.sessions = new Map();
    /** @type {Map<string, { status: 'idle'|'running'|'error', error?: string, model: string, createdAt: number }>} */
    this.meta = new Map();
  }

  async ensureClient() {
    if (this.client) return this.client;
    const { CopilotClient } = await loadSdk();
    this.client = new CopilotClient({
      useLoggedInUser: true,
      logLevel: process.env.COPILOT_LOG_LEVEL || "info",
    });
    await this.client.start();
    return this.client;
  }

  /**
   * Create a new agent session. The browser-facing handlers all return promises
   * fulfilled when the UI posts an answer via /respond.
   *
   * @param {{ initialPrompt?: string, model?: string, skillDirectories?: string[] }} cfg
   * @returns {Promise<{ sessionId: string }>}
   */
  async createSession(cfg = {}) {
    const client = await this.ensureClient();
    const skillDirectories = cfg.skillDirectories ?? [
      path.join(this.repoRoot, ".apm", "skills"),
    ];
    // Resolve model: explicit > env override > SDK default (omit field).
    const model = cfg.model || process.env.COPILOT_MODEL || undefined;

    const sessionOpts = {
      streaming: true,
      skillDirectories,
    };
    if (model) sessionOpts.model = model;

    const session = await client.createSession({
      ...sessionOpts,

      onUserInputRequest: async (request) => {
        // request: { question, choices?, allowFreeform? }
        const { requestId, promise } = this.pending.create(
          session.sessionId,
          "user_input",
          {
            question: request.question,
            choices: request.choices,
            allowFreeform: request.allowFreeform ?? true,
          },
        );
        this.bus.publish(session.sessionId, {
          type: "user_input.request",
          requestId,
          data: {
            question: request.question,
            choices: request.choices,
            allowFreeform: request.allowFreeform ?? true,
          },
        });
        const reply = await promise;
        return {
          answer: reply.answer,
          wasFreeform: reply.wasFreeform ?? true,
        };
      },

      onElicitationRequest: async (context) => {
        const { requestId, promise } = this.pending.create(
          session.sessionId,
          "elicitation",
          {
            message: context.message,
            requestedSchema: context.requestedSchema,
            mode: context.mode,
            elicitationSource: context.elicitationSource,
          },
        );
        this.bus.publish(session.sessionId, {
          type: "elicitation.request",
          requestId,
          data: {
            message: context.message,
            requestedSchema: context.requestedSchema,
            mode: context.mode,
            elicitationSource: context.elicitationSource,
          },
        });
        const reply = await promise;
        // reply: { action: "accept"|"decline"|"cancel", content?: object }
        return {
          action: reply.action,
          content: reply.content,
        };
      },

      onPermissionRequest: async (request) => {
        // Forward every field the SDK gives us so the UI can render a
        // meaningful summary. The SDK's permission requests vary in shape
        // by `kind`: a "read" permission typically carries `directory` or
        // `path` instead of `fileName`, while shell permissions carry
        // `fullCommandText`. We include all of them plus the named fields
        // the UI types already understand.
        const payload = {
          ...request,
          kind: request.kind,
          toolName: request.toolName,
          toolCallId: request.toolCallId,
          fileName: request.fileName,
          fullCommandText: request.fullCommandText,
        };
        const { requestId, promise } = this.pending.create(
          session.sessionId,
          "permission",
          payload,
        );
        this.bus.publish(session.sessionId, {
          type: "permission.request",
          requestId,
          data: payload,
        });
        const reply = await promise;
        // reply: { decision: "approve-once"|"approve-for-session"|"reject", feedback? }
        if (reply.decision === "reject") {
          return { kind: "reject", feedback: reply.feedback };
        }
        if (reply.decision === "approve-for-session") {
          return { kind: "approve-for-session" };
        }
        return { kind: "approve-once" };
      },
    });

    this.sessions.set(session.sessionId, session);
    this.meta.set(session.sessionId, {
      status: "idle",
      model: model || "(sdk default)",
      createdAt: Date.now(),
    });

    // When `silentInitialPrompt` is set we suppress the first user.message
    // echo so the bootstrap text never shows up in the UI transcript.
    const suppressUserEcho = !!cfg.silentInitialPrompt && !!cfg.initialPrompt;
    let userEchoSuppressed = false;

    // Forward streaming and lifecycle events to the SSE bus.
    const forward = (type) =>
      session.on(type, (event) => {
        this.bus.publish(session.sessionId, { type, data: event.data });
      });
    forward("assistant.message");
    forward("assistant.message_delta");
    forward("assistant.reasoning");
    forward("assistant.reasoning_delta");
    forward("tool.execution_start");
    forward("tool.execution_complete");
    session.on("user.message", (event) => {
      if (suppressUserEcho && !userEchoSuppressed) {
        userEchoSuppressed = true;
        return;
      }
      this.bus.publish(session.sessionId, {
        type: "user.message",
        data: event.data,
      });
    });

    session.on("session.idle", () => {
      const m = this.meta.get(session.sessionId);
      if (m) m.status = "idle";
      this.bus.publish(session.sessionId, { type: "session.idle" });
    });

    if (cfg.initialPrompt) {
      const m = this.meta.get(session.sessionId);
      if (m) m.status = "running";
      session
        .send({ prompt: cfg.initialPrompt })
        .catch((err) => this.#handleSessionError(session.sessionId, err));
    }

    return { sessionId: session.sessionId };
  }

  async sendMessage(sessionId, prompt) {
    const session = this.sessions.get(sessionId);
    if (!session) throw new Error(`Unknown session: ${sessionId}`);
    const m = this.meta.get(sessionId);
    if (m) m.status = "running";
    await session.send({ prompt });
  }

  respond(sessionId, requestId, value) {
    if (!this.sessions.has(sessionId)) {
      throw new Error(`Unknown session: ${sessionId}`);
    }
    const ok = this.pending.resolve(requestId, value);
    if (!ok) throw new Error(`Unknown or already-answered request: ${requestId}`);
  }

  async disconnect(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) return;
    this.pending.rejectSession(sessionId, new Error("Session disconnected"));
    this.sessions.delete(sessionId);
    this.meta.delete(sessionId);
    try {
      await session.disconnect();
    } catch {
      /* ignore */
    }
    this.bus.publish(sessionId, { type: "session.closed" });
    this.bus.drop(sessionId);
  }

  meta_get(sessionId) {
    return this.meta.get(sessionId);
  }

  pendingFor(sessionId) {
    return this.pending.list(sessionId);
  }

  async shutdown() {
    for (const sessionId of [...this.sessions.keys()]) {
      await this.disconnect(sessionId);
    }
    if (this.client) {
      try {
        await this.client.stop();
      } catch {
        /* ignore */
      }
      this.client = null;
    }
  }

  #handleSessionError(sessionId, err) {
    const m = this.meta.get(sessionId);
    if (m) {
      m.status = "error";
      m.error = err?.message || String(err);
    }
    this.bus.publish(sessionId, {
      type: "session.error",
      data: { message: err?.message || String(err) },
    });
  }
}
