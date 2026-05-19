import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  AskUserAnswer,
  ChatTurn,
  ElicitationAnswer,
  PendingRequest,
  PermissionAnswer,
  SessionStatus,
  ToolEvent,
} from "./types";

const DEFAULT_BRIDGE_URL =
  (typeof import.meta !== "undefined" &&
    (import.meta as { env?: { VITE_COPILOT_BRIDGE_URL?: string } }).env
      ?.VITE_COPILOT_BRIDGE_URL) ||
  "http://localhost:8788";

export interface UseCopilotSessionResult {
  bridgeUrl: string;
  sessionId: string | null;
  status: SessionStatus;
  turns: ChatTurn[];
  tools: ToolEvent[];
  pending: PendingRequest | null;
  /** Start a new session. Resolves with the new sessionId. */
  start: (opts?: {
    initialPrompt?: string;
    /** When true, the bootstrap prompt is sent to the agent but hidden from the transcript. */
    silentInitialPrompt?: boolean;
    model?: string;
    skillDirectories?: string[];
  }) => Promise<string>;
  send: (prompt: string) => Promise<void>;
  answerUserInput: (answer: AskUserAnswer) => Promise<void>;
  answerElicitation: (answer: ElicitationAnswer) => Promise<void>;
  answerPermission: (answer: PermissionAnswer) => Promise<void>;
  /** When true, incoming permission.request events are auto-approved
   * (decision: approve-once) without enqueuing a dialog. */
  autoApprove: boolean;
  setAutoApprove: (on: boolean) => void;
  /** Push a synthetic user turn into the transcript (used to echo answers). */
  addLocalUserTurn: (content: string) => void;
  end: () => Promise<void>;
}

interface SseEnvelope<T = unknown> {
  type: string;
  requestId?: string;
  data?: T;
  replay?: boolean;
}

export function useCopilotSession(
  bridgeUrl: string = DEFAULT_BRIDGE_URL,
): UseCopilotSessionResult {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SessionStatus>({ state: "closed" });
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [tools, setTools] = useState<ToolEvent[]>([]);
  // Pending requests are queued so a permission prompt that arrives mid-form
  // doesn't lose the elicitation request behind it.
  const [pendingQueue, setPendingQueue] = useState<PendingRequest[]>([]);
  const seenRequestIds = useRef<Set<string>>(new Set());
  const esRef = useRef<EventSource | null>(null);
  const streamingIdRef = useRef<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const bridgeUrlRef = useRef<string>(bridgeUrl);
  const [autoApprove, setAutoApproveState] = useState(false);
  const autoApproveRef = useRef(false);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  useEffect(() => {
    bridgeUrlRef.current = bridgeUrl;
  }, [bridgeUrl]);

  const setAutoApprove = useCallback((on: boolean) => {
    autoApproveRef.current = on;
    setAutoApproveState(on);
  }, []);

  const pending = pendingQueue[0] ?? null;

  const closeStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  useEffect(() => () => closeStream(), [closeStream]);

  const handleEnvelope = useCallback((env: SseEnvelope) => {
    switch (env.type) {
      case "assistant.message_delta": {
        const data = env.data as { deltaContent?: string } | undefined;
        const delta = data?.deltaContent ?? "";
        if (!delta) return;
        setTurns((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            const updated: ChatTurn = { ...last, content: last.content + delta };
            return [...prev.slice(0, -1), updated];
          }
          const newTurn: ChatTurn = {
            id: crypto.randomUUID(),
            role: "assistant",
            content: delta,
            streaming: true,
          };
          streamingIdRef.current = newTurn.id;
          return [...prev, newTurn];
        });
        break;
      }
      case "assistant.message": {
        const data = env.data as { content?: string } | undefined;
        const content = (data?.content ?? "").trim();
        // Drop empty / whitespace-only messages — the SDK emits these around
        // tool calls and they'd show up as ghost "Copilot" bubbles.
        if (!content) {
          setTurns((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant" && last.streaming && !last.content.trim()) {
              return prev.slice(0, -1);
            }
            return prev;
          });
          streamingIdRef.current = null;
          return;
        }
        setTurns((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            return [
              ...prev.slice(0, -1),
              { ...last, content, streaming: false },
            ];
          }
          return [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content,
              streaming: false,
            },
          ];
        });
        streamingIdRef.current = null;
        break;
      }
      case "user.message": {
        const data = env.data as { content?: string } | undefined;
        const content = data?.content ?? "";
        if (!content) return;
        // The SDK injects skill content as a synthetic user message wrapped
        // in <skill-context>. That's plumbing, not user-authored text —
        // hide it from the transcript.
        if (content.includes("<skill-context")) return;
        setTurns((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "user",
            content,
          },
        ]);
        break;
      }
      case "tool.execution_start": {
        const data = env.data as {
          toolCallId?: string;
          toolName?: string;
          args?: unknown;
        };
        setTools((prev) => [
          ...prev,
          {
            id: data.toolCallId ?? crypto.randomUUID(),
            toolName: data.toolName ?? "tool",
            args: data.args,
            status: "running",
            startedAt: Date.now(),
          },
        ]);
        break;
      }
      case "tool.execution_complete": {
        const data = env.data as { toolCallId?: string; result?: unknown };
        setTools((prev) =>
          prev.map((t) =>
            t.id === data.toolCallId
              ? { ...t, status: "complete", result: data.result, endedAt: Date.now() }
              : t,
          ),
        );
        break;
      }
      case "user_input.request":
      case "elicitation.request":
      case "permission.request": {
        if (!env.requestId) return;
        if (seenRequestIds.current.has(env.requestId)) return;
        seenRequestIds.current.add(env.requestId);
        // If the user previously chose "Approve all for this session",
        // auto-respond and skip the inline card entirely. Only applies to
        // permission requests — ask_user / elicitation always need a reply.
        if (
          env.type === "permission.request" &&
          autoApproveRef.current &&
          sessionIdRef.current
        ) {
          const sid = sessionIdRef.current;
          const rid = env.requestId;
          void fetch(
            `${bridgeUrlRef.current}/api/copilot/sessions/${encodeURIComponent(
              sid,
            )}/respond`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                requestId: rid,
                kind: "permission",
                value: { decision: "approve-once" },
              }),
            },
          ).catch(() => {
            /* network error — the UI will surface session.error */
          });
          break;
        }
        const kind = env.type.replace(".request", "") as PendingRequest["kind"];
        setPendingQueue((prev) => [
          ...prev,
          {
            kind,
            requestId: env.requestId!,
            // We trust the bridge to send the right shape per kind; the
            // dialogs perform their own narrow checks before rendering.
            payload: env.data as PendingRequest["payload"],
          },
        ]);
        break;
      }
      case "session.idle":
        setStatus({ state: "idle" });
        break;
      case "session.error": {
        const data = env.data as { message?: string } | undefined;
        setStatus({ state: "error", error: data?.message });
        break;
      }
      case "session.closed":
        setStatus({ state: "closed" });
        break;
      default:
        // Unhandled event types are intentionally ignored — assistant.reasoning
        // and similar belong to a future "reasoning trace" panel.
        break;
    }
  }, []);

  const openStream = useCallback(
    (id: string) => {
      closeStream();
      const es = new EventSource(
        `${bridgeUrl}/api/copilot/sessions/${encodeURIComponent(id)}/events`,
      );
      esRef.current = es;

      // The bridge tags each event with `event: <type>`. We register handlers
      // for the known set so we get proper typing per dispatch; everything
      // else falls through to the default `message` handler.
      const known = [
        "assistant.message",
        "assistant.message_delta",
        "assistant.reasoning",
        "assistant.reasoning_delta",
        "tool.execution_start",
        "tool.execution_complete",
        "user.message",
        "user_input.request",
        "elicitation.request",
        "permission.request",
        "session.idle",
        "session.error",
        "session.closed",
      ];
      const dispatch = (e: MessageEvent) => {
        try {
          handleEnvelope(JSON.parse(e.data));
        } catch {
          /* malformed event — drop */
        }
      };
      for (const t of known) es.addEventListener(t, dispatch as EventListener);
      es.onerror = () => {
        setStatus((prev) =>
          prev.state === "closed"
            ? prev
            : { state: "error", error: "Lost connection to copilot-bridge" },
        );
      };
    },
    [bridgeUrl, closeStream, handleEnvelope],
  );

  const start = useCallback<UseCopilotSessionResult["start"]>(
    async (opts) => {
      setStatus({ state: "connecting" });
      setTurns([]);
      setTools([]);
      setPendingQueue([]);
      seenRequestIds.current = new Set();
      const res = await fetch(`${bridgeUrl}/api/copilot/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(opts ?? {}),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        setStatus({ state: "error", error: err.error });
        throw new Error(err.error || "Failed to start session");
      }
      const { sessionId: id } = (await res.json()) as { sessionId: string };
      setSessionId(id);
      setStatus({ state: "running" });
      if (opts?.initialPrompt && !opts.silentInitialPrompt) {
        setTurns([
          {
            id: crypto.randomUUID(),
            role: "user",
            content: opts.initialPrompt,
          },
        ]);
      }
      openStream(id);
      return id;
    },
    [bridgeUrl, openStream],
  );

  const send = useCallback(
    async (prompt: string) => {
      if (!sessionId) throw new Error("No active session");
      setTurns((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: prompt },
      ]);
      setStatus({ state: "running" });
      const res = await fetch(
        `${bridgeUrl}/api/copilot/sessions/${encodeURIComponent(sessionId)}/message`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || "Failed to send message");
      }
    },
    [bridgeUrl, sessionId],
  );

  const respond = useCallback(
    async (requestId: string, kind: PendingRequest["kind"], value: unknown) => {
      if (!sessionId) throw new Error("No active session");
      const res = await fetch(
        `${bridgeUrl}/api/copilot/sessions/${encodeURIComponent(sessionId)}/respond`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ requestId, kind, value }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || "Failed to send answer");
      }
      setPendingQueue((prev) => prev.filter((p) => p.requestId !== requestId));
    },
    [bridgeUrl, sessionId],
  );

  const answerUserInput = useCallback<
    UseCopilotSessionResult["answerUserInput"]
  >(
    async (answer) => {
      if (!pending || pending.kind !== "user_input") return;
      await respond(pending.requestId, "user_input", answer);
    },
    [pending, respond],
  );

  const answerElicitation = useCallback<
    UseCopilotSessionResult["answerElicitation"]
  >(
    async (answer) => {
      if (!pending || pending.kind !== "elicitation") return;
      await respond(pending.requestId, "elicitation", answer);
    },
    [pending, respond],
  );

  const answerPermission = useCallback<
    UseCopilotSessionResult["answerPermission"]
  >(
    async (answer) => {
      if (!pending || pending.kind !== "permission") return;
      await respond(pending.requestId, "permission", answer);
    },
    [pending, respond],
  );

  const addLocalUserTurn = useCallback((content: string) => {
    if (!content.trim()) return;
    setTurns((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content },
    ]);
  }, []);

  const end = useCallback(async () => {
    closeStream();
    if (!sessionId) {
      setStatus({ state: "closed" });
      return;
    }
    try {
      await fetch(
        `${bridgeUrl}/api/copilot/sessions/${encodeURIComponent(sessionId)}`,
        { method: "DELETE" },
      );
    } finally {
      setSessionId(null);
      setStatus({ state: "closed" });
      setPendingQueue([]);
    }
  }, [bridgeUrl, closeStream, sessionId]);

  return useMemo(
    () => ({
      bridgeUrl,
      sessionId,
      status,
      turns,
      tools,
      pending,
      start,
      send,
      answerUserInput,
      answerElicitation,
      answerPermission,
      autoApprove,
      setAutoApprove,
      addLocalUserTurn,
      end,
    }),
    [
      bridgeUrl,
      sessionId,
      status,
      turns,
      tools,
      pending,
      start,
      send,
      answerUserInput,
      answerElicitation,
      answerPermission,
      autoApprove,
      setAutoApprove,
      addLocalUserTurn,
      end,
    ],
  );
}
