import { useEffect, useMemo, useRef, useState } from "react";
import { Dismiss16Regular, Send16Regular } from "@fluentui/react-icons";
import { marked } from "marked";
import { useCopilotSession } from "../copilot/useCopilotSession";
import type {
  AskUserPayload,
  ChatTurn,
  ElicitationPayload,
  PermissionPayload,
} from "../copilot/types";
import { CopilotAskUserDialog } from "./CopilotAskUserDialog";
import { CopilotElicitationDialog } from "./CopilotElicitationDialog";
import { CopilotPermissionDialog } from "./CopilotPermissionDialog";
import { CopilotAuthBadge } from "./CopilotAuthBadge";

interface Props {
  open: boolean;
  /** Bridge URL — defaults match useCopilotSession's default. */
  bridgeUrl?: string;
  /** Skill discovery roots passed to the SDK. */
  skillDirectories?: string[];
  /** Auto-sent first message that kicks off the skill. */
  initialPrompt?: string;
  /** Send `initialPrompt` to the agent but don't show it in the transcript. */
  silentInitialPrompt?: boolean;
  /** Model id, e.g. "gpt-5" or "claude-sonnet-4-5". */
  model?: string;
  /** Called whenever the user answers with an absolute filesystem path. */
  onFolderDetected?: (folderPath: string) => void;
  /**
   * Called whenever the agent goes idle after running (a "turn done" signal).
   * Used by the parent to refresh workspace trees so files the agent just
   * created or deleted show up without a manual reload.
   */
  onActivitySettled?: () => void;
  /**
   * Called whenever a noteworthy event happens inside the assistant
   * (session started, user answered a prompt, tool ran, error). The host
   * uses this to append entries to its global Activity Log.
   */
  onLogEvent?: (
    action: string,
    details?: string,
    type?: "info" | "success" | "warning" | "error",
  ) => void;
  onClose: () => void;
}

export function CopilotPanel({
  open,
  bridgeUrl,
  skillDirectories,
  initialPrompt,
  silentInitialPrompt,
  model,
  onFolderDetected,
  onActivitySettled,
  onLogEvent,
  onClose,
}: Props) {
  const session = useCopilotSession(bridgeUrl);
  const {
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
  } = session;

  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [answering, setAnswering] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // Start session when the panel opens; tear down when it closes.
  // Guard with a ref so the "started" log + start() call only fire once
  // per open cycle (the effect may re-run before `start()` resolves and
  // sets sessionId because callback prop identities can change between
  // renders).
  const startedRef = useRef(false);
  useEffect(() => {
    if (!open) {
      startedRef.current = false;
      return;
    }
    if (sessionId || startedRef.current) return;
    startedRef.current = true;
    onLogEvent?.("AI Assistant started", "benchmark-qed-setup", "info");
    void start({ initialPrompt, silentInitialPrompt, model, skillDirectories }).catch((e) => {
      console.error("Failed to start copilot session", e);
      onLogEvent?.(
        "AI Assistant failed to start",
        e instanceof Error ? e.message : String(e),
        "error",
      );
      startedRef.current = false;
    });
  }, [open, sessionId, start, initialPrompt, silentInitialPrompt, model, skillDirectories, onLogEvent]);

  // Track running totals so the "ended" log entry can include a summary.
  const turnsCountRef = useRef(0);
  const toolsCountRef = useRef(0);
  useEffect(() => {
    turnsCountRef.current = turns.length;
  }, [turns]);
  useEffect(() => {
    toolsCountRef.current = tools.filter((t) => t.status === "complete").length;
  }, [tools]);

  useEffect(() => {
    if (!open && sessionId) {
      const t = turnsCountRef.current;
      const k = toolsCountRef.current;
      onLogEvent?.(
        "AI Assistant ended",
        `${t} message${t === 1 ? "" : "s"}, ${k} tool call${k === 1 ? "" : "s"}`,
        "info",
      );
      void end();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns, tools]);

  // Fire onActivitySettled whenever the session transitions from running
  // (or any non-idle state) back to idle. The agent has stopped doing work,
  // so any filesystem mutations it performed should now be reflected.
  const prevStateRef = useRef(status.state);
  useEffect(() => {
    const prev = prevStateRef.current;
    if (prev !== "idle" && status.state === "idle") {
      onActivitySettled?.();
    }
    if (prev !== "error" && status.state === "error") {
      onLogEvent?.(
        "AI Assistant error",
        status.error ?? "unknown",
        "error",
      );
    }
    prevStateRef.current = status.state;
  }, [status.state, status.error, onActivitySettled, onLogEvent]);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;
    setSending(true);
    try {
      await send(text);
      setDraft("");
    } catch (err) {
      console.error(err);
    } finally {
      setSending(false);
    }
  };

  const handleAnswered = async <T,>(fn: (v: T) => Promise<void>, value: T) => {
    setAnswering(true);
    try {
      await fn(value);
    } finally {
      setAnswering(false);
    }
  };

  // Echo answers back into the transcript so the conversation reads naturally,
  // and forward any absolute paths to the parent so the workspace tree can
  // pick the new folder up — but ONLY when the originating question was
  // about the workspace location itself. Picking a folder in response to
  // "where is your data?" must not register that folder as a workspace.
  const looksLikeWorkspaceCreationPrompt = (q: string): boolean => {
    if (!q) return false;
    return /\b(workspace|project)\b/i.test(q) &&
      /\b(where|create|location|path|root|folder|directory)\b/i.test(q);
  };
  const echoAnswer = (label: string, body: string, question = "") => {
    addLocalUserTurn(body ? `${label}: ${body}` : label);
    if (!onFolderDetected || !looksLikeWorkspaceCreationPrompt(question)) return;
    // Loose detection: any token that looks like an absolute Unix path.
    const matches = body.match(/(^|\s)(\/[^\s,;]+)/g);
    if (matches) {
      for (const raw of matches) {
        const candidate = raw.trim();
        if (candidate.length > 1) onFolderDetected(candidate);
      }
    }
  };

  // Compress a question/prompt down to a single line so it fits as a
  // header above the user's answer in the transcript.
  const summarizeQuestion = (text: string, max = 160): string => {
    const oneLine = text.replace(/\s+/g, " ").trim();
    return oneLine.length > max ? `${oneLine.slice(0, max - 1)}…` : oneLine;
  };

  const onAnswerUserInput = (v: { answer: string; wasFreeform?: boolean }) => {
    const rawQ =
      pending?.kind === "user_input"
        ? (pending.payload as AskUserPayload).question
        : "";
    const q = summarizeQuestion(rawQ);
    echoAnswer(q ? `Q: ${q}\nA` : "You answered", v.answer, rawQ);
    return handleAnswered(answerUserInput, v);
  };

  const onAnswerElicitation = (v: {
    action: "accept" | "decline" | "cancel";
    content?: Record<string, unknown>;
  }) => {
    const rawQ =
      pending?.kind === "elicitation"
        ? (pending.payload as ElicitationPayload).message || "Form"
        : "";
    const q = summarizeQuestion(rawQ);
    const prefix = q ? `Q: ${q}\n` : "";
    if (v.action === "accept" && v.content) {
      const lines = Object.entries(v.content)
        .map(([k, val]) => `  ${k}: ${JSON.stringify(val)}`)
        .join("\n");
      echoAnswer(`${prefix}Submitted`, lines, rawQ);
    } else {
      echoAnswer(
        `${prefix}${v.action === "decline" ? "Declined" : "Cancelled"}`,
        "",
        rawQ,
      );
    }
    return handleAnswered(answerElicitation, v);
  };

  const onAnswerPermission = (v: {
    decision: "approve-once" | "approve-for-session" | "reject";
    feedback?: string;
  }) => {
    let q = "";
    if (pending?.kind === "permission") {
      const p = pending.payload as PermissionPayload;
      const summary = p.fullCommandText
        ? `Run command: ${p.fullCommandText}`
        : p.fileName
          ? `${p.kind}: ${p.fileName}`
          : p.toolName
            ? `Tool: ${p.toolName}`
            : p.kind;
      q = summarizeQuestion(summary);
    }
    const label =
      v.decision === "reject"
        ? "Rejected"
        : v.decision === "approve-for-session"
          ? "Approved (for session)"
          : "Approved";
    echoAnswer(q ? `Q: ${q}\n${label}` : label, v.feedback ?? "");
    return handleAnswered(answerPermission, v);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal copilot-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ width: "min(880px, 96vw)", maxHeight: "92vh" }}
      >
        <div className="modal-header">
          <h3 style={{ margin: 0 }}>Copilot AI Assistant</h3>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <CopilotAuthBadge bridgeUrl={session.bridgeUrl} />
            <button
              className="modal-close"
              onClick={onClose}
              aria-label="Close"
              title="Close"
            >
              <Dismiss16Regular />
            </button>
          </div>
        </div>

        <div
          className="modal-body"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            minHeight: 360,
          }}
        >
          <div
            style={{
              fontSize: 12,
              opacity: 0.7,
              display: "flex",
              gap: 8,
              alignItems: "center",
            }}
          >
            <span>Status:</span>
            <strong>{status.state}</strong>
            {status.error && (
              <span style={{ color: "#d4453e" }}>· {status.error}</span>
            )}
            {model && <span>· model: {model}</span>}
            {autoApprove && (
              <span
                style={{
                  marginLeft: "auto",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "3px 10px",
                  borderRadius: 12,
                  background: "var(--accent, #4a90e2)",
                  color: "#fff",
                  fontWeight: 600,
                  fontSize: 12,
                }}
              >
                Auto-approving permissions
                <button
                  type="button"
                  onClick={() => setAutoApprove(false)}
                  style={{
                    background: "rgba(255, 255, 255, 0.18)",
                    border: "none",
                    color: "#fff",
                    cursor: "pointer",
                    padding: "1px 8px",
                    borderRadius: 8,
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                  title="Stop auto-approving and ask me again"
                >
                  turn off
                </button>
              </span>
            )}
          </div>

          <div
            ref={transcriptRef}
            className="copilot-transcript"
            style={{
              flex: 1,
              overflowY: "auto",
              border: "1px solid var(--border, rgba(127,127,127,0.25))",
              borderRadius: 6,
              padding: 12,
              minHeight: 320,
              background: "var(--panel-bg, rgba(127,127,127,0.04))",
              display: "flex",
              flexDirection: "column",
              gap: 10,
            }}
          >
            {turns.length === 0 && status.state === "connecting" && (
              <p style={{ opacity: 0.6, margin: 0 }}>Spinning up Copilot...</p>
            )}
            {turns.length === 0 && status.state === "running" && (
              <p style={{ opacity: 0.6, margin: 0 }}>
                Waiting for the agent...
              </p>
            )}
            {turns.map((t) => (
              <CopilotTurnView key={t.id} turn={t} />
            ))}
            {pending?.kind === "user_input" && (
              <CopilotAskUserDialog
                key={pending.requestId}
                payload={pending.payload as AskUserPayload}
                submitting={answering}
                onSubmit={onAnswerUserInput}
                onCancel={() =>
                  onAnswerUserInput({ answer: "cancel", wasFreeform: true })
                }
              />
            )}
            {pending?.kind === "elicitation" && (
              <CopilotElicitationDialog
                key={pending.requestId}
                payload={pending.payload as ElicitationPayload}
                submitting={answering}
                onSubmit={onAnswerElicitation}
                onCancel={() => onAnswerElicitation({ action: "cancel" })}
              />
            )}
            {pending?.kind === "permission" && (
              <CopilotPermissionDialog
                key={pending.requestId}
                payload={pending.payload as PermissionPayload}
                submitting={answering}
                onSubmit={onAnswerPermission}
                onApproveAll={() => {
                  setAutoApprove(true);
                  addLocalUserTurn(
                    "Approved all future permission prompts for this session.",
                  );
                  void onAnswerPermission({ decision: "approve-once" });
                }}
                onCancel={() => onAnswerPermission({ decision: "reject" })}
              />
            )}
            {tools.length > 0 && (
              <details style={{ marginTop: 8 }}>
                <summary style={{ cursor: "pointer", opacity: 0.7 }}>
                  Tool activity ({tools.length})
                </summary>
                <ul style={{ fontSize: 12, paddingLeft: 16, margin: "6px 0" }}>
                  {tools.map((t) => (
                    <li key={t.id}>
                      <code>{t.toolName}</code> · {t.status}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {status.state === "running" && !pending && turns.length > 0 && (
              <div
                className="copilot-thinking"
                role="status"
                aria-live="polite"
              >
                <span className="copilot-thinking-dot" />
                <span className="copilot-thinking-dot" />
                <span className="copilot-thinking-dot" />
                <span className="copilot-thinking-label">
                  Copilot is thinking…
                </span>
              </div>
            )}
            {status.state === "idle" && !pending && turns.length > 0 && (
              <div className="copilot-done">
                <span className="copilot-done-label">
                  ✓ Assistant finished
                </span>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onClose}
                >
                  Close window
                </button>
              </div>
            )}
          </div>

          <form
            onSubmit={submit}
            style={{ display: "flex", gap: 8, alignItems: "stretch" }}
          >
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={
                pending
                  ? "Answer the prompt above first..."
                  : "Send a follow-up message"
              }
              rows={2}
              disabled={!sessionId || !!pending || sending}
              style={{
                flex: 1,
                resize: "vertical",
                padding: 8,
                fontFamily: "inherit",
              }}
              onKeyDown={(e) => {
                // Enter submits; Shift+Enter inserts a newline.
                // (Cmd/Ctrl+Enter also submits for muscle memory.)
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submit(e as unknown as React.FormEvent);
                }
              }}
            />
            <button
              type="submit"
              className="btn btn-primary"
              disabled={!sessionId || !!pending || sending || !draft.trim()}
              style={{ alignSelf: "stretch", padding: "0 14px" }}
            >
              <Send16Regular />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

// Long assistant messages are collapsed by default so an accidental skill
// dump doesn't bury the next prompt.
const COLLAPSE_THRESHOLD = 600;

function CopilotTurnView({ turn }: { turn: ChatTurn }) {
  const [expanded, setExpanded] = useState(false);
  const long = turn.content.length > COLLAPSE_THRESHOLD;
  const display =
    long && !expanded
      ? `${turn.content.slice(0, COLLAPSE_THRESHOLD).trimEnd()}…`
      : turn.content;

  // Render assistant turns as Markdown so fenced code blocks, lists, and
  // inline code show up styled instead of as raw backticks. User turns stay
  // plain text so the Q:/A: echo keeps its exact formatting.
  const html = useMemo(() => {
    if (turn.role !== "assistant") return null;
    try {
      return marked.parse(display, { async: false, gfm: true }) as string;
    } catch {
      return null;
    }
  }, [display, turn.role]);

  return (
    <div
      className={`copilot-turn copilot-turn-${turn.role}`}
      style={{
        padding: "8px 12px",
        borderRadius: 8,
        background:
          turn.role === "user"
            ? "var(--turn-user-bg, rgba(74,144,226,0.18))"
            : "var(--turn-asst-bg, rgba(127,127,127,0.12))",
        wordBreak: "break-word",
      }}
    >
      <small style={{ opacity: 0.6, display: "block" }}>
        {turn.role === "user" ? "You" : "Copilot"}
        {turn.streaming ? " · streaming…" : ""}
      </small>
      {html ? (
        <div
          className="copilot-md"
          // marked's output is sanitized-by-default; the agent's text is
          // already trusted (it comes from our own bridge), so we just
          // render it directly.
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <div style={{ whiteSpace: "pre-wrap" }}>{display}</div>
      )}
      {long && (
        <button
          type="button"
          className="btn"
          style={{
            display: "inline-block",
            marginTop: 6,
            padding: "2px 8px",
            fontSize: 11,
          }}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show less" : `Show full (${turn.content.length} chars)`}
        </button>
      )}
    </div>
  );
}
