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
import { summarizePermission } from "../copilot/permissionSummary";

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
  /** Optional label for the skill being launched (used in activity log). */
  skillLabel?: string;
  /**
   * Optional list of skills to let the user choose from inside the panel.
   * When provided AND `initialPrompt` is empty, the panel shows a chooser
   * instead of auto-starting a session; selecting a card calls
   * `onSkillSelected` so the host can either set `initialPrompt` (which
   * starts the session) or route to a different flow (e.g. a folder picker
   * dialog).
   */
  skillChoices?: Array<{
    id: string;
    label: string;
    description?: string;
  }>;
  onSkillSelected?: (id: string) => void;
  /**
   * Optional inline view rendered inside the panel before the session
   * starts. When provided it replaces the skill picker (used e.g. for the
   * "Evaluate Question Quality" folder selector — keeps the user inside
   * the Copilot popup instead of opening a separate dialog).
   */
  inlinePicker?: { title: string; node: React.ReactNode };
  /**
   * When true, render as a regular page region (full width/height of its
   * parent, no modal backdrop, no close button — the host's tab UI owns
   * lifecycle). When false (default), render as a centered modal.
   */
  inline?: boolean;
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
  /**
   * Optional callback invoked when the user clicks "Back to options" after
   * a session finishes. The host should clear whatever prompt/skill state
   * caused the session to start so the skill picker renders again. The
   * panel ends the current bridge session and keeps the prior transcript
   * visible as history above the picker.
   */
  onBackToOptions?: () => void;
  onClose: () => void;
}

export function CopilotPanel({
  open,
  bridgeUrl,
  skillDirectories,
  initialPrompt,
  silentInitialPrompt,
  skillLabel,
  skillChoices,
  onSkillSelected,
  inlinePicker,
  inline,
  model,
  onFolderDetected,
  onActivitySettled,
  onLogEvent,
  onBackToOptions,
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
    resetTurns,
    end,
  } = session;

  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  // Past, finished sessions kept around so the user can scroll back through
  // earlier skill runs after clicking "Back to options". Each entry holds
  // an immutable snapshot of the chat turns plus a label.
  const [archivedSessions, setArchivedSessions] = useState<
    Array<{ id: string; label: string; turns: ChatTurn[] }>
  >([]);
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
    // If the host is going to let the user pick a skill inside the panel,
    // wait until that selection produces an `initialPrompt` before starting
    // the session.
    if (skillChoices && skillChoices.length > 0 && !initialPrompt) return;
    // If the host is showing an inline picker (e.g. folder selection),
    // wait until it's dismissed / submitted before starting.
    if (inlinePicker && !initialPrompt) return;
    startedRef.current = true;
    onLogEvent?.("AI Assistant started", skillLabel ?? "benchmark-qed", "info");
    void start({ initialPrompt, silentInitialPrompt, model, skillDirectories }).catch((e) => {
      console.error("Failed to start copilot session", e);
      onLogEvent?.(
        "AI Assistant failed to start",
        e instanceof Error ? e.message : String(e),
        "error",
      );
      startedRef.current = false;
    });
  }, [open, sessionId, start, initialPrompt, silentInitialPrompt, skillLabel, skillChoices, inlinePicker, model, skillDirectories, onLogEvent]);

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

  // Auto-scroll the transcript to the bottom as new content arrives — but
  // only when the user is already pinned to the bottom. If they have
  // scrolled up to review earlier output we leave their position alone so
  // streaming deltas don't yank them back down. The "near bottom" check
  // uses a small threshold so trailing whitespace / inline cards don't
  // accidentally unstick.
  const stuckToBottomRef = useRef(true);
  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      stuckToBottomRef.current = distance < 40;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!stuckToBottomRef.current) return;
    const el = transcriptRef.current;
    if (!el) return;
    // Instant (not smooth): smooth animation conflicts with wheel events
    // and makes scrolling feel glitchy while content streams in.
    el.scrollTop = el.scrollHeight;
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

  /**
   * Tear down the active Copilot session. Wired to the header "Cancel"
   * button so the user can bail out of a long-running operation (the
   * bridge's DELETE handler aborts the SDK run). We log a turn so the
   * transcript shows that the user interrupted the run, and we close any
   * pending permission / ask_user prompt by rejecting it first.
   */
  const cancelSession = async () => {
    if (cancelling) return;
    setCancelling(true);
    try {
      if (sessionId) {
        addLocalUserTurn("⏹ Cancelled the current Copilot operation.");
        onLogEvent?.(
          "Copilot operation cancelled by user",
          `session ${sessionId}`,
          "info",
        );
      }
      await end();
    } finally {
      setCancelling(false);
    }
  };

  /**
   * Archive the current chat turns, end the bridge session, and ask the
   * host to reset its skill-selection state so the in-panel picker shows
   * again. The archived turns stay visible above the picker as history.
   */
  const backToOptions = async () => {
    if (!onBackToOptions) return;
    if (turns.length > 0) {
      const label = skillLabel ?? "Previous session";
      setArchivedSessions((prev) => [
        ...prev,
        { id: crypto.randomUUID(), label, turns },
      ]);
    }
    try {
      await end();
    } finally {
      // Clear the live transcript: the snapshot now lives in
      // archivedSessions and would otherwise render twice.
      resetTurns();
      // Reset the once-per-open guard so the next skill selection is
      // allowed to start a fresh session. Without this, picking a skill
      // after "Back to options" silently does nothing.
      startedRef.current = false;
      onBackToOptions();
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
      q = summarizeQuestion(summarizePermission(p));
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

  const headerNode = (
    <div className="modal-header">
      <h3 style={{ margin: 0 }}>Copilot AI Assistant</h3>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <CopilotAuthBadge bridgeUrl={session.bridgeUrl} />
        {sessionId &&
          status.state !== "closed" &&
          status.state !== "error" && (
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => void cancelSession()}
              disabled={cancelling}
              title="Stop the current Copilot operation"
              style={{
                padding: "2px 10px",
                fontSize: 12,
              }}
            >
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
        {!inline && (
          <button
            className="modal-close"
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            <Dismiss16Regular />
          </button>
        )}
      </div>
    </div>
  );

  const bodyContent = (
    <>
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
            {archivedSessions.map((a) => (
              <div key={a.id} className="copilot-archived-session">
                <div className="copilot-archived-separator">
                  <span>Previous run · {a.label}</span>
                </div>
                {a.turns.map((t) => (
                  <CopilotTurnView key={t.id} turn={t} />
                ))}
              </div>
            ))}
            {archivedSessions.length > 0 && !sessionId && (
              <div className="copilot-archived-separator">
                <span>New run</span>
              </div>
            )}
            {!sessionId && inlinePicker && (
              <div className="copilot-skill-picker">
                <p className="copilot-skill-picker-title">
                  {inlinePicker.title}
                </p>
                {inlinePicker.node}
              </div>
            )}
            {!sessionId &&
              !inlinePicker &&
              !initialPrompt &&
              skillChoices &&
              skillChoices.length > 0 &&
              status.state !== "connecting" && (
                <div className="copilot-skill-picker">
                  <p className="copilot-skill-picker-title">
                    What do you want to do?
                  </p>
                  <div className="copilot-skill-picker-grid">
                    {skillChoices.map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        className="copilot-skill-card"
                        onClick={() => onSkillSelected?.(s.id)}
                      >
                        <span className="copilot-skill-card-label">
                          {s.label}
                        </span>
                        {s.description && (
                          <span className="copilot-skill-card-desc">
                            {s.description}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
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
                {onBackToOptions && (
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void backToOptions()}
                    title="Return to the skill picker and keep this transcript as history"
                  >
                    Back to options
                  </button>
                )}
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
        </>
      );

  if (inline) {
    return (
      <div className="copilot-inline">
        {headerNode}
        <div
          className="modal-body copilot-inline-body"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            flex: 1,
            minHeight: 0,
          }}
        >
          {bodyContent}
        </div>
      </div>
    );
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal copilot-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ width: "min(880px, 96vw)", maxHeight: "92vh" }}
      >
        {headerNode}
        <div
          className="modal-body"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 12,
            minHeight: 360,
          }}
        >
          {bodyContent}
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
