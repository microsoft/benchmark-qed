import { useState } from "react";
import { ShieldKeyhole20Regular } from "@fluentui/react-icons";
import type { PermissionAnswer, PermissionPayload } from "../copilot/types";
import { summarizePermission } from "../copilot/permissionSummary";

interface Props {
  payload: PermissionPayload;
  submitting: boolean;
  onSubmit: (answer: PermissionAnswer) => void;
  onApproveAll: () => void;
  onCancel: () => void;
}

export function CopilotPermissionDialog({
  payload,
  submitting,
  onSubmit,
  onApproveAll,
  onCancel,
}: Props) {
  const [feedback, setFeedback] = useState("");

  const reject = () =>
    onSubmit({ decision: "reject", feedback: feedback.trim() || undefined });

  const summary = summarizePermission(payload);

  // onCancel is wired by the panel close button; not surfaced inline.
  void onCancel;

  return (
    <div className="copilot-inline-card">
      <div className="copilot-inline-card-header">
        <strong>
          <ShieldKeyhole20Regular
            style={{ verticalAlign: "-4px", marginRight: 6 }}
          />
          Approval required
        </strong>
      </div>
      <div className="copilot-inline-card-body">
        <p style={{ margin: 0 }}>Copilot wants to:</p>
        <pre
          style={{
            background: "var(--code-bg, #1e1e1e)",
            color: "var(--code-fg, #d4d4d4)",
            padding: "10px 12px",
            borderRadius: 6,
            fontSize: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            margin: 0,
          }}
        >
          {summary}
        </pre>
        {payload.toolName && payload.fullCommandText && (
          <small style={{ opacity: 0.7 }}>via {payload.toolName}</small>
        )}

        <label className="field">
          <span>Feedback (only sent if rejected)</span>
          <textarea
            rows={2}
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Optional: explain why you're rejecting..."
          />
        </label>

        <div className="copilot-inline-actions" style={{ flexWrap: "wrap", gap: 8 }}>
          <button type="button" className="btn" onClick={reject} disabled={submitting}>
            Reject
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={submitting}
            onClick={onApproveAll}
            title="Auto-approve all future permission prompts in this session without asking"
          >
            Approve all (this session)
          </button>
          <button
            type="button"
            className="btn"
            disabled={submitting}
            onClick={() => onSubmit({ decision: "approve-for-session" })}
          >
            Approve for session
          </button>
          <button
            type="button"
            className="btn"
            disabled={submitting}
            onClick={() => onSubmit({ decision: "approve-once" })}
          >
            Approve once
          </button>
        </div>
      </div>
    </div>
  );
}
