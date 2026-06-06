import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";
import type { WorkspaceConfigType } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (configType: WorkspaceConfigType) => Promise<void>;
}

const RUN_OPTIONS: Array<{
  value: WorkspaceConfigType;
  label: string;
  description: string;
}> = [
  {
    value: "autoq",
    label: "AutoQ",
    description: "Run question generation.",
  },
  {
    value: "autoe_pairwise",
    label: "AutoE Pairwise",
    description: "Run pairwise scoring.",
  },
  {
    value: "autoe_reference",
    label: "AutoE Reference",
    description: "Run reference-based scoring.",
  },
  {
    value: "autoe_assertion",
    label: "AutoE Assertion",
    description: "Run assertion scoring.",
  },
];

export function RunConfigDialog({ open, onClose, onSubmit }: Props) {
  const [configType, setConfigType] =
    useState<WorkspaceConfigType>("autoq");
  const [submitting, setSubmitting] = useState(false);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(configType);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={submitting ? undefined : onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Run Workspace</h3>
          <button
            type="button"
            className="modal-close"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close"
          >
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit} aria-busy={submitting}>
          <section className="option-group">
            <h4>Choose Run Type</h4>
            <div className="option-grid">
              {RUN_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`option-card ${configType === opt.value ? "active" : ""}`}
                  disabled={submitting}
                  onClick={() => setConfigType(opt.value)}
                >
                  <span className="option-card-label">{opt.label}</span>
                  <span className="option-card-desc">{opt.description}</span>
                </button>
              ))}
            </div>
          </section>

          {submitting && (
            <div className="modal-callout modal-callout-info" role="status" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              <span>
                Starting the job in the background. The UI is still responsive;
                check the Jobs panel for live progress.
              </span>
            </div>
          )}

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? "Starting..." : "Run"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
