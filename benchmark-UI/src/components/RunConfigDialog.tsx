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
  commands: Array<{ value: string; label: string; description: string }>;
}> = [
  {
    value: "autoq",
    label: "AutoQ",
    description: "Run question generation.",
    commands: [
      {
        value: "autoq",
        label: "Question Generation",
        description: "Generate questions from input data",
      },
    ],
  },
  {
    value: "autoe_pairwise",
    label: "AutoE Pairwise",
    description: "Run pairwise scoring.",
    commands: [
      {
        value: "autoe_pairwise",
        label: "Pairwise Scoring",
        description: "Compare two sets of answers pairwise",
      },
    ],
  },
  {
    value: "autoe_reference",
    label: "AutoE Reference",
    description: "Run reference-based scoring.",
    commands: [
      {
        value: "autoe_reference",
        label: "Reference-Based Scoring",
        description: "Score answers against reference answers",
      },
    ],
  },
  {
    value: "autoe_assertion",
    label: "AutoE Assertion",
    description: "Run assertion scoring.",
    commands: [
      {
        value: "autoe_assertion",
        label: "Assertion Scoring",
        description: "Score answers using assertion validation",
      },
    ],
  },
];

export function RunConfigDialog({ open, onClose, onSubmit }: Props) {
  const [configType, setConfigType] =
    useState<WorkspaceConfigType>("autoq");

  if (!open) return null;

  const selectedOption = RUN_OPTIONS.find((opt) => opt.value === configType);
  const selectedCommand = selectedOption?.commands[0];

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(configType);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Run Workspace</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit}>
          <section className="option-group">
            <h4>Choose Run Type</h4>
            <div className="option-grid">
              {RUN_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`option-card ${configType === opt.value ? "active" : ""}`}
                  onClick={() => setConfigType(opt.value)}
                >
                  <span className="option-card-label">{opt.label}</span>
                  <span className="option-card-desc">{opt.description}</span>
                </button>
              ))}
            </div>
          </section>

          {selectedCommand && (
            <section className="option-group">
              <h4>Command</h4>
              <div className="command-display">
                <div className="command-info">
                  <div className="command-label">{selectedCommand.label}</div>
                  <div className="command-desc">{selectedCommand.description}</div>
                </div>
              </div>
            </section>
          )}

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary">
              Run
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
