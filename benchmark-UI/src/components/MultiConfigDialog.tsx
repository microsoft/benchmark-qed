import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";
import type { WorkspaceConfigType } from "../types";

export interface DetectedConfig {
  path: string;
  name: string;
  configType: WorkspaceConfigType;
  isRoot: boolean;
}

interface Props {
  open: boolean;
  folderPath: string;
  configs: DetectedConfig[];
  submitting: boolean;
  onClose: () => void;
  onRunSelected: (selectedConfigs: DetectedConfig[]) => Promise<void>;
  mode?: "add" | "run";
}

const ALL_COMMANDS: Array<{ value: WorkspaceConfigType; label: string; description: string }> = [
  { value: "autoq", label: "AutoQ", description: "Question generation" },
  { value: "autoe_pairwise", label: "AutoE Pairwise", description: "Pairwise evaluation" },
  { value: "autoe_reference", label: "AutoE Reference", description: "Reference-based scoring" },
  { value: "autoe_assertion", label: "AutoE Assertion", description: "Assertion-based scoring" },
];

export function MultiConfigDialog({
  open,
  folderPath,
  configs,
  submitting,
  onClose,
  onRunSelected,
  mode = "add",
}: Props) {
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(
    new Set(configs.map((c) => c.path)),
  );
  // Per-config command overrides (keyed by path)
  const [commandOverrides, setCommandOverrides] = useState<Record<string, WorkspaceConfigType>>(
    () => Object.fromEntries(configs.map((c) => [c.path, c.configType])),
  );

  if (!open || configs.length === 0) return null;

  const handleToggle = (path: string) => {
    const next = new Set(selectedPaths);
    if (next.has(path)) {
      next.delete(path);
    } else {
      next.add(path);
    }
    setSelectedPaths(next);
  };

  const handleCommandOverride = (path: string, cmd: WorkspaceConfigType) => {
    setCommandOverrides((prev) => ({ ...prev, [path]: cmd }));
  };

  const handleSelectAll = () => {
    setSelectedPaths(new Set(configs.map((c) => c.path)));
  };

  const handleDeselectAll = () => {
    setSelectedPaths(new Set());
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedPaths.size === 0) return;

    const selected = configs
      .filter((c) => selectedPaths.has(c.path))
      .map((c) => ({ ...c, configType: commandOverrides[c.path] ?? c.configType }));
    await onRunSelected(selected);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{configs.length === 1 ? "Run Configuration" : "Multiple Configurations Detected"}</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>

        <form className="modal-body" onSubmit={handleSubmit}>
          <p className="modal-subtext">
            {configs.length === 1
              ? <>Configuration found in <code>{folderPath}</code>. Choose which command to run.</>
              : <>Found {configs.length} configurations in <code>{folderPath}</code>. Select which to {mode === "run" ? "run" : "add as workspaces"}.</>
            }
          </p>

          <div className="config-list">
            {configs.map((config) => (
              <div key={config.path} className="config-item-wrapper">
                <label className="config-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedPaths.has(config.path)}
                    onChange={() => handleToggle(config.path)}
                    disabled={submitting}
                  />
                  <div className="config-info">
                    <div className="config-name">
                      {config.name}
                      {config.isRoot && <span className="config-root-badge">root</span>}
                    </div>
                    <div className="config-path">{config.path}</div>
                  </div>
                </label>
                {selectedPaths.has(config.path) && mode === "run" && (
                  <div className="config-command-picker">
                    <div className="config-command-picker-label">Command</div>
                    <div className="config-command-grid">
                      {ALL_COMMANDS.map((cmd) => (
                        <button
                          key={cmd.value}
                          type="button"
                          disabled={submitting}
                          className={`config-command-option ${commandOverrides[config.path] === cmd.value ? "active" : ""}`}
                          onClick={() => handleCommandOverride(config.path, cmd.value)}
                        >
                          <span className="config-command-option-label">{cmd.label}</span>
                          <span className="config-command-option-desc">{cmd.description}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="modal-actions-inline">
            <button
              type="button"
              onClick={handleSelectAll}
              disabled={submitting}
              className="btn btn-secondary"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={handleDeselectAll}
              disabled={submitting}
              className="btn btn-secondary"
            >
              Deselect All
            </button>
          </div>

          <div className="modal-actions">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="btn"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting || selectedPaths.size === 0}
              className="btn btn-primary"
            >
              {mode === "run"
                ? submitting
                  ? "Running..."
                  : "Run Selected"
                : submitting
                ? "Adding..."
                : "Add Selected"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

