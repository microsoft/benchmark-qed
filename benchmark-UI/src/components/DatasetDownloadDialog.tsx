import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

export type PredefinedDataset = "AP_news" | "podcast" | "example_answers";

interface Props {
  open: boolean;
  workspaceName: string;
  workspaceRootPath?: string;
  submitting: boolean;
  onClose: () => void;
  onSubmit: (
    dataset: PredefinedDataset,
    destinationPath: string,
  ) => Promise<void>;
}

const DATASET_OPTIONS: Array<{
  value: PredefinedDataset;
  label: string;
  description: string;
}> = [
  {
    value: "AP_news",
    label: "AP-news",
    description: "Health-focused Associated Press news corpus.",
  },
  {
    value: "podcast",
    label: "Podcast",
    description: "Behind the Tech transcript corpus.",
  },
  {
    value: "example_answers",
    label: "Example answers",
    description: "Sample answer files for notebook examples.",
  },
];

export function DatasetDownloadDialog({
  open,
  workspaceName,
  workspaceRootPath,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const [dataset, setDataset] = useState<PredefinedDataset>("AP_news");
  const [destinationPath, setDestinationPath] = useState("input");
  const [pickingDestination, setPickingDestination] = useState(false);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(dataset, destinationPath.trim());
  };

  const selectDataset = (nextDataset: PredefinedDataset) => {
    setDataset(nextDataset);
  };

  const pickDestination = async () => {
    setPickingDestination(true);
    try {
      const res = await fetch("http://localhost:8787/api/pick-folder");
      const payload = (await res.json()) as {
        path?: string;
        cancelled?: boolean;
        error?: string;
      };
      if (payload.cancelled) return;
      if (!res.ok || !payload.path) {
        throw new Error(payload.error ?? "Failed to pick destination folder.");
      }
      setDestinationPath(payload.path);
    } catch (err) {
      alert(
        `Destination picker is unavailable. Ensure init runner is running. ${String(err)}`,
      );
    } finally {
      setPickingDestination(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Download Predefined Input</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit}>
          <p>
            Select a dataset to download into <strong>{workspaceName}</strong>.
          </p>

          <section className="option-group">
            <h4>Dataset</h4>
            <div className="option-grid">
              {DATASET_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`option-card ${dataset === opt.value ? "active" : ""}`}
                  onClick={() => selectDataset(opt.value)}
                >
                  <span className="option-card-label">{opt.label}</span>
                  <span className="option-card-desc">{opt.description}</span>
                </button>
              ))}
            </div>
          </section>

          <label className="field">
            <span>Destination folder</span>
            <div className="picker-row">
              <input
                type="text"
                value={destinationPath}
                onChange={(e) => setDestinationPath(e.target.value)}
                placeholder={workspaceRootPath || "Workspace root"}
              />
              <button
                type="button"
                className="btn"
                onClick={pickDestination}
                disabled={pickingDestination}
              >
                {pickingDestination ? "Picking..." : "Pick Folder"}
              </button>
            </div>
            <small>
              Leave empty to use the workspace root. Relative paths resolve from
              the workspace root.
            </small>
          </label>

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? "Downloading..." : "Download"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
