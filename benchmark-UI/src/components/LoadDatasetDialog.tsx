import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

export interface LoadDatasetSubmit {
  sourceFolder: string;
  destinationFolder: string;
  workspaceName: string;
  subdir: string;
}

interface Props {
  open: boolean;
  submitting: boolean;
  onClose: () => void;
  onSubmit: (payload: LoadDatasetSubmit) => Promise<void>;
}

type Picker = "source" | "destination" | null;

export function LoadDatasetDialog({ open, submitting, onClose, onSubmit }: Props) {
  const [sourceFolder, setSourceFolder] = useState("");
  const [destinationFolder, setDestinationFolder] = useState("");
  const [workspaceName, setWorkspaceName] = useState("");
  const [subdir, setSubdir] = useState("input");
  const [picking, setPicking] = useState<Picker>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const reset = () => {
    setSourceFolder("");
    setDestinationFolder("");
    setWorkspaceName("");
    setSubdir("input");
    setError(null);
  };

  const close = () => {
    reset();
    onClose();
  };

  const pickFolder = async (which: "source" | "destination") => {
    setPicking(which);
    setError(null);
    try {
      const res = await fetch("http://localhost:8787/api/pick-folder");
      const payload = (await res.json()) as {
        path?: string;
        cancelled?: boolean;
        error?: string;
      };
      if (payload.cancelled) return;
      if (!res.ok || !payload.path) {
        throw new Error(payload.error ?? "Failed to pick folder.");
      }
      if (which === "source") {
        setSourceFolder(payload.path);
      } else {
        setDestinationFolder(payload.path);
        if (!workspaceName.trim()) {
          const parts = payload.path.split(/[/\\]/).filter(Boolean);
          if (parts.length) setWorkspaceName(parts[parts.length - 1]);
        }
      }
    } catch (e) {
      setError(
        `Folder picker is unavailable. Ensure init runner is running. ${String(e)}`,
      );
    } finally {
      setPicking(null);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const source = sourceFolder.trim();
    const folder = destinationFolder.trim();
    if (!source) {
      setError("Pick the dataset root folder.");
      return;
    }
    if (!folder) {
      setError("Pick a destination folder for the new workspace.");
      return;
    }
    setError(null);
    await onSubmit({
      sourceFolder: source,
      destinationFolder: folder,
      workspaceName:
        workspaceName.trim() ||
        folder.split(/[/\\]/).filter(Boolean).pop() ||
        folder,
      subdir: subdir.trim() || "input",
    });
  };

  return (
    <div className="modal-backdrop" onClick={close}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Load Dataset</h3>
          <button className="modal-close" onClick={close}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit}>
          <p>
            Pick a dataset root folder and a destination workspace folder. All
            subfolders and files inside the source are copied recursively into{" "}
            <code>
              {destinationFolder
                ? `${destinationFolder}/${subdir || "input"}`
                : `<workspace>/${subdir || "input"}`}
            </code>
            , and the workspace appears in the sidebar tree.
          </p>

          <label className="field">
            <span>Source dataset root folder</span>
            <div className="picker-row">
              <input
                type="text"
                value={sourceFolder}
                onChange={(e) => setSourceFolder(e.target.value)}
                placeholder="Pick the folder containing your dataset (subfolders will be imported)"
              />
              <button
                type="button"
                className="btn"
                onClick={() => pickFolder("source")}
                disabled={picking === "source"}
              >
                {picking === "source" ? "Picking..." : "Browse..."}
              </button>
            </div>
            <small className="field-hint">
              The folder itself is not copied — only its direct children
              (subfolders and files) land inside the destination's input
              subfolder.
            </small>
          </label>

          <label className="field">
            <span>Destination workspace folder</span>
            <div className="picker-row">
              <input
                type="text"
                value={destinationFolder}
                onChange={(e) => setDestinationFolder(e.target.value)}
                placeholder="Pick or type an absolute folder path"
              />
              <button
                type="button"
                className="btn"
                onClick={() => pickFolder("destination")}
                disabled={picking === "destination"}
              >
                {picking === "destination" ? "Picking..." : "Browse..."}
              </button>
            </div>
            <small className="field-hint">
              Created if it doesn't exist. Becomes the new workspace root.
            </small>
          </label>

          <label className="field">
            <span>Workspace name</span>
            <input
              type="text"
              value={workspaceName}
              onChange={(e) => setWorkspaceName(e.target.value)}
              placeholder="Defaults to destination folder name"
            />
          </label>

          <label className="field">
            <span>Input subfolder</span>
            <input
              type="text"
              value={subdir}
              onChange={(e) => setSubdir(e.target.value)}
              placeholder="input"
            />
            <small className="field-hint">
              Relative to the workspace folder. Use <code>input</code> for the
              default benchmark-qed layout.
            </small>
          </label>

          {error && <div className="error-banner">{error}</div>}

          <div className="modal-actions">
            <button
              type="button"
              className="btn"
              onClick={close}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                submitting || !sourceFolder.trim() || !destinationFolder.trim()
              }
            >
              {submitting ? "Importing..." : "Load Dataset"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
