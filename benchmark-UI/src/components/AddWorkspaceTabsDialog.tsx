import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  onClose: () => void;
  onPickLocalFolder: () => Promise<string | null>;
  onAddLocal: (path: string) => Promise<void>;
  onAddBlob: (data: { sasUrl: string; prefix: string; label: string }) => void;
}

export function AddWorkspaceTabsDialog({
  open,
  onClose,
  onPickLocalFolder,
  onAddLocal,
  onAddBlob,
}: Props) {
  const [tab, setTab] = useState<"local" | "blob">("local");
  // Local fields
  const [localPath, setLocalPath] = useState("");
  const [pickingLocal, setPickingLocal] = useState(false);
  // Blob fields
  const [sasUrl, setSasUrl] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  if (!open) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (tab === "local") {
      if (!localPath.trim()) return;
      await onAddLocal(localPath.trim());
    } else {
      if (!sasUrl.trim()) return;
      let computedLabel = label.trim();
      if (!computedLabel) {
        try {
          const u = new URL(sasUrl);
          const container = u.pathname.replace(/^\//, "").split("/")[0];
          computedLabel = `${u.hostname.split(".")[0]}/${container}${
            prefix ? `/${prefix.replace(/\/$/, "")}` : ""
          }`;
        } catch {
          computedLabel = "blob-container";
        }
      }
      onAddBlob({ sasUrl: sasUrl.trim(), prefix: prefix.trim(), label: computedLabel });
    }
  };

  const handlePickLocalFolder = async () => {
    setPickingLocal(true);
    try {
      const pickedPath = await onPickLocalFolder();
      if (pickedPath) {
        setLocalPath(pickedPath);
      }
    } finally {
      setPickingLocal(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Add Workspace</h3>
          <button className="modal-close" onClick={onClose}><Dismiss16Regular /></button>
        </div>
        <div className="segmented-toggle" style={{ margin: "0 14px" }}>
          <button
            type="button"
            className={tab === "local" ? "active" : ""}
            onClick={() => setTab("local")}
          >
            Local Folder
          </button>
          <button
            type="button"
            className={tab === "blob" ? "active" : ""}
            onClick={() => setTab("blob")}
          >
            Azure Blob
          </button>
        </div>
        <form className="modal-body" onSubmit={handleSubmit}>
          {tab === "local" && (
            <>
              <p>Pick a local folder to add as a workspace.</p>
              <label className="field">
                <span>Selected Folder</span>
                <div className="picker-row">
                  <input
                    type="text"
                    value={localPath}
                    placeholder="No folder selected"
                    readOnly
                  />
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void handlePickLocalFolder()}
                    disabled={pickingLocal}
                  >
                    {pickingLocal ? "Picking..." : "Pick Folder"}
                  </button>
                </div>
              </label>
            </>
          )}
          {tab === "blob" && (
            <>
              <label className="field">
                <span>Container SAS URL</span>
                <textarea
                  value={sasUrl}
                  onChange={e => setSasUrl(e.target.value)}
                  placeholder="https://<account>.blob.core.windows.net/<container>?sv=...&sig=..."
                  required={tab === "blob"}
                  rows={3}
                />
              </label>
              <label className="field">
                <span>Prefix (optional)</span>
                <input
                  type="text"
                  value={prefix}
                  onChange={e => setPrefix(e.target.value)}
                  placeholder="path/inside/container"
                />
              </label>
              <label className="field">
                <span>Display name (optional)</span>
                <input
                  type="text"
                  value={label}
                  onChange={e => setLabel(e.target.value)}
                  placeholder="my-init-run"
                />
              </label>
            </>
          )}
          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>Cancel</button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={tab === "local" ? !localPath.trim() : !sasUrl.trim()}
            >
              Add
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
