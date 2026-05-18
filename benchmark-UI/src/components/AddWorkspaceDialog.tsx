import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  onClose: () => void;
  onAddLocal: () => void;
  onAddBlob: (data: { sasUrl: string; prefix: string; label: string }) => void;
}

export function AddWorkspaceDialog({ open, onClose, onAddLocal, onAddBlob }: Props) {
  const [type, setType] = useState<"local" | "blob">("local");
  // Blob fields
  const [sasUrl, setSasUrl] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (type === "local") {
      onAddLocal();
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

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Add Workspace</h3>
          <button className="modal-close" onClick={onClose}><Dismiss16Regular /></button>
        </div>
        <form className="modal-body" onSubmit={handleSubmit}>
          <label className="field">
            <span>Workspace Type</span>
            <select value={type} onChange={e => setType(e.target.value as "local" | "blob")}> 
              <option value="local">Local Folder</option>
              <option value="blob">Azure Blob Container</option>
            </select>
          </label>
          {type === "blob" && (
            <>
              <label className="field">
                <span>Container SAS URL</span>
                <textarea
                  value={sasUrl}
                  onChange={e => setSasUrl(e.target.value)}
                  placeholder="https://<account>.blob.core.windows.net/<container>?sv=...&sig=..."
                  required={type === "blob"}
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
            <button type="submit" className="btn btn-primary">Add</button>
          </div>
        </form>
      </div>
    </div>
  );
}
