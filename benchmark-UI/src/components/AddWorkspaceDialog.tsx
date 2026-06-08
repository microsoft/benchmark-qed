import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  onClose: () => void;
  onAddLocal: () => void;
  onAddBlob: (data: {
    accountUrl: string;
    containerName: string;
    prefix: string;
    label: string;
  }) => void;
}

export function AddWorkspaceDialog({ open, onClose, onAddLocal, onAddBlob }: Props) {
  const [type, setType] = useState<"local" | "blob">("local");
  // Blob fields
  const [accountUrl, setAccountUrl] = useState("");
  const [containerName, setContainerName] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (type === "local") {
      onAddLocal();
    } else {
      if (!accountUrl.trim() || !containerName.trim()) return;
      let computedLabel = label.trim();
      if (!computedLabel) {
        const host = accountUrl.trim().replace(/^https?:\/\//, "").replace(/\/+$/, "");
        computedLabel = `${host.split(".")[0]}/${containerName.trim()}${
          prefix ? `/${prefix.replace(/\/$/, "")}` : ""
        }`;
      }
      onAddBlob({
        accountUrl: accountUrl.trim(),
        containerName: containerName.trim(),
        prefix: prefix.trim(),
        label: computedLabel,
      });
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
                <span>Storage account URL</span>
                <input
                  type="text"
                  value={accountUrl}
                  onChange={e => setAccountUrl(e.target.value)}
                  placeholder="https://<account>.blob.core.windows.net"
                  required={type === "blob"}
                />
              </label>
              <label className="field">
                <span>Container name</span>
                <input
                  type="text"
                  value={containerName}
                  onChange={e => setContainerName(e.target.value)}
                  placeholder="my-container"
                  required={type === "blob"}
                />
              </label>
              <label className="field">
                <span>Prefix / root (optional)</span>
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
