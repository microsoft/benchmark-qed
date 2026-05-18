import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

export interface BlobConnectSubmit {
  sasUrl: string;
  prefix: string;
  label: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: BlobConnectSubmit) => void;
}

export function BlobConnectDialog({ open, onClose, onSubmit }: Props) {
  const [sasUrl, setSasUrl] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
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
    onSubmit({
      sasUrl: sasUrl.trim(),
      prefix: prefix.trim(),
      label: computedLabel,
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Connect to Azure Blob Container</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={handleSubmit}>
          <label className="field">
            <span>Container SAS URL</span>
            <textarea
              value={sasUrl}
              onChange={(e) => setSasUrl(e.target.value)}
              placeholder="https://<account>.blob.core.windows.net/<container>?sv=...&sig=..."
              required
              rows={3}
            />
            <small>
              Generate via Azure Portal → Container → "Shared access tokens".
              Include <code>r</code> + <code>l</code> permissions; add{" "}
              <code>w</code> + <code>c</code> for editing.
            </small>
          </label>
          <label className="field">
            <span>Prefix (optional)</span>
            <input
              type="text"
              value={prefix}
              onChange={(e) => setPrefix(e.target.value)}
              placeholder="path/inside/container"
            />
          </label>
          <label className="field">
            <span>Display name (optional)</span>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="my-init-run"
            />
          </label>
          <div className="cors-hint">
            <strong>CORS required:</strong> storage account must allow{" "}
            <code>{window.location.origin}</code> with methods{" "}
            <code>GET, HEAD, PUT, OPTIONS</code> in the Blob service CORS rules.
          </div>
          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary">
              Connect
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
