import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

export interface BlobConnectSubmit {
  accountUrl: string;
  containerName: string;
  prefix: string;
  label: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: BlobConnectSubmit) => void;
}

export function BlobConnectDialog({ open, onClose, onSubmit }: Props) {
  const [accountUrl, setAccountUrl] = useState("");
  const [containerName, setContainerName] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!accountUrl.trim() || !containerName.trim()) return;
    let computedLabel = label.trim();
    if (!computedLabel) {
      const host = accountUrl.trim().replace(/^https?:\/\//, "").replace(/\/+$/, "");
      computedLabel = `${host.split(".")[0]}/${containerName.trim()}${
        prefix ? `/${prefix.replace(/\/$/, "")}` : ""
      }`;
    }
    onSubmit({
      accountUrl: accountUrl.trim(),
      containerName: containerName.trim(),
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
            <span>Storage account URL</span>
            <input
              type="text"
              value={accountUrl}
              onChange={(e) => setAccountUrl(e.target.value)}
              placeholder="https://<account>.blob.core.windows.net"
              required
            />
          </label>
          <label className="field">
            <span>Container name</span>
            <input
              type="text"
              value={containerName}
              onChange={(e) => setContainerName(e.target.value)}
              placeholder="my-container"
              required
            />
          </label>
          <label className="field">
            <span>Prefix / root (optional)</span>
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
            Uses the local runner's Azure managed identity or Azure CLI login.
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
