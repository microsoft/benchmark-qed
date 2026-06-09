import { useEffect, useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  kind: "file" | "directory";
  currentName: string;
  parentPath: string;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (newName: string) => Promise<void> | void;
}

export function RenameNodeDialog({
  open,
  kind,
  currentName,
  parentPath,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const [name, setName] = useState(currentName);

  useEffect(() => {
    if (open) setName(currentName);
  }, [open, currentName]);

  if (!open) return null;

  const trimmed = name.trim();
  const invalid = !trimmed || /[\\/]/.test(trimmed);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (invalid || trimmed === currentName) {
      onClose();
      return;
    }
    await onSubmit(trimmed);
  };

  const title = kind === "file" ? "Rename File" : "Rename Folder";
  const noun = kind === "file" ? "file" : "folder";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit} autoComplete="off">
          <p>
            Rename {noun} <strong>{currentName}</strong> in{" "}
            <strong>{parentPath || "workspace root"}</strong>.
          </p>
          <label className="field">
            <span>New name</span>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name="rename-node-name"
            />
          </label>
          {invalid && name.length > 0 && (
            <p className="hint" style={{ fontSize: 12, color: "var(--danger, #c33)" }}>
              Name cannot be empty or contain slashes.
            </p>
          )}

          <div className="modal-actions">
            <button
              type="button"
              className="btn"
              onClick={onClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || invalid || trimmed === currentName}
            >
              {submitting ? "Renaming..." : "Rename"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
