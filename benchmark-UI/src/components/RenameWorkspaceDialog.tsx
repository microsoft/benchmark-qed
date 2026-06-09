import { useEffect, useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  currentName: string;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void> | void;
}

export function RenameWorkspaceDialog({
  open,
  currentName,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const [name, setName] = useState(currentName);

  useEffect(() => {
    if (open) setName(currentName);
  }, [open, currentName]);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || trimmed === currentName) {
      onClose();
      return;
    }
    await onSubmit(trimmed);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Rename Workspace</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit} autoComplete="off">
          <p>
            Rename only the sidebar label. The folder on disk is not changed.
          </p>
          <label className="field">
            <span>Name</span>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={currentName}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name="rename-workspace-name"
            />
          </label>

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
              disabled={submitting || !name.trim()}
            >
              {submitting ? "Renaming..." : "Rename"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
