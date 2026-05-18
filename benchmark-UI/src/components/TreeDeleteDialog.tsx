import { Dismiss16Regular } from "@fluentui/react-icons";
import type { TreeNode } from "../types";

interface Props {
  open: boolean;
  node?: TreeNode;
  submitting?: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}

export function TreeDeleteDialog({
  open,
  node,
  submitting,
  onClose,
  onConfirm,
}: Props) {
  if (!open || !node) return null;

  const confirmDelete = async (e: React.FormEvent) => {
    e.preventDefault();
    await onConfirm();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Delete {node.kind}</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={confirmDelete}>
          <p>
            Delete <strong>{node.path}</strong>?
          </p>
          <p>This action cannot be undone.</p>

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="submit" className="btn btn-danger" disabled={submitting}>
              {submitting ? "Deleting..." : "Delete"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
