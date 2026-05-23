import { useEffect, useMemo, useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";
import type { TreeNode } from "../types";

interface Props {
  open: boolean;
  kind: "file" | "directory";
  parentNode?: TreeNode;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void>;
}

export function TreeCreateDialog({
  open,
  kind,
  parentNode,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const defaultName = kind === "file" ? "new-file.txt" : "new-folder";
  const [name, setName] = useState(defaultName);

  // Reset the input whenever the dialog re-opens or switches kind so it
  // doesn't carry over text from a previous create.
  useEffect(() => {
    if (open) setName(defaultName);
  }, [open, defaultName]);

  const title = kind === "file" ? "Create File" : "Create Folder";
  const noun = kind === "file" ? "file" : "folder";

  const parentLabel = useMemo(() => {
    if (!parentNode) return "workspace root";
    return parentNode.path;
  }, [parentNode]);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(name);
  };

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
            Create a new {noun} inside <strong>{parentLabel}</strong>.
          </p>
          <label className="field">
            <span>Name</span>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={kind === "file" ? "new-file.txt" : "new-folder"}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name={
                kind === "file"
                  ? "tree-create-file-name"
                  : "tree-create-folder-name"
              }
            />
          </label>

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
