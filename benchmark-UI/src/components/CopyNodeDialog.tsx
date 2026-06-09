import { useEffect, useMemo, useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";
import type { Workspace } from "../types";

interface Props {
  open: boolean;
  sourceWorkspace?: Workspace;
  sourceNodePath?: string;
  sourceNodeName?: string;
  destWorkspaces: Workspace[];
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (params: {
    destWorkspaceId: string;
    destRelPath: string;
    newName: string;
    overwrite: boolean;
  }) => Promise<void>;
}

function joinRel(parent: string, name: string): string {
  const p = parent.replace(/^[/\\]+|[/\\]+$/g, "");
  const n = name.replace(/^[/\\]+/, "");
  if (!p) return n;
  return `${p}/${n}`;
}

export function CopyNodeDialog({
  open,
  sourceWorkspace,
  sourceNodePath,
  sourceNodeName,
  destWorkspaces,
  submitting,
  onClose,
  onSubmit,
}: Props) {
  const [destWorkspaceId, setDestWorkspaceId] = useState<string>("");
  const [destRelPath, setDestRelPath] = useState<string>("");
  const [newName, setNewName] = useState<string>("");
  const [overwrite, setOverwrite] = useState(false);

  useEffect(() => {
    if (!open) return;
    setDestWorkspaceId(sourceWorkspace?.id ?? destWorkspaces[0]?.id ?? "");
    setDestRelPath("");
    setNewName(sourceNodeName ?? "");
    setOverwrite(false);
  }, [open, sourceWorkspace, sourceNodeName, destWorkspaces]);

  const destWorkspace = useMemo(
    () => destWorkspaces.find((w) => w.id === destWorkspaceId),
    [destWorkspaces, destWorkspaceId],
  );

  const isSameWorkspace =
    !!sourceWorkspace && sourceWorkspace.id === destWorkspaceId;

  // Disallow copying into the source folder itself or any of its descendants
  // when copying within the same workspace.
  const destInsideSource = (() => {
    if (!isSameWorkspace || !sourceNodePath) return false;
    const dest = joinRel(destRelPath, newName).toLowerCase();
    const src = sourceNodePath.replace(/^[/\\]+|[/\\]+$/g, "").toLowerCase();
    return dest === src || dest.startsWith(`${src}/`);
  })();

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!destWorkspaceId || !newName.trim() || destInsideSource) return;
    await onSubmit({
      destWorkspaceId,
      destRelPath: destRelPath.trim(),
      newName: newName.trim(),
      overwrite,
    });
  };

  const previewRel = joinRel(destRelPath, newName);
  const previewAbs =
    destWorkspace?.rootPath && previewRel
      ? `${destWorkspace.rootPath.replace(/[/\\]+$/, "")}/${previewRel}`
      : "";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Copy Folder</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit} autoComplete="off">
          <p>
            Copy <strong>{sourceNodeName}</strong> from{" "}
            <strong>{sourceWorkspace?.name}</strong> into another (or the same)
            workspace.
          </p>
          <p className="hint" style={{ fontSize: 12, opacity: 0.7 }}>
            Source: <code>{sourceNodePath}</code>
          </p>

          <label className="field">
            <span>Destination workspace</span>
            <select
              value={destWorkspaceId}
              onChange={(e) => setDestWorkspaceId(e.target.value)}
            >
              {destWorkspaces.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                  {w.id === sourceWorkspace?.id ? " (same)" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Destination subfolder (relative, blank = root)</span>
            <input
              value={destRelPath}
              onChange={(e) => setDestRelPath(e.target.value)}
              placeholder="e.g. inputs or experiments/run1"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name="copy-node-dest-parent"
            />
          </label>

          <label className="field">
            <span>New folder name</span>
            <input
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name="copy-node-name"
            />
          </label>

          <label
            className="field"
            style={{ flexDirection: "row", alignItems: "center", gap: 6 }}
          >
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(e) => setOverwrite(e.target.checked)}
            />
            <span>Overwrite if destination already exists</span>
          </label>

          {previewAbs && (
            <p className="hint" style={{ fontSize: 12, opacity: 0.7 }}>
              Will copy to: <code>{previewAbs}</code>
            </p>
          )}
          {destInsideSource && (
            <p className="hint" style={{ fontSize: 12, color: "var(--danger, #c33)" }}>
              Destination cannot be inside the source folder.
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
              disabled={
                submitting ||
                !destWorkspaceId ||
                !newName.trim() ||
                destInsideSource
              }
            >
              {submitting ? "Copying..." : "Copy"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
