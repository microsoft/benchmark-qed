import { useEffect, useState } from "react";
import { Dismiss16Regular, Folder16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  sourceName: string;
  sourcePath: string;
  defaultParentDir: string;
  submitting?: boolean;
  pickFolder: () => Promise<string | null>;
  onClose: () => void;
  onSubmit: (params: {
    destPath: string;
    workspaceName: string;
    overwrite: boolean;
  }) => Promise<void>;
}

function joinPath(parent: string, name: string): string {
  if (!parent) return name;
  const trimmedParent = parent.replace(/[/\\]+$/, "");
  const trimmedName = name.replace(/^[/\\]+/, "");
  const sep = parent.includes("\\") && !parent.includes("/") ? "\\" : "/";
  return `${trimmedParent}${sep}${trimmedName}`;
}

export function CopyWorkspaceDialog({
  open,
  sourceName,
  sourcePath,
  defaultParentDir,
  submitting,
  pickFolder,
  onClose,
  onSubmit,
}: Props) {
  const [parentDir, setParentDir] = useState(defaultParentDir);
  const [folderName, setFolderName] = useState(`${sourceName}-copy`);
  const [workspaceName, setWorkspaceName] = useState(`${sourceName} (copy)`);
  const [overwrite, setOverwrite] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Reset whenever the dialog re-opens for a (potentially different) source.
  useEffect(() => {
    if (!open) return;
    setParentDir(defaultParentDir);
    setFolderName(`${sourceName}-copy`);
    setWorkspaceName(`${sourceName} (copy)`);
    setOverwrite(false);
    setSubmitError(null);
  }, [open, defaultParentDir, sourceName]);

  if (!open) return null;

  const destPath = parentDir && folderName ? joinPath(parentDir, folderName) : "";

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!destPath || !workspaceName.trim()) return;
    setSubmitError(null);
    try {
      await onSubmit({
        destPath,
        workspaceName: workspaceName.trim(),
        overwrite,
      });
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  };

  const handlePick = async () => {
    const picked = await pickFolder();
    if (picked) setParentDir(picked);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Copy Workspace</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit} autoComplete="off">
          <p>
            Copy <strong>{sourceName}</strong> to a new location and load it as
            a new workspace.
          </p>
          <p className="hint" style={{ fontSize: 12, opacity: 0.7 }}>
            Source: <code>{sourcePath}</code>
          </p>

          <label className="field">
            <span>Destination parent folder</span>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                value={parentDir}
                onChange={(e) => {
                  setParentDir(e.target.value);
                  setSubmitError(null);
                }}
                placeholder="/absolute/path/to/parent"
                autoComplete="off"
                spellCheck={false}
                style={{ flex: 1 }}
              />
              <button
                type="button"
                className="btn"
                onClick={() => void handlePick()}
                title="Pick folder"
              >
                <Folder16Regular /> Browse
              </button>
            </div>
          </label>

          <label className="field">
            <span>New folder name</span>
            <input
              value={folderName}
              onChange={(e) => {
                setFolderName(e.target.value);
                setSubmitError(null);
              }}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              name="copy-workspace-folder-name"
            />
          </label>

          <label className="field">
            <span>Workspace display name</span>
            <input
              value={workspaceName}
              onChange={(e) => setWorkspaceName(e.target.value)}
              autoComplete="off"
              spellCheck={false}
              name="copy-workspace-display-name"
            />
          </label>

          <label
            className="field"
            style={{ flexDirection: "row", alignItems: "center", gap: 6 }}
          >
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(e) => {
                setOverwrite(e.target.checked);
                if (e.target.checked) setSubmitError(null);
              }}
            />
            <span>Overwrite if destination already exists</span>
          </label>

          {destPath && (
            <p className="hint" style={{ fontSize: 12, opacity: 0.7 }}>
              Will copy to: <code>{destPath}</code>
            </p>
          )}

          {submitError && (() => {
            const isExists = /already exists/i.test(submitError);
            return (
              <div
                role="alert"
                style={{
                  marginTop: 4,
                  padding: "10px 12px",
                  borderRadius: 6,
                  background: "rgba(220, 80, 80, 0.10)",
                  border: "1px solid rgba(220, 80, 80, 0.45)",
                  color: "var(--text, inherit)",
                  fontSize: 13,
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                {isExists ? (
                  <>
                    <strong>That destination already has a folder.</strong>
                    <span style={{ opacity: 0.85 }}>
                      A folder named <code>{folderName}</code> already exists in
                      <br />
                      <code>{parentDir}</code>.
                    </span>
                    <span style={{ opacity: 0.85 }}>
                      Pick a different name, choose another parent folder, or
                      check <em>Overwrite</em> above to replace it.
                    </span>
                    {!overwrite && (
                      <div>
                        <button
                          type="button"
                          className="btn"
                          onClick={() => {
                            setOverwrite(true);
                            setSubmitError(null);
                          }}
                        >
                          Enable overwrite
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <span>{submitError}</span>
                )}
              </div>
            );
          })()}

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
              disabled={submitting || !destPath || !workspaceName.trim()}
            >
              {submitting ? "Copying..." : "Copy & Load"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
