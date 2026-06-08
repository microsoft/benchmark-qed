import { useEffect, useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

interface Props {
  open: boolean;
  onClose: () => void;
  onPickLocalFolder: () => Promise<string | null>;
  onAddLocal: (data: {
    path: string;
    hasChildWorkspaces: boolean;
    childWorkspacePaths: string[];
  }) => Promise<void>;
  onAddBlob: (data: {
    accountUrl: string;
    containerName: string;
    prefix: string;
    label: string;
  }) => void;
}

export function AddWorkspaceTabsDialog({
  open,
  onClose,
  onPickLocalFolder,
  onAddLocal,
  onAddBlob,
}: Props) {
  const [tab, setTab] = useState<"local" | "blob">("local");
  // Local fields
  const [localPath, setLocalPath] = useState("");
  const [localHasChildren, setLocalHasChildren] = useState(false);
  const [childWorkspacePaths, setChildWorkspacePaths] = useState<string[]>([]);
  const [pickingChildIndex, setPickingChildIndex] = useState<number | null>(null);
  const [pickingLocal, setPickingLocal] = useState(false);
  // Blob fields
  const [accountUrl, setAccountUrl] = useState("");
  const [containerName, setContainerName] = useState("");
  const [prefix, setPrefix] = useState("");
  const [label, setLabel] = useState("");

  const resetForm = () => {
    setTab("local");
    setLocalPath("");
    setLocalHasChildren(false);
    setChildWorkspacePaths([]);
    setPickingChildIndex(null);
    setPickingLocal(false);
    setAccountUrl("");
    setContainerName("");
    setPrefix("");
    setLabel("");
  };

  useEffect(() => {
    if (open) {
      resetForm();
    }
  }, [open]);

  if (!open) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (tab === "local") {
      if (!localPath.trim()) return;
      await onAddLocal({
        path: localPath.trim(),
        hasChildWorkspaces: localHasChildren,
        childWorkspacePaths: localHasChildren
          ? childWorkspacePaths.map((p) => p.trim()).filter(Boolean)
          : [],
      });
    } else {
      if (!accountUrl.trim() || !containerName.trim()) return;
      let computedLabel = label.trim();
      if (!computedLabel) {
        const url = accountUrl.trim().replace(/^https?:\/\//, "").replace(/\/+$/, "");
        computedLabel = `${url.split(".")[0]}/${containerName.trim()}${
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

  const handlePickLocalFolder = async () => {
    setPickingLocal(true);
    try {
      const pickedPath = await onPickLocalFolder();
      if (pickedPath) {
        setLocalPath(pickedPath);
      }
    } finally {
      setPickingLocal(false);
    }
  };

  const handleToggleHasChildren = (checked: boolean) => {
    setLocalHasChildren(checked);
    if (checked && childWorkspacePaths.length === 0) {
      setChildWorkspacePaths([""]);
    }
  };

  const handleAddChildRow = () => {
    setChildWorkspacePaths((prev) => [...prev, ""]);
  };

  const handleRemoveChildRow = (index: number) => {
    setChildWorkspacePaths((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpdateChildPath = (index: number, value: string) => {
    setChildWorkspacePaths((prev) =>
      prev.map((path, i) => (i === index ? value : path)),
    );
  };

  const handlePickChildFolder = async (index: number) => {
    setPickingChildIndex(index);
    try {
      const pickedPath = await onPickLocalFolder();
      if (pickedPath) {
        setChildWorkspacePaths((prev) =>
          prev.map((path, i) => (i === index ? pickedPath : path)),
        );
      }
    } finally {
      setPickingChildIndex(null);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Add Workspace</h3>
          <button className="modal-close" onClick={onClose}><Dismiss16Regular /></button>
        </div>
        <div className="segmented-toggle" style={{ margin: "0 14px" }}>
          <button
            type="button"
            className={tab === "local" ? "active" : ""}
            onClick={() => setTab("local")}
          >
            Local Folder
          </button>
          <button
            type="button"
            className={tab === "blob" ? "active" : ""}
            onClick={() => setTab("blob")}
          >
            Azure Blob
          </button>
        </div>
        <form className="modal-body" onSubmit={handleSubmit}>
          {tab === "local" && (
            <>
              <p>Pick a local folder to add as a workspace.</p>
              <label className="field">
                <span>Selected Folder</span>
                <div className="picker-row">
                  <input
                    type="text"
                    value={localPath}
                    placeholder="No folder selected"
                    readOnly
                  />
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void handlePickLocalFolder()}
                    disabled={pickingLocal}
                  >
                    {pickingLocal ? "Picking..." : "Pick Folder"}
                  </button>
                </div>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={localHasChildren}
                  onChange={e => handleToggleHasChildren(e.target.checked)}
                />
                <span>This folder contains child workspaces</span>
              </label>
              {localHasChildren && (
                <div className="field">
                  <span>Child workspace folders</span>
                  <div className="child-workspace-list">
                    {childWorkspacePaths.map((childPath, index) => (
                      <div key={`child-${index}`} className="child-workspace-row">
                        <input
                          type="text"
                          value={childPath}
                          onChange={e => handleUpdateChildPath(index, e.target.value)}
                          placeholder="path/to/child or pick folder"
                        />
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={() => void handlePickChildFolder(index)}
                          disabled={pickingChildIndex !== null}
                        >
                          {pickingChildIndex === index ? "Picking..." : "Pick Folder"}
                        </button>
                        <button
                          type="button"
                          className="btn"
                          onClick={() => handleRemoveChildRow(index)}
                          disabled={childWorkspacePaths.length <= 1}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                  <div>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={handleAddChildRow}
                    >
                      + Add Child Workspace
                    </button>
                  </div>
                  <small>
                    Use relative paths or absolute paths. Relative paths are resolved from the selected parent folder.
                  </small>
                </div>
              )}
            </>
          )}
          {tab === "blob" && (
            <>
              <label className="field">
                <span>Storage account URL</span>
                <input
                  type="text"
                  value={accountUrl}
                  onChange={e => setAccountUrl(e.target.value)}
                  placeholder="https://<account>.blob.core.windows.net"
                  required={tab === "blob"}
                />
              </label>
              <label className="field">
                <span>Container name</span>
                <input
                  type="text"
                  value={containerName}
                  onChange={e => setContainerName(e.target.value)}
                  placeholder="my-container"
                  required={tab === "blob"}
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
              <small>
                Uses the local runner's Azure managed identity or Azure CLI login.
                Start the runner from an environment that can access this storage account.
              </small>
            </>
          )}
          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>Cancel</button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={
                tab === "local"
                  ? !localPath.trim()
                  : !accountUrl.trim() || !containerName.trim()
              }
            >
              Add
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
