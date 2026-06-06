import { useState } from "react";
import { Dismiss16Regular } from "@fluentui/react-icons";

export type InitConfigType =
  | "autoq"
  | "autoe_pairwise"
  | "autoe_reference"
  | "autoe_assertion";

export interface InitRunRequest {
  configType: InitConfigType;
  rootPath: string;
  storageType: "local" | "blob";
  containerName?: string;
  accountUrl?: string;
  connectionString?: string;
  baseDir?: string;
}

interface Props {
  open: boolean;
  submitting: boolean;
  onClose: () => void;
  onSubmit: (data: InitRunRequest) => Promise<void>;
}

const CONFIG_OPTIONS: Array<{
  value: InitConfigType;
  label: string;
  description: string;
}> = [
  {
    value: "autoq",
    label: "AutoQ",
    description: "Question generation configuration.",
  },
  {
    value: "autoe_pairwise",
    label: "AutoE Pairwise",
    description: "Pairwise evaluation configuration.",
  },
  {
    value: "autoe_reference",
    label: "AutoE Reference",
    description: "Reference-based evaluation configuration.",
  },
  {
    value: "autoe_assertion",
    label: "AutoE Assertion",
    description: "Assertion scoring configuration.",
  },
];

export function InitRunDialog({ open, submitting, onClose, onSubmit }: Props) {
  const [configType, setConfigType] = useState<InitConfigType>("autoq");
  const [rootPath, setRootPath] = useState("");
  const [storageType, setStorageType] = useState<"local" | "blob">("local");
  const [containerName, setContainerName] = useState("");
  const [accountUrl, setAccountUrl] = useState("");
  const [connectionString, setConnectionString] = useState("");
  const [baseDir, setBaseDir] = useState("");
  const [pickingRoot, setPickingRoot] = useState(false);

  if (!open) return null;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    // For blob storage the CLI still wants a positional root argument but
    // does not use it when uploading directly to the container. Default to
    // "." (the runner's cwd) so the user doesn't have to think about it.
    const effectiveRootPath =
      storageType === "blob" ? (rootPath.trim() || ".") : rootPath.trim();
    if (!effectiveRootPath) return;

    await onSubmit({
      configType,
      rootPath: effectiveRootPath,
      storageType,
      containerName: containerName.trim() || undefined,
      accountUrl: accountUrl.trim() || undefined,
      connectionString: connectionString.trim() || undefined,
      baseDir: baseDir.trim() || undefined,
    });
  };

  // The CLI's `benchmark-qed config init` always takes a local positional
  // root argument, but its meaning changes with --storage-type:
  //   * local                                 → settings.yaml/.env/prompts/
  //                                             are written under <root>/
  //   * blob                                  → the positional root argument
  //                                             is unused when an account-url
  //                                             or connection-string is
  //                                             provided (files are uploaded
  //                                             directly to the container);
  //                                             without auth it falls back to
  //                                             local scaffolding only.
  // For blob mode the field is hidden and submit() substitutes "." so the
  // user doesn't have to invent a meaningless path.

  const pickRootPath = async () => {
    setPickingRoot(true);
    try {
      const res = await fetch("http://localhost:8787/api/pick-folder");
      const payload = (await res.json()) as {
        path?: string;
        cancelled?: boolean;
        error?: string;
      };
      if (payload.cancelled) {
        return;
      }
      if (!res.ok || !payload.path) {
        throw new Error(payload.error ?? "Failed to pick folder.");
      }
      setRootPath(payload.path);
    } catch (err) {
      alert(
        `Folder picker is unavailable. Ensure init runner is running. ${String(err)}`,
      );
    } finally {
      setPickingRoot(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Create Configuration</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <form className="modal-body" onSubmit={submit}>
          <section className="option-group">
            <h4>Configuration Type</h4>
            <div className="option-grid">
              {CONFIG_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`option-card ${configType === opt.value ? "active" : ""}`}
                  onClick={() => setConfigType(opt.value)}
                >
                  <span className="option-card-label">{opt.label}</span>
                  <span className="option-card-desc">{opt.description}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="option-group">
            <h4>Storage Mode</h4>
            <div className="segmented-toggle">
              <button
                type="button"
                className={storageType === "local" ? "active" : ""}
                onClick={() => setStorageType("local")}
              >
                Local
              </button>
              <button
                type="button"
                className={storageType === "blob" ? "active" : ""}
                onClick={() => setStorageType("blob")}
              >
                Azure Blob
              </button>
            </div>
          </section>

          {storageType === "local" && (
            <section className="option-group option-group-muted">
              <h4>Local Details</h4>
              <label className="field">
                <span>Root folder path</span>
                <div className="picker-row">
                  <input
                    type="text"
                    value={rootPath}
                    onChange={(e) => setRootPath(e.target.value)}
                    placeholder="/absolute/path/to/new-run-folder"
                    required
                  />
                  <button
                    type="button"
                    className="btn"
                    onClick={pickRootPath}
                    disabled={pickingRoot}
                  >
                    {pickingRoot ? "Picking..." : "Pick Folder"}
                  </button>
                </div>
                <small>
                  Folder on the machine running the local runner where the
                  workspace scaffold (settings.yaml, prompts/, .env, input/)
                  will be created.
                </small>
              </label>
            </section>
          )}

          {storageType === "blob" && (
            <section className="option-group option-group-muted">
              <h4>Blob Details</h4>
              <label className="field">
                <span>Container name</span>
                <input
                  type="text"
                  value={containerName}
                  onChange={(e) => setContainerName(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Account URL (optional)</span>
                <input
                  type="text"
                  value={accountUrl}
                  onChange={(e) => setAccountUrl(e.target.value)}
                  placeholder="https://<account>.blob.core.windows.net"
                />
              </label>
              <label className="field">
                <span>Connection string (optional)</span>
                <input
                  type="text"
                  value={connectionString}
                  onChange={(e) => setConnectionString(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Base dir (optional)</span>
                <input
                  type="text"
                  value={baseDir}
                  onChange={(e) => setBaseDir(e.target.value)}
                />
              </label>
              <div className="modal-callout modal-callout-warning" role="note">
                <strong>Heads up:</strong> blob workspaces are not
                automatically mounted. After this job succeeds, click
                <strong> + Add Workspace</strong> and connect to the container
                with a SAS URL to browse it in the sidebar.
              </div>
            </section>
          )}

          <div className="cors-hint">
            Requires local runner: run <code>npm run init-runner</code> in benchmark-UI.
          </div>

          <div className="modal-actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? "Creating..." : "Create in background"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
