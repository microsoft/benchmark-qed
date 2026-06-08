import { useEffect, useMemo, useState } from "react";
import {
  Add16Regular,
  Dismiss16Regular,
  Folder16Regular,
  DocumentArrowLeft16Regular,
} from "@fluentui/react-icons";
import {
  listRecentReports,
  removeRecentReport,
  type RecentReport,
} from "../recentReports";

interface FolderEntry {
  id: string;
  label: string;
  path: string;
  /** True until the user manually edits the label — keeps it in sync with path. */
  labelAuto: boolean;
}

export interface EvaluateQuestionsSubmit {
  entries: { label: string; path: string }[];
  /** Absolute folder path where the report should be written. */
  destinationPath: string;
  /** File name (just the basename, e.g. `quality_report.md`). */
  reportFilename: string;
}

interface WorkspaceOption {
  id: string;
  name: string;
  rootPath: string;
}

interface Props {
  open: boolean;
  workspaces: WorkspaceOption[];
  pickFolder: () => Promise<string | null>;
  onClose: () => void;
  onSubmit: (params: EvaluateQuestionsSubmit) => void;
}

function basename(p: string): string {
  if (!p) return "";
  return p.split(/[/\\]/).filter(Boolean).pop() ?? p;
}

function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function newRow(label = "", path = ""): FolderEntry {
  return { id: makeId(), label, path, labelAuto: !label };
}

/**
 * Inline form (no modal chrome) for picking N folders to compare. Used
 * both by the standalone dialog wrapper below and embedded directly inside
 * the Copilot panel.
 */
export function EvaluateQuestionsPickerForm({
  workspaces,
  pickFolder,
  onCancel,
  onSubmit,
  onOpenReport,
  resetKey,
}: {
  workspaces: WorkspaceOption[];
  pickFolder: () => Promise<string | null>;
  onCancel: () => void;
  onSubmit: (params: EvaluateQuestionsSubmit) => void;
  /** Called when the user clicks a recent report. The host should open
   *  the markdown file in its editor area. */
  onOpenReport?: (report: RecentReport) => void;
  /** Bumping this key resets the row state (used to clear stale rows
   *  whenever the form is re-shown). */
  resetKey?: number | string;
}) {
  const [rows, setRows] = useState<FolderEntry[]>(() => []);
  const [activeBrowseId, setActiveBrowseId] = useState<string | null>(null);
  // The workspace dropdown is a staging area: picking an option just
  // primes the selection. Nothing is added to the row list until the user
  // clicks the adjacent "Add" button.
  const [pendingWorkspaceId, setPendingWorkspaceId] = useState("");
  // Where to save the QUALITY_REPORT.md. Edits track the first row by
  // default until the user manually changes it.
  const [destinationPath, setDestinationPath] = useState("");
  const [destinationAuto, setDestinationAuto] = useState(true);
  const [destBrowsing, setDestBrowsing] = useState(false);
  const [destWorkspaceId, setDestWorkspaceId] = useState("");
  const [reportFilename, setReportFilename] = useState("QUALITY_REPORT.md");
  const [recentReports, setRecentReports] = useState<RecentReport[]>(() =>
    listRecentReports(),
  );

  useEffect(() => {
    setRows([]);
    setDestinationPath("");
    setDestinationAuto(true);
    setDestWorkspaceId("");
    setReportFilename("QUALITY_REPORT.md");
    setRecentReports(listRecentReports());
  }, [resetKey]);

  const workspaceOptions = useMemo(
    () => workspaces.filter((w) => w.rootPath),
    [workspaces],
  );

  const updateRow = (id: string, patch: Partial<FolderEntry>) => {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const updatePath = (id: string, path: string) => {
    setRows((prev) =>
      prev.map((r) => {
        if (r.id !== id) return r;
        const nextLabel = r.labelAuto ? basename(path) : r.label;
        return { ...r, path, label: nextLabel };
      }),
    );
  };

  // Keep the auto-tracked destination in sync with the first row's path.
  useEffect(() => {
    if (!destinationAuto) return;
    const firstPath = rows.find((r) => r.path.trim())?.path.trim() ?? "";
    if (firstPath !== destinationPath) {
      setDestinationPath(firstPath);
    }
  }, [rows, destinationAuto, destinationPath]);

  const handleLabelChange = (id: string, label: string) => {
    updateRow(id, { label, labelAuto: false });
  };

  const handlePick = async (id: string) => {
    setActiveBrowseId(id);
    try {
      const picked = await pickFolder();
      if (picked) updatePath(id, picked);
    } finally {
      setActiveBrowseId(null);
    }
  };

  const addRow = () => setRows((prev) => [...prev, newRow()]);
  const removeRow = (id: string) =>
    setRows((prev) => prev.filter((r) => r.id !== id));

  const addFromWorkspace = (workspaceId: string) => {
    const ws = workspaceOptions.find((w) => w.id === workspaceId);
    if (!ws) return;
    setRows((prev) => [...prev, newRow(ws.name, ws.rootPath)]);
  };

  const validRows = rows.filter((r) => r.path.trim() && r.label.trim());
  const trimmedDestination = destinationPath.trim();
  // Filename: strip any path separators the user might have typed and
  // force a `.md` extension so the saved file is consistent.
  const sanitizedFilename = (() => {
    let n = reportFilename.trim().replace(/[\\/]+/g, "_");
    if (!n) return "";
    if (!/\.md$/i.test(n)) n += ".md";
    return n;
  })();
  const canSubmit =
    validRows.length >= 2 && !!trimmedDestination && !!sanitizedFilename;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      entries: validRows.map((r) => ({
        label: r.label.trim(),
        path: r.path.trim(),
      })),
      destinationPath: trimmedDestination,
      reportFilename: sanitizedFilename,
    });
  };

  const handlePickDestination = async () => {
    setDestBrowsing(true);
    try {
      const picked = await pickFolder();
      if (picked) {
        setDestinationPath(picked);
        setDestinationAuto(false);
      }
    } finally {
      setDestBrowsing(false);
    }
  };

  const handleAddDestFromWorkspace = (workspaceId: string) => {
    const ws = workspaceOptions.find((w) => w.id === workspaceId);
    if (!ws) return;
    setDestinationPath(ws.rootPath);
    setDestinationAuto(false);
    setDestWorkspaceId("");
  };

  const handleOpenRecent = (report: RecentReport) => {
    onOpenReport?.(report);
  };

  const handleRemoveRecent = (id: string) => {
    removeRecentReport(id);
    setRecentReports(listRecentReports());
  };

  return (
    <form className="evaluate-questions-form" onSubmit={submit} autoComplete="off">
      <p>
        Pick two or more folders containing autoq question JSON files
        (e.g. <code>output/.../questions/activity_global_questions/</code>).
        Folders can live in any workspace or anywhere on disk.
      </p>

      {recentReports.length > 0 && (
        <details className="eval-recent-reports" open>
          <summary>
            Recent reports ({recentReports.length})
          </summary>
          <ul className="eval-recent-reports-list">
            {recentReports.map((r) => (
              <li key={r.id}>
                <button
                  type="button"
                  className="eval-recent-report-open"
                  title={`Open ${r.reportPath}`}
                  onClick={() => handleOpenRecent(r)}
                  disabled={!onOpenReport}
                >
                  <DocumentArrowLeft16Regular />
                  <span className="eval-recent-report-label">{r.label}</span>
                  <span className="eval-recent-report-meta">
                    {new Date(r.savedAt).toLocaleString()}
                    {r.setLabels.length > 0
                      ? ` · ${r.setLabels.join(" vs ")}`
                      : ""}
                  </span>
                </button>
                <button
                  type="button"
                  className="eval-recent-report-remove"
                  title="Remove from history"
                  aria-label="Remove from history"
                  onClick={() => handleRemoveRecent(r.id)}
                >
                  <Dismiss16Regular />
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}

      {workspaceOptions.length > 0 && (
        <div className="eval-quick-add">
          <span className="eval-quick-add-label">Add from workspace:</span>
          <select
            value={pendingWorkspaceId}
            onChange={(e) => setPendingWorkspaceId(e.target.value)}
          >
            <option value="" disabled>
              Choose a workspace…
            </option>
            {workspaceOptions.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn"
            disabled={!pendingWorkspaceId}
            onClick={() => {
              if (!pendingWorkspaceId) return;
              addFromWorkspace(pendingWorkspaceId);
              setPendingWorkspaceId("");
            }}
          >
            Add
          </button>
        </div>
      )}

      <div className="eval-rows">
        {rows.map((row, idx) => (
          <div key={row.id} className="eval-row">
            <div className="eval-row-header">
              <span className="eval-row-index">#{idx + 1}</span>
              <button
                type="button"
                className="eval-row-remove"
                onClick={() => removeRow(row.id)}
                title="Remove this folder"
                aria-label="Remove folder"
              >
                <Dismiss16Regular />
              </button>
            </div>
            <label className="field eval-row-field">
              <span>Label</span>
              <input
                value={row.label}
                onChange={(e) => handleLabelChange(row.id, e.target.value)}
                placeholder="Set A"
                spellCheck={false}
              />
            </label>
            <label className="field eval-row-field">
              <span>Folder path</span>
              <div style={{ display: "flex", gap: 6 }}>
                <input
                  value={row.path}
                  onChange={(e) => updatePath(row.id, e.target.value)}
                  placeholder="/absolute/path/to/questions"
                  spellCheck={false}
                  style={{ flex: 1 }}
                />
                <button
                  type="button"
                  onClick={() => handlePick(row.id)}
                  disabled={activeBrowseId === row.id}
                  title="Browse for a folder"
                >
                  <Folder16Regular /> Browse…
                </button>
              </div>
            </label>
          </div>
        ))}
      </div>

      <button
        type="button"
        className="btn secondary eval-add-row"
        onClick={addRow}
      >
        <Add16Regular /> Add another folder
      </button>

      <fieldset className="eval-destination">
        <legend>Save report to</legend>
        <p className="eval-destination-hint">
          Choose the destination folder and file name for the generated
          markdown report. Defaults to the first compared folder.
        </p>
        <label className="field eval-row-field">
          <span>File name</span>
          <input
            value={reportFilename}
            onChange={(e) => setReportFilename(e.target.value)}
            placeholder="quality_report.md"
            spellCheck={false}
          />
        </label>
        <div className="eval-destination-row">
          <input
            value={destinationPath}
            onChange={(e) => {
              setDestinationPath(e.target.value);
              setDestinationAuto(false);
            }}
            placeholder="/absolute/path/to/report-folder"
            spellCheck={false}
            style={{ flex: 1 }}
          />
          <button
            type="button"
            onClick={handlePickDestination}
            disabled={destBrowsing}
            title="Browse for a folder"
          >
            <Folder16Regular /> Browse…
          </button>
        </div>
        {workspaceOptions.length > 0 && (
          <div className="eval-destination-row">
            <span className="eval-quick-add-label">Or pick a workspace:</span>
            <select
              value={destWorkspaceId}
              onChange={(e) => setDestWorkspaceId(e.target.value)}
            >
              <option value="" disabled>
                Choose a workspace…
              </option>
              {workspaceOptions.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="btn"
              disabled={!destWorkspaceId}
              onClick={() => handleAddDestFromWorkspace(destWorkspaceId)}
            >
              Use
            </button>
          </div>
        )}
      </fieldset>

      <div className="modal-actions">
        <button type="button" className="btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!canSubmit}
          title={
            canSubmit
              ? undefined
              : "Provide at least two rows with a label and folder path, plus a destination folder and a file name for the report"
          }
        >
          Start evaluation
        </button>
      </div>
    </form>
  );
}

/**
 * Dialog for picking N folders (from any workspace or anywhere on disk) to
 * pass to the benchmark-qed-question-quality skill. Replaces the agent's
 * "how many sets?" question with a direct multi-folder picker.
 */
export function EvaluateQuestionsPickerDialog({
  open,
  workspaces,
  pickFolder,
  onClose,
  onSubmit,
}: Props) {
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal evaluate-questions-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h3>Evaluate Question Quality</h3>
          <button className="modal-close" onClick={onClose}>
            <Dismiss16Regular />
          </button>
        </div>
        <div className="modal-body">
          <EvaluateQuestionsPickerForm
            workspaces={workspaces}
            pickFolder={pickFolder}
            onCancel={onClose}
            onSubmit={onSubmit}
            resetKey={open ? "open" : "closed"}
          />
        </div>
      </div>
    </div>
  );
}
