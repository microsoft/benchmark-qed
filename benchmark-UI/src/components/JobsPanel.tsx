import { useState } from "react";
import {
  ChevronRight16Regular,
  ChevronDown16Regular,
  ClipboardTask16Regular,
  Play16Regular,
  ArrowClockwise16Regular,
  ArrowDownload16Regular,
  Delete16Regular,
} from "@fluentui/react-icons";
import type { InitJob } from "./InitJobsPanel";
import type { RunJob } from "./RunJobsBottomTab";
import { CancelJobButton } from "./CancelJobButton";

type JobStatus = InitJob["status"] | RunJob["status"];

interface Props {
  initJobs: InitJob[];
  runJobs: RunJob[];
  selectedJob: { kind: "init" | "run"; id: string } | null;
  onOpenInitJob: (id: string) => void;
  onOpenRunJob: (id: string) => void;
  onCancelRunJob: (id: string) => void;
  onRerunJob?: (id: string) => void;
  onDeleteJob: (kind: "init" | "run", id: string) => void;
  /** Map of workspace `rootPath` → display name, so jobs can show the
   *  workspace they belong to instead of just the folder basename. */
  workspaceNameByRoot?: Record<string, string>;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

function statusOrder(s: JobStatus): number {
  switch (s) {
    case "running":
      return 0;
    case "failed":
      return 1;
    case "succeeded":
      return 2;
    default:
      return 3;
  }
}

function jobFolder(job: InitJob | RunJob): string | undefined {
  if (!job.rootPath) return undefined;
  // For blob init jobs `rootPath` is typically "." (the runner's cwd) and
  // isn't a meaningful display name. Hide it.
  const isBlobJob =
    (job as InitJob).storageType === "blob" ||
    /--storage-type\s+blob\b/.test(job.command ?? "");
  if (isBlobJob && (job.rootPath === "." || job.rootPath.trim() === ""))
    return undefined;
  return job.rootPath.split(/[/\\]/).filter(Boolean).pop();
}

function normalizePath(p: string): string {
  // Strip trailing slashes/backslashes and lowercase for case-insensitive
  // filesystems (macOS default, Windows).
  return p.replace(/[/\\]+$/, "").toLowerCase();
}

function lookupWorkspaceName(
  rootPath: string,
  map?: Record<string, string>,
): string | undefined {
  if (!map) return undefined;
  if (map[rootPath]) return map[rootPath];
  const target = normalizePath(rootPath);
  // 1. Exact match (after normalization).
  for (const [k, v] of Object.entries(map)) {
    if (normalizePath(k) === target) return v;
  }
  // 2. Workspace contains the job (job path starts with workspace path),
  //    or job contains the workspace (workspace path starts with job path).
  //    Pick the LONGEST such match so the nearest workspace wins.
  let best: { len: number; name: string } | undefined;
  for (const [k, v] of Object.entries(map)) {
    const w = normalizePath(k);
    if (!w) continue;
    const fits =
      target === w ||
      target.startsWith(w + "/") ||
      w.startsWith(target + "/");
    if (fits && (!best || w.length > best.len)) {
      best = { len: w.length, name: v };
    }
  }
  return best?.name;
}

function jobLabel(job: InitJob | RunJob): string {
  const folder = jobFolder(job);
  const config = job.configType ?? "job";
  return folder ? `${folder} · ${config}` : config;
}
function downloadJobLog(job: InitJob | RunJob, kind: "init" | "run"): void {
  const folder = job.rootPath
    ? job.rootPath.split(/[/\\]/).filter(Boolean).pop() ?? "job"
    : "job";
  const config = job.configType ?? kind;
  const stamp = new Date(job.startedAt)
    .toISOString()
    .replace(/[:.]/g, "-")
    .replace(/Z$/, "");
  const filename = `${folder}-${config}-${stamp}.log`;
  const ended = (job as { endedAt?: string }).endedAt;
  const exit = (job as { exitCode?: number | null }).exitCode;
  const header = [
    `# benchmark-qed ${kind} job log`,
    `id:         ${job.id}`,
    `command:    ${job.command ?? ""}`,
    `rootPath:   ${job.rootPath ?? ""}`,
    `configType: ${job.configType ?? ""}`,
    `status:     ${job.status}`,
    `startedAt:  ${job.startedAt}`,
    `endedAt:    ${ended ?? ""}`,
    `exitCode:   ${exit ?? ""}`,
    "",
    "----- output -----",
    "",
  ].join("\n");
  const body = (job as { output?: string }).output ?? "";
  const blob = new Blob([header + body], {
    type: "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Defer revoking to give the browser time to start the download.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

interface SectionProps {
  title: string;
  jobs: (InitJob | RunJob)[];
  kind: "init" | "run";
  selectedId: string | null;
  onOpen: (id: string) => void;
  onCancel?: (id: string) => void;
  onRerun?: (id: string) => void;
  onDelete: (id: string) => void;
  workspaceNameByRoot?: Record<string, string>;
}

function JobSection({
  title,
  jobs,
  kind,
  selectedId,
  onOpen,
  onCancel,
  onRerun,
  onDelete,
  workspaceNameByRoot,
}: SectionProps) {
  const [open, setOpen] = useState(true);
  const sorted = [...jobs].sort((a, b) => {
    const so = statusOrder(a.status) - statusOrder(b.status);
    if (so !== 0) return so;
    return Date.parse(b.startedAt) - Date.parse(a.startedAt);
  });
  const running = jobs.filter((j) => j.status === "running").length;

  return (
    <div className="jobs-section">
      <button
        type="button"
        className="jobs-section-header"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? <ChevronDown16Regular /> : <ChevronRight16Regular />}
        <span className="jobs-section-title">{title}</span>
        <span className="jobs-section-count">
          {jobs.length}
          {running > 0 ? ` · ${running} running` : ""}
        </span>
      </button>
      {open && (
        <div className="jobs-section-body">
          {jobs.length === 0 ? (
            <div className="jobs-section-empty">No {title.toLowerCase()}.</div>
          ) : (
            sorted.map((job) => {
              const label = jobLabel(job);
              const wsBadge = job.rootPath
                ? lookupWorkspaceName(job.rootPath, workspaceNameByRoot)
                : undefined;
              const active = selectedId === job.id;
              const canRerun =
                kind === "run" &&
                !!onRerun &&
                !!job.rootPath &&
                !!job.configType &&
                job.status !== "running";
              return (
                <div
                  key={job.id}
                  className={`jobs-panel-item${active ? " active" : ""}`}
                >
                  <button
                    type="button"
                    className="jobs-panel-item-main"
                    onClick={() => onOpen(job.id)}
                    title={`${wsBadge ? `Workspace: ${wsBadge}\n` : ""}${label}\n${job.command}`}
                  >
                    <ClipboardTask16Regular className="jobs-panel-item-icon" />
                    <span className="jobs-panel-item-text">
                      <span className="jobs-panel-item-label">
                        {wsBadge && (
                          <span
                            className="jobs-panel-item-ws"
                            title={`Workspace: ${wsBadge}`}
                          >
                            {wsBadge}
                          </span>
                        )}
                        <span className="jobs-panel-item-label-text">
                          {label}
                        </span>
                      </span>
                      <span className="jobs-panel-item-meta">
                        <span
                          className={`init-status init-status-${job.status}`}
                        >
                          {job.status}
                        </span>
                        <span className="jobs-panel-item-time">
                          {new Date(job.startedAt).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      </span>
                    </span>
                  </button>
                  <div className="jobs-panel-item-actions">
                    {job.status === "running" && onCancel ? (
                      <CancelJobButton jobId={job.id} onCancel={onCancel} />
                    ) : canRerun ? (
                      <button
                        type="button"
                        className="jobs-panel-item-action"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRerun!(job.id);
                        }}
                        title="Re-run this job"
                        aria-label="Re-run job"
                      >
                        <ArrowClockwise16Regular />
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="jobs-panel-item-action"
                      onClick={(e) => {
                        e.stopPropagation();
                        downloadJobLog(job, kind);
                      }}
                      title="Download job log"
                      aria-label="Download job log"
                    >
                      <ArrowDownload16Regular />
                    </button>
                    <button
                      type="button"
                      className="jobs-panel-item-action jobs-panel-item-delete"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (job.status === "running") return;
                        onDelete(job.id);
                      }}
                      disabled={job.status === "running"}
                      title={
                        job.status === "running"
                          ? "Cancel the job before removing it"
                          : "Remove job from list"
                      }
                      aria-label="Remove job"
                    >
                      <Delete16Regular />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

export function JobsPanel({
  initJobs,
  runJobs,
  selectedJob,
  onOpenInitJob,
  onOpenRunJob,
  onCancelRunJob,
  onRerunJob,
  onDeleteJob,
  workspaceNameByRoot,
  collapsed = false,
  onToggleCollapsed,
}: Props) {
  const totalRunning =
    initJobs.filter((j) => j.status === "running").length +
    runJobs.filter((j) => j.status === "running").length;

  return (
    <aside className={`jobs-panel ${collapsed ? "collapsed" : ""}`}>
      <div className="jobs-panel-header">
        <div className="jobs-panel-header-content">
          <Play16Regular />
          <h3>Jobs</h3>
          {totalRunning > 0 && (
            <span className="jobs-panel-running-badge" title="Running jobs">
              {totalRunning}
            </span>
          )}
          {onToggleCollapsed && (
            <button
              className="jobs-panel-toggle"
              onClick={onToggleCollapsed}
              title={collapsed ? "Expand jobs" : "Collapse jobs"}
            >
              {collapsed ? (
                <ChevronRight16Regular />
              ) : (
                <ChevronDown16Regular />
              )}
            </button>
          )}
        </div>
      </div>
      {!collapsed && (
        <div className="jobs-panel-body">
          <JobSection
            title="Run jobs"
            jobs={runJobs}
            kind="run"
            selectedId={selectedJob?.kind === "run" ? selectedJob.id : null}
            onOpen={onOpenRunJob}
            onCancel={onCancelRunJob}
            onRerun={onRerunJob}
            onDelete={(id) => onDeleteJob("run", id)}
            workspaceNameByRoot={workspaceNameByRoot}
          />
          <JobSection
            title="Init jobs"
            jobs={initJobs}
            kind="init"
            selectedId={selectedJob?.kind === "init" ? selectedJob.id : null}
            onOpen={onOpenInitJob}
            onDelete={(id) => onDeleteJob("init", id)}
            workspaceNameByRoot={workspaceNameByRoot}
          />
        </div>
      )}
    </aside>
  );
}
