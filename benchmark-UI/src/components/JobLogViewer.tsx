import { ClipboardTask16Regular } from "@fluentui/react-icons";
import { TerminalLog } from "./TerminalLog";
import { CancelJobButton } from "./CancelJobButton";
import type { RunJob } from "./RunJobsBottomTab";

interface Props {
  job: RunJob;
  onCancel: (jobId: string) => void;
  onRerun?: (jobId: string) => void;
}

export function JobLogViewer({ job, onCancel, onRerun }: Props) {
  const canRerun =
    !!onRerun &&
    !!job.rootPath &&
    !!job.configType &&
    job.status !== "running";

  // Trigger a browser download of the current in-memory log as a .log file.
  // The runner no longer persists logs to disk, so this is the user's only
  // way to keep a copy of a finished run's output.
  const downloadLog = () => {
    const content = job.output || "(no output)";
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const stamp = job.startedAt.replace(/[:.]/g, "-");
    const base = job.configType || "job";
    const a = document.createElement("a");
    a.href = url;
    a.download = `${base}-${stamp}.log`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="job-log-viewer">
      <div className="file-header">
        <span className="file-path">
          <span className="job-label"><ClipboardTask16Regular /> Job</span>
          <span className="job-status-badge" data-status={job.status}>
            {job.status}
          </span>
          {new Date(job.startedAt).toLocaleTimeString()}
        </span>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            className="init-jobs-clear"
            title="Download the log as a .log file"
            onClick={downloadLog}
            disabled={!job.output}
          >
            Download log
          </button>
          {job.status === "running" ? (
            <CancelJobButton jobId={job.id} onCancel={onCancel} />
          ) : canRerun ? (
            <button
              type="button"
              className="init-jobs-clear"
              title="Re-run this job"
              onClick={() => onRerun!(job.id)}
            >
              Re-run
            </button>
          ) : null}
        </div>
      </div>
      <div className="job-log-content">
        <div className="jobs-log-command">{job.command}</div>
        <TerminalLog text={job.output || "(no output)"} />
      </div>
    </div>
  );
}
