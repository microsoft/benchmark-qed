import { ClipboardTask16Regular } from "@fluentui/react-icons";
import { TerminalLog } from "./TerminalLog";
import { CancelJobButton } from "./CancelJobButton";
import type { RunJob } from "./RunJobsBottomTab";

interface Props {
  job: RunJob;
  onCancel: (jobId: string) => void;
}

export function JobLogViewer({ job, onCancel }: Props) {
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
        {job.status === "running" && (
          <CancelJobButton jobId={job.id} onCancel={onCancel} />
        )}
      </div>
      <div className="job-log-content">
        <div className="jobs-log-command">{job.command}</div>
        <TerminalLog text={job.output || "(no output)"} />
      </div>
    </div>
  );
}
