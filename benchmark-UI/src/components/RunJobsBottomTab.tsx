import { TerminalLog } from "./TerminalLog";
import { CancelJobButton } from "./CancelJobButton";

export interface RunJob {
  id: string;
  status: "running" | "succeeded" | "failed" | "cancelled";
  startedAt: string;
  endedAt: string | null;
  rootPath?: string;
  configType?: "autoq" | "autoe_pairwise" | "autoe_reference" | "autoe_assertion";
  command: string;
  output: string;
  exitCode: number | null;
}

interface Props {
  jobs: RunJob[];
  selectedJobId: string | null;
  onSelectJob: (id: string) => void;
  clearing?: boolean;
  onClear?: () => void;
  onCancel: (id: string) => void;
  onClose: () => void;
}

export function RunJobsBottomTab({
  jobs,
  selectedJobId,
  onSelectJob,
  clearing,
  onClear,
  onCancel,
  onClose,
}: Props) {
  const selected = jobs.find((j) => j.id === selectedJobId) ?? jobs[0];

  return (
    <section className="jobs-dock" aria-label="Run jobs logs">
      <div className="jobs-dock-header">
        <div className="jobs-dock-title">Run Jobs</div>
        <div className="jobs-dock-actions">
          <button
            type="button"
            className="init-jobs-clear"
            onClick={onClear}
            disabled={clearing || jobs.length === 0}
          >
            {clearing ? "Clearing..." : "Clear"}
          </button>
          <button type="button" className="init-jobs-clear" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <div className="jobs-dock-body">
        <div className="jobs-list">
          {jobs.length === 0 ? (
            <div className="jobs-empty">No run jobs yet.</div>
          ) : (
            jobs.map((job) => {
              const folderName = job.rootPath
                ? job.rootPath.split(/[/\\]/).filter(Boolean).pop()
                : undefined;
              const configLabel = job.configType ?? "job";
              const label = folderName ? `${folderName} · ${configLabel}` : configLabel;
              return (
                <div key={job.id} className={`jobs-item ${selected?.id === job.id ? "active" : ""}`}>
                  <button
                    type="button"
                    className="jobs-item-select"
                    onClick={() => onSelectJob(job.id)}
                    title={`${label}\n${job.command}`}
                  >
                    <span className="jobs-item-name">{label}</span>
                    <span className="jobs-item-meta">
                      <span className={`init-status init-status-${job.status}`}>
                        {job.status}
                      </span>
                      <span className="jobs-item-time">
                        {new Date(job.startedAt).toLocaleTimeString()}
                      </span>
                    </span>
                  </button>
                  {job.status === "running" && (
                    <CancelJobButton jobId={job.id} onCancel={onCancel} />
                  )}
                </div>
              );
            })
          )}
        </div>
        <div className="jobs-log" role="log">
          {selected ? (
            <>
              <div className="jobs-log-command">{selected.command}</div>
              <TerminalLog text={selected.output || "(no output)"} />
            </>
          ) : (
            "No run job selected."
          )}
        </div>
      </div>
    </section>
  );
}
