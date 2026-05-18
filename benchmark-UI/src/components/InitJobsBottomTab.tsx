import type { InitJob } from "./InitJobsPanel";

interface Props {
  jobs: InitJob[];
  selectedJobId: string | null;
  onSelectJob: (id: string) => void;
  clearing?: boolean;
  onClear?: () => void;
  onClose: () => void;
}

export function InitJobsBottomTab({
  jobs,
  selectedJobId,
  onSelectJob,
  clearing,
  onClear,
  onClose,
}: Props) {
  const selected = jobs.find((j) => j.id === selectedJobId) ?? jobs[0];

  return (
    <section className="jobs-dock" aria-label="Init jobs logs">
      <div className="jobs-dock-header">
        <div className="jobs-dock-title">Init Jobs</div>
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
            <div className="jobs-empty">No init jobs yet.</div>
          ) : (
            jobs.map((job) => (
              <button
                key={job.id}
                type="button"
                className={`jobs-item ${selected?.id === job.id ? "active" : ""}`}
                onClick={() => onSelectJob(job.id)}
                title={job.command}
              >
                <span className={`init-status init-status-${job.status}`}>
                  {job.status}
                </span>
                <span className="jobs-item-time">
                  {new Date(job.startedAt).toLocaleTimeString()}
                </span>
              </button>
            ))
          )}
        </div>
        <pre className="jobs-log" role="log">
          {selected
            ? `${selected.command}\n\n${selected.output || "(no output)"}`
            : "No init job selected."}
        </pre>
      </div>
    </section>
  );
}
