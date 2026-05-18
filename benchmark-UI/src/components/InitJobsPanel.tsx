export interface InitJob {
  id: string;
  status: "running" | "succeeded" | "failed";
  startedAt: string;
  endedAt: string | null;
  rootPath?: string;
  configType?: "autoq" | "autoe_pairwise" | "autoe_reference" | "autoe_assertion";
  command: string;
  output: string;
  exitCode: number | null;
}

interface Props {
  jobs: InitJob[];
  clearing?: boolean;
  onClear?: () => void;
}

export function InitJobsPanel({ jobs, clearing, onClear }: Props) {
  if (!jobs.length) return null;

  return (
    <div className="init-jobs">
      <div className="init-jobs-header">
        <div className="init-jobs-title">Init jobs</div>
        <button
          type="button"
          className="init-jobs-clear"
          onClick={onClear}
          disabled={clearing}
        >
          {clearing ? "Clearing..." : "Clear"}
        </button>
      </div>
      {jobs.slice(0, 4).map((job) => (
        <div key={job.id} className="init-job-item" title={job.command}>
          <span className={`init-status init-status-${job.status}`}>
            {job.status}
          </span>
          <span className="init-time">
            {new Date(job.startedAt).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}
