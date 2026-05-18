interface CancelJobButtonProps {
  jobId: string;
  disabled?: boolean;
  onCancel: (jobId: string) => void;
}

export function CancelJobButton({ jobId, disabled, onCancel }: CancelJobButtonProps) {
  return (
    <button
      type="button"
      className="init-jobs-cancel"
      disabled={disabled}
      title="Cancel job"
      onClick={() => onCancel(jobId)}
    >
      Cancel
    </button>
  );
}
