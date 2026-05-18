interface Props {
  open: boolean;
  onAllow: () => void;
  onDeny: () => void;
}

export function FileAccessPermissionDialog({ open, onAllow, onDeny }: Props) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onDeny}>
      <div className="modal file-access-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Open Workspace Folder</h3>
        </div>
        <div className="modal-body">
          <p>
            Choose a folder on your machine to open it as a workspace.
          </p>
          <div className="modal-actions">
            <button type="button" className="btn" onClick={onDeny}>
              Cancel
            </button>
            <button type="button" className="btn btn-primary" onClick={onAllow}>
              Pick Folder
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
