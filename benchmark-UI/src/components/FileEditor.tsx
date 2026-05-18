import type { OpenFile } from "../types";
import { CodeEditor } from "./CodeEditor";
import { CsvViewer } from "./CsvViewer";
import { MarkdownEditor } from "./MarkdownEditor";

interface Props {
  file: OpenFile | null;
  workspaceName?: string;
  readOnly?: boolean;
  onContentChange: (value: string) => void;
  onSave: () => void;
  saving: boolean;
  theme?: "dark" | "light";
}

export function FileEditor({
  file,
  workspaceName,
  readOnly,
  onContentChange,
  onSave,
  saving,
  theme = "dark",
}: Props) {
  if (!file) {
    return (
      <div className="empty-state">
        <h2>No file open</h2>
        <p>Open a folder on the left and select a file to view or edit it.</p>
      </div>
    );
  }

  const isCsv = file.kind === "csv";
  const delimiter = file.node.name.toLowerCase().endsWith(".tsv") ? "\t" : ",";

  return (
    <div className="file-editor">
      <div className="file-header">
        <span className="file-path">
          {workspaceName && (
            <span className="ws-prefix">{workspaceName} / </span>
          )}
          {file.node.path}
          {file.dirty && <span className="dirty"> •</span>}
          <span className="lang-badge">{file.language}</span>
          {readOnly && <span className="ro-badge">READ-ONLY</span>}
        </span>
        {!isCsv && !readOnly && (
          <button
            onClick={onSave}
            disabled={!file.dirty || saving}
            className="save-btn"
          >
            {saving ? "Saving…" : "Save"}
          </button>
        )}
      </div>
      <div className="file-body">
        {file.kind === "markdown" && (
          <MarkdownEditor
            content={file.content}
            readOnly={readOnly}
            onChange={onContentChange}
          />
        )}
        {file.kind === "csv" && (
          <CsvViewer content={file.content} delimiter={delimiter} />
        )}
        {file.kind === "code" && (
          <CodeEditor
            value={file.content}
            language={file.language}
            readOnly={readOnly}
            onChange={onContentChange}
            theme={theme}
          />
        )}
        {file.kind === "unsupported" && (
          <div className="empty-state">
            <p>Preview not available for this file type.</p>
          </div>
        )}
      </div>
    </div>
  );
}
