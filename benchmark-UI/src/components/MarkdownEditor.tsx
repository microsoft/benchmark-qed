import { useMemo, useState } from "react";
import { marked } from "marked";
import { CodeEditor } from "./CodeEditor";

interface Props {
  content: string;
  readOnly?: boolean;
  onChange: (value: string) => void;
}

export function MarkdownEditor({ content, readOnly, onChange }: Props) {
  const [mode, setMode] = useState<"edit" | "preview" | "split">("split");

  const html = useMemo(() => {
    try {
      return marked.parse(content, { async: false }) as string;
    } catch {
      return "";
    }
  }, [content]);

  return (
    <div className="md-editor">
      <div className="editor-toolbar">
        <button
          className={mode === "edit" ? "active" : ""}
          onClick={() => setMode("edit")}
        >
          Edit
        </button>
        <button
          className={mode === "split" ? "active" : ""}
          onClick={() => setMode("split")}
        >
          Split
        </button>
        <button
          className={mode === "preview" ? "active" : ""}
          onClick={() => setMode("preview")}
        >
          Preview
        </button>
      </div>
      <div className={`md-body ${mode}`}>
        {(mode === "edit" || mode === "split") && (
          <div className="md-edit-pane">
            <CodeEditor
              value={content}
              language="markdown"
              readOnly={readOnly}
              onChange={onChange}
            />
          </div>
        )}
        {(mode === "preview" || mode === "split") && (
          <div
            className="md-preview"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}
      </div>
    </div>
  );
}
