import Editor, { type OnMount } from "@monaco-editor/react";
import { useRef } from "react";

interface Props {
  value: string;
  language: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  theme?: "dark" | "light";
}

export function CodeEditor({ value, language, onChange, readOnly, theme = "dark" }: Props) {
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  const handleMount: OnMount = (editor) => {
    editorRef.current = editor;
  };

  return (
    <Editor
      value={value}
      language={language}
      onChange={(v) => onChange(v ?? "")}
      onMount={handleMount}
      theme={theme === "dark" ? "vs-dark" : "vs-light"}
      options={{
        readOnly,
        automaticLayout: true,
        fontSize: 13,
        fontFamily:
          '"SF Mono", Menlo, Monaco, Consolas, "Courier New", monospace',
        minimap: { enabled: true },
        scrollBeyondLastLine: false,
        wordWrap: "on",
        folding: true,
        foldingStrategy: "indentation",
        showFoldingControls: "always",
        bracketPairColorization: { enabled: true },
        renderWhitespace: "selection",
        tabSize: 2,
      }}
    />
  );
}
