import { useEffect, useState } from "react";
import { Folder16Regular, Document16Regular } from "@fluentui/react-icons";
import type { AskUserAnswer, AskUserPayload } from "../copilot/types";
import {
  pickFiles,
  pickFolder,
  looksLikeFolderPrompt,
  looksLikeFilesPrompt,
} from "../copilot/pickers";

interface Props {
  payload: AskUserPayload;
  submitting: boolean;
  onSubmit: (answer: AskUserAnswer) => void;
  onCancel: () => void;
}

export function CopilotAskUserDialog({
  payload,
  submitting,
  onSubmit,
  onCancel,
}: Props) {
  const [choice, setChoice] = useState<string | null>(
    payload.choices?.[0] ?? null,
  );
  const [freeform, setFreeform] = useState("");
  const [pickerError, setPickerError] = useState<string | null>(null);

  // Reset state whenever a new request arrives (component is keyed by requestId).
  useEffect(() => {
    setChoice(payload.choices?.[0] ?? null);
    setFreeform("");
    setPickerError(null);
  }, [payload]);

  const usingChoice =
    payload.choices && payload.choices.length > 0 && !freeform.trim();

  const canSubmit = usingChoice ? !!choice : freeform.trim().length > 0;

  const showFolderHelper =
    payload.allowFreeform && looksLikeFolderPrompt(payload.question);
  const showFilesHelper =
    payload.allowFreeform && looksLikeFilesPrompt(payload.question);

  const doPickFolder = async () => {
    setPickerError(null);
    try {
      const p = await pickFolder();
      if (p) setFreeform(p);
    } catch (e) {
      setPickerError(e instanceof Error ? e.message : String(e));
    }
  };

  const doPickFiles = async () => {
    setPickerError(null);
    try {
      const paths = await pickFiles();
      if (paths && paths.length > 0) {
        // Newline-separated so the agent can split them; falls back gracefully
        // for single-file answers.
        setFreeform(paths.join("\n"));
      }
    } catch (e) {
      setPickerError(e instanceof Error ? e.message : String(e));
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    if (usingChoice) {
      onSubmit({ answer: choice!, wasFreeform: false });
    } else {
      onSubmit({ answer: freeform.trim(), wasFreeform: true });
    }
  };

  return (
    <div className="copilot-inline-card">
      <div className="copilot-inline-card-header">
        <strong>Copilot asks:</strong>
      </div>
      <form className="copilot-inline-card-body" onSubmit={submit}>
        <p style={{ margin: 0 }}>{payload.question}</p>

        {payload.choices && payload.choices.length > 0 && (
          <div className="copilot-choice-row">
            {payload.choices.map((opt) => (
              <button
                key={opt}
                type="button"
                className={`copilot-choice ${choice === opt ? "active" : ""}`}
                onClick={() => {
                  setChoice(opt);
                  setFreeform("");
                }}
              >
                {opt}
              </button>
            ))}
          </div>
        )}

        {payload.allowFreeform && (
          <div>
            <textarea
              rows={2}
              value={freeform}
              onChange={(e) => setFreeform(e.target.value)}
              placeholder={
                payload.choices?.length
                  ? "Or type a different answer..."
                  : "Type your answer..."
              }
              style={{ width: "100%", fontFamily: "inherit" }}
            />
            {(showFolderHelper || showFilesHelper) && (
              <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                {showFolderHelper && (
                  <button type="button" className="btn" onClick={doPickFolder}>
                    <Folder16Regular
                      style={{ verticalAlign: "-3px", marginRight: 4 }}
                    />
                    Pick folder
                  </button>
                )}
                {showFilesHelper && (
                  <button type="button" className="btn" onClick={doPickFiles}>
                    <Document16Regular
                      style={{ verticalAlign: "-3px", marginRight: 4 }}
                    />
                    Pick files
                  </button>
                )}
              </div>
            )}
            {pickerError && (
              <small style={{ color: "#d4453e" }}>{pickerError}</small>
            )}
          </div>
        )}

        <div className="copilot-inline-actions">
          <button type="button" className="btn" onClick={onCancel}>
            Skip
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={submitting || !canSubmit}
          >
            {submitting ? "Sending..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}
