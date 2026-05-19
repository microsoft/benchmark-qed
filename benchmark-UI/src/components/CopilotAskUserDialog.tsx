import { useEffect, useState } from "react";
import { Folder16Regular, Document16Regular } from "@fluentui/react-icons";
import type { AskUserAnswer, AskUserPayload } from "../copilot/types";
import {
  pickFiles,
  pickFolder,
  looksLikeFolderPrompt,
  looksLikeFilesPrompt,
  choicesLookLikePaths,
  choiceImpliesOwnData,
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
  // When choices OR a folder/files picker are available, the question is
  // effectively closed — keep the freeform input hidden by default. The user
  // can opt in via the "type a custom answer" link.
  const hasChoices = !!payload.choices && payload.choices.length > 0;
  const choicesArePaths = choicesLookLikePaths(payload.choices);
  const browseAllowed = !hasChoices || choicesArePaths;
  const showFolderHelper =
    browseAllowed &&
    payload.allowFreeform &&
    looksLikeFolderPrompt(payload.question);
  const showFilesHelper =
    browseAllowed &&
    payload.allowFreeform &&
    looksLikeFilesPrompt(payload.question);
  const hasPicker = showFolderHelper || showFilesHelper;
  const [freeformVisible, setFreeformVisible] = useState(
    !hasChoices && !hasPicker,
  );

  // Reset state whenever a new request arrives (component is keyed by requestId).
  useEffect(() => {
    setChoice(payload.choices?.[0] ?? null);
    setFreeform("");
    setPickerError(null);
    setFreeformVisible(!hasChoices && !hasPicker);
  }, [payload, hasChoices, hasPicker]);

  const usingChoice =
    payload.choices && payload.choices.length > 0 && !freeform.trim();

  const canSubmit = usingChoice ? !!choice : freeform.trim().length > 0;

  const doPickFolder = async () => {
    setPickerError(null);
    try {
      const p = await pickFolder();
      if (p) {
        setFreeform(p);
        // The picked path is the answer — submit immediately so the user
        // doesn't have to confirm again.
        onSubmit({ answer: p, wasFreeform: true });
      }
    } catch (e) {
      setPickerError(e instanceof Error ? e.message : String(e));
    }
  };

  const doPickFiles = async () => {
    setPickerError(null);
    try {
      const paths = await pickFiles();
      if (paths && paths.length > 0) {
        const joined = paths.join("\n");
        setFreeform(joined);
        onSubmit({ answer: joined, wasFreeform: true });
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
        <p style={{ margin: 0, fontSize: 14 }}>{payload.question}</p>

        {(showFolderHelper || showFilesHelper) && (
          <div className="copilot-section">
            <span className="copilot-section-label">Browse</span>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {showFolderHelper && (
                <button
                  type="button"
                  className="btn"
                  onClick={doPickFolder}
                  disabled={submitting}
                >
                  <Folder16Regular
                    style={{ verticalAlign: "-3px", marginRight: 4 }}
                  />
                  Pick folder…
                </button>
              )}
              {showFilesHelper && (
                <button
                  type="button"
                  className="btn"
                  onClick={doPickFiles}
                  disabled={submitting}
                >
                  <Document16Regular
                    style={{ verticalAlign: "-3px", marginRight: 4 }}
                  />
                  Pick files…
                </button>
              )}
            </div>
          </div>
        )}
        {pickerError && (
          <small style={{ color: "#d4453e" }}>{pickerError}</small>
        )}

        {hasChoices && (
          <div className="copilot-section">
            <span className="copilot-section-label">
              {showFolderHelper || showFilesHelper ? "Or pick a preset" : "Choose one"}
            </span>
            <div className="copilot-option-grid">
              {payload.choices!.map((opt) => {
                // Split "Label — Description" so we can render the same
                // two-line card style used by the Create Configuration dialog.
                const m = opt.match(/^(.+?)\s+[—–-]\s+(.+)$/);
                const label = m ? m[1] : opt;
                const desc = m ? m[2] : null;
                // "Bring my own data"-style options open a folder picker on
                // click so the user can immediately point at their dataset.
                const ownData = choiceImpliesOwnData(opt);
                return (
                  <button
                    key={opt}
                    type="button"
                    className={`option-card ${choice === opt ? "active" : ""}`}
                    onClick={() => {
                      setChoice(opt);
                      setFreeform("");
                      if (ownData) void doPickFolder();
                    }}
                  >
                    <span className="option-card-label">
                      {label}
                      {ownData && (
                        <Folder16Regular
                          style={{ verticalAlign: "-3px", marginLeft: 6 }}
                        />
                      )}
                    </span>
                    {desc && <span className="option-card-desc">{desc}</span>}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {payload.allowFreeform && !freeformVisible && (
          <button
            type="button"
            onClick={() => setFreeformVisible(true)}
            className="copilot-link-btn"
          >
            Or type a different answer…
          </button>
        )}

        {payload.allowFreeform && freeformVisible && (
          <div className="copilot-section">
            <span className="copilot-section-label">Custom answer</span>
            <textarea
              rows={2}
              value={freeform}
              onChange={(e) => setFreeform(e.target.value)}
              onKeyDown={(e) => {
                // Enter submits; Shift+Enter inserts a newline.
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (canSubmit && !submitting) {
                    onSubmit({
                      answer: freeform.trim(),
                      wasFreeform: true,
                    });
                  }
                }
              }}
              placeholder={
                hasChoices
                  ? "Type a different answer..."
                  : "Type your answer..."
              }
              style={{ width: "100%", fontFamily: "inherit" }}
            />
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
