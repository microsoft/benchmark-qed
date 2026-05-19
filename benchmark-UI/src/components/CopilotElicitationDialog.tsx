import { useEffect, useMemo, useState } from "react";
import { Folder16Regular, Document16Regular } from "@fluentui/react-icons";
import type {
  ElicitationAnswer,
  ElicitationPayload,
  JsonSchema,
} from "../copilot/types";
import { pickFiles, pickFolder } from "../copilot/pickers";

// A field is a "folder picker" if its schema format or its property name
// says so. Same idea for "files" — except files can target either an array
// or a newline/comma-separated string (we pick based on schema.type).
function isFolderField(name: string, sub: JsonSchema): boolean {
  const fmt = (sub.format ?? "").toLowerCase();
  if (["directory", "folder", "path", "dirpath"].includes(fmt)) return true;
  return /^(root_?path|directory|folder|workspace|location|out(put)?_?dir)$/i.test(
    name,
  );
}

function isFilesField(name: string, sub: JsonSchema): boolean {
  const fmt = (sub.format ?? "").toLowerCase();
  if (["file", "files", "filepath", "uri-reference"].includes(fmt)) return true;
  if (sub.type === "array" && (sub.items?.format ?? "").toLowerCase() === "file")
    return true;
  return /^(files?|dataset|documents?|sources?|inputs?|uploads?)$/i.test(name);
}

interface Props {
  payload: ElicitationPayload;
  submitting: boolean;
  onSubmit: (answer: ElicitationAnswer) => void;
  onCancel: () => void;
}

type FormState = Record<string, string | number | boolean>;

function defaultsFor(schema: JsonSchema): FormState {
  const props = schema.properties ?? {};
  const out: FormState = {};
  for (const [key, sub] of Object.entries(props)) {
    if (sub.default !== undefined) {
      out[key] = sub.default as FormState[string];
    } else if (sub.enum && sub.enum.length > 0) {
      out[key] = sub.enum[0] as FormState[string];
    } else if (sub.type === "boolean") {
      out[key] = false;
    } else if (sub.type === "number" || sub.type === "integer") {
      out[key] = sub.minimum ?? 0;
    } else {
      out[key] = "";
    }
  }
  return out;
}

function coerce(schema: JsonSchema, value: string | boolean | number): unknown {
  if (schema.type === "boolean") return Boolean(value);
  if (schema.type === "number" || schema.type === "integer") {
    const n = typeof value === "number" ? value : Number(value);
    return Number.isFinite(n) ? n : undefined;
  }
  if (schema.type === "array") {
    // Picker writes JSON-encoded arrays; freeform input may be newline/comma-separated.
    if (typeof value !== "string") return value;
    const trimmed = value.trim();
    if (!trimmed) return [];
    if (trimmed.startsWith("[")) {
      try {
        return JSON.parse(trimmed);
      } catch {
        /* fall through */
      }
    }
    return trimmed
      .split(/\r?\n|,/)
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return value;
}

export function CopilotElicitationDialog({
  payload,
  submitting,
  onSubmit,
  onCancel,
}: Props) {
  void onCancel;
  const schema = payload.requestedSchema;
  const properties = useMemo(
    () => Object.entries(schema.properties ?? {}),
    [schema],
  );
  const required = useMemo(
    () => new Set(schema.required ?? []),
    [schema.required],
  );
  const [form, setForm] = useState<FormState>(() => defaultsFor(schema));

  useEffect(() => {
    setForm(defaultsFor(schema));
  }, [schema]);

  const canSubmit = useMemo(() => {
    for (const key of required) {
      const v = form[key];
      if (v === undefined || v === "" || v === null) return false;
    }
    return true;
  }, [form, required]);

  const accept = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    const content: Record<string, unknown> = {};
    for (const [key, sub] of properties) {
      const raw = form[key];
      if (raw === "" || raw === undefined) continue;
      content[key] = coerce(sub, raw);
    }
    onSubmit({ action: "accept", content });
  };

  // `mode: "url"` means the agent wants the user to visit a link (e.g. OAuth).
  if (payload.mode === "url") {
    const url = (schema.properties?.url?.default as string) ?? "";
    return (
      <div className="copilot-inline-card">
        <div className="copilot-inline-card-header">
          <strong>Copilot needs you to open a link:</strong>
        </div>
        <div className="copilot-inline-card-body">
          <p style={{ margin: 0 }}>{payload.message}</p>
          {url && (
            <a href={url} target="_blank" rel="noreferrer">
              {url}
            </a>
          )}
          <div className="copilot-inline-actions">
            <button className="btn" onClick={() => onSubmit({ action: "decline" })}>
              Decline
            </button>
            <button
              className="btn btn-primary"
              disabled={submitting}
              onClick={() => onSubmit({ action: "accept" })}
            >
              I've completed it
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="copilot-inline-card">
      <div className="copilot-inline-card-header">
        <strong>{schema.title ?? "Copilot needs information"}</strong>
      </div>
      <form className="copilot-inline-card-body" onSubmit={accept}>
        <p style={{ margin: 0 }}>{payload.message}</p>
        {properties.length === 0 && (
          <p style={{ opacity: 0.7 }}>(No fields requested — accept to confirm.)</p>
        )}
        {properties.map(([key, sub]) => {
          const label = sub.title ?? key;
          const isReq = required.has(key);
          const value = form[key];

          if (sub.enum && sub.enum.length > 0) {
            return (
              <label className="field" key={key}>
                <span>
                  {label}
                  {isReq && " *"}
                </span>
                <select
                  value={String(value ?? "")}
                  onChange={(e) =>
                    setForm((prev) => ({ ...prev, [key]: e.target.value }))
                  }
                >
                    {sub.enum.map((opt) => (
                      <option key={String(opt)} value={String(opt)}>
                        {String(opt)}
                      </option>
                    ))}
                  </select>
                  {sub.description && <small>{sub.description}</small>}
                </label>
              );
            }

            if (sub.type === "boolean") {
              return (
                <label
                  className="field field-inline"
                  key={key}
                  style={{ flexDirection: "row", alignItems: "center", gap: 8 }}
                >
                  <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(e) =>
                      setForm((prev) => ({ ...prev, [key]: e.target.checked }))
                    }
                  />
                  <span>
                    {label}
                    {isReq && " *"}
                  </span>
                  {sub.description && <small>{sub.description}</small>}
                </label>
              );
            }

            if (sub.type === "number" || sub.type === "integer") {
              return (
                <label className="field" key={key}>
                  <span>
                    {label}
                    {isReq && " *"}
                  </span>
                  <input
                    type="number"
                    value={String(value ?? "")}
                    min={sub.minimum}
                    max={sub.maximum}
                    step={sub.type === "integer" ? 1 : "any"}
                    onChange={(e) =>
                      setForm((prev) => ({ ...prev, [key]: e.target.value }))
                    }
                  />
                  {sub.description && <small>{sub.description}</small>}
                </label>
              );
            }

            const long = (sub.maxLength ?? 0) > 120;
            const folderField = isFolderField(key, sub);
            const filesField = isFilesField(key, sub);
            const isArrayFiles = filesField && sub.type === "array";

            const pickFolderFor = async () => {
              try {
                const p = await pickFolder();
                if (p) setForm((prev) => ({ ...prev, [key]: p }));
              } catch (err) {
                console.error(err);
              }
            };
            const pickFilesFor = async () => {
              try {
                const paths = await pickFiles();
                if (paths && paths.length > 0) {
                  // For string fields we join with newlines; for array fields
                  // we serialize to JSON (the form coerces strings → JSON when
                  // the schema is array on accept below).
                  const serialized = isArrayFiles
                    ? JSON.stringify(paths)
                    : paths.join("\n");
                  setForm((prev) => ({ ...prev, [key]: serialized }));
                }
              } catch (err) {
                console.error(err);
              }
            };

            return (
              <label className="field" key={key}>
                <span>
                  {label}
                  {isReq && " *"}
                </span>
                {long || filesField ? (
                  <textarea
                    rows={filesField ? 4 : 3}
                    value={String(value ?? "")}
                    onChange={(e) =>
                      setForm((prev) => ({ ...prev, [key]: e.target.value }))
                    }
                    placeholder={
                      filesField
                        ? "One absolute path per line, or click 'Pick files'."
                        : undefined
                    }
                  />
                ) : (
                  <input
                    type={sub.format === "password" ? "password" : "text"}
                    value={String(value ?? "")}
                    onChange={(e) =>
                      setForm((prev) => ({ ...prev, [key]: e.target.value }))
                    }
                    placeholder={
                      folderField ? "/absolute/path — or click 'Pick folder'" : undefined
                    }
                  />
                )}
                {(folderField || filesField) && (
                  <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
                    {folderField && (
                      <button
                        type="button"
                        className="btn"
                        onClick={pickFolderFor}
                      >
                        <Folder16Regular
                          style={{ verticalAlign: "-3px", marginRight: 4 }}
                        />
                        Pick folder
                      </button>
                    )}
                    {filesField && (
                      <button
                        type="button"
                        className="btn"
                        onClick={pickFilesFor}
                      >
                        <Document16Regular
                          style={{ verticalAlign: "-3px", marginRight: 4 }}
                        />
                        Pick files
                      </button>
                    )}
                  </div>
                )}
                {sub.description && <small>{sub.description}</small>}
              </label>
            );
          })}

          <div className="copilot-inline-actions">
            <button
              type="button"
              className="btn"
              onClick={() => onSubmit({ action: "decline" })}
            >
              Decline
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || !canSubmit}
            >
              {submitting ? "Sending..." : "Submit"}
            </button>
          </div>
        </form>
    </div>
  );
}
