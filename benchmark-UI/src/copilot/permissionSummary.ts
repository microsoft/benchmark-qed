import type { PermissionPayload } from "./types";

/**
 * Build a human-readable one-line description of a permission request.
 *
 * The SDK ships different field names per `kind`. For shell prompts we get
 * `fullCommandText`; for file edits we get `fileName`; for `read` / `list`
 * permissions we typically only see `directory` or `path`. Older callers
 * used to fall back to `payload.kind` alone, which produced confusing
 * one-word summaries like "Q: read".
 */
export function summarizePermission(payload: PermissionPayload): string {
  if (payload.fullCommandText) {
    return `Run command: ${payload.fullCommandText}`;
  }

  const target = pickPath(payload);
  const kind = (payload.kind || "").toLowerCase();

  if (target) {
    switch (kind) {
      case "read":
        return `Read ${target}`;
      case "list":
      case "list_directory":
      case "listdir":
        return `List directory ${target}`;
      case "write":
      case "create":
        return `Write ${target}`;
      case "edit":
      case "update":
        return `Edit ${target}`;
      case "delete":
      case "remove":
        return `Delete ${target}`;
      default:
        return `${payload.kind || "Access"}: ${target}`;
    }
  }

  if (payload.toolName) return `Tool: ${payload.toolName}`;
  return payload.kind || "Approval required";
}

function pickPath(payload: PermissionPayload): string | null {
  // Direct file/dir-ish keys, in priority order.
  const direct = [
    "fileName",
    "filePath",
    "path",
    "targetPath",
    "directory",
    "directoryPath",
    "folder",
    "uri",
  ];
  for (const key of direct) {
    const v = payload[key];
    if (typeof v === "string" && v.trim()) return v;
  }
  // Nested under common containers (`arguments`/`args`/`input`).
  for (const container of ["arguments", "args", "input"]) {
    const c = payload[container];
    if (c && typeof c === "object") {
      for (const key of direct) {
        const v = (c as Record<string, unknown>)[key];
        if (typeof v === "string" && v.trim()) return v;
      }
    }
  }
  return null;
}
