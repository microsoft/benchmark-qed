// Thin client around init-runner's native pickers, used by the Copilot dialogs
// so the agent can ask for "a folder" or "files" and we hand back real paths
// instead of making the user paste them.

const INIT_RUNNER_URL =
  (typeof import.meta !== "undefined" &&
    (import.meta as { env?: { VITE_INIT_RUNNER_URL?: string } }).env
      ?.VITE_INIT_RUNNER_URL) ||
  "http://localhost:8787";

export interface PickerError {
  cancelled?: boolean;
  error?: string;
}

export async function pickFolder(): Promise<string | null> {
  const res = await fetch(`${INIT_RUNNER_URL}/api/pick-folder`);
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as PickerError;
    throw new Error(err.error || `Folder picker failed (${res.status})`);
  }
  const data = (await res.json()) as { path?: string; cancelled?: boolean };
  if (data.cancelled) return null;
  return data.path ?? null;
}

export async function pickFiles(): Promise<string[] | null> {
  const res = await fetch(`${INIT_RUNNER_URL}/api/pick-files`);
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as PickerError;
    throw new Error(err.error || `File picker failed (${res.status})`);
  }
  const data = (await res.json()) as {
    paths?: string[];
    cancelled?: boolean;
  };
  if (data.cancelled) return null;
  return data.paths ?? null;
}

/**
 * Loose heuristic: does this question/property name look like it's asking for
 * a filesystem path? Used to surface a "Pick folder" button automatically
 * when the SKILL doesn't supply a JSON Schema format hint.
 */
export function looksLikeFolderPrompt(text: string): boolean {
  return /\b(folder|directory|path|root\s?path|workspace|location|where)\b/i.test(
    text,
  );
}

export function looksLikeFilesPrompt(text: string): boolean {
  return /\b(file|files|dataset|upload|import|csv|json|jsonl|documents?)\b/i.test(
    text,
  );
}
