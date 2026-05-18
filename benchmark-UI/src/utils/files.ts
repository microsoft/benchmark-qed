import type { EditorKind } from "../types";

const LANGUAGE_BY_EXT: Record<string, string> = {
  md: "markdown",
  markdown: "markdown",
  json: "json",
  jsonc: "json",
  yaml: "yaml",
  yml: "yaml",
  toml: "ini",
  ini: "ini",
  py: "python",
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  html: "html",
  htm: "html",
  css: "css",
  scss: "scss",
  sh: "shell",
  bash: "shell",
  zsh: "shell",
  sql: "sql",
  xml: "xml",
  txt: "plaintext",
  log: "plaintext",
};

export function getExt(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

function isDotEnvFile(name: string): boolean {
  const lower = name.toLowerCase();
  return lower === ".env" || lower.startsWith(".env.");
}

export function detectLanguage(name: string): string {
  if (isDotEnvFile(name)) return "shell";
  return LANGUAGE_BY_EXT[getExt(name)] ?? "plaintext";
}

export function detectKind(name: string): EditorKind {
  if (isDotEnvFile(name)) return "code";
  const ext = getExt(name);
  if (ext === "md" || ext === "markdown") return "markdown";
  if (ext === "csv" || ext === "tsv") return "csv";
  if (ext in LANGUAGE_BY_EXT) return "code";
  return "unsupported";
}

export function sortNodes<T extends { kind: string; name: string }>(
  entries: T[],
): T[] {
  const visibleEntries = entries.filter(
    (entry) => entry.name.toLowerCase() !== ".ds_store",
  );
  return visibleEntries.slice().sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === "directory" ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}
