// Persistent registry of question-quality reports the user has generated.
// Stored in localStorage so the list survives reloads and lets the user
// re-open a report whose editor tab was closed.

const STORAGE_KEY = "benchmark-qed:recent-quality-reports";
const MAX_ENTRIES = 20;

export interface RecentReport {
  id: string;
  /** Absolute path to the saved QUALITY_REPORT.md file. */
  reportPath: string;
  /** Absolute path of the folder containing the report. */
  destinationPath: string;
  /** Short label (typically the destination folder's basename). */
  label: string;
  /** ISO timestamp. */
  savedAt: string;
  /** Labels of the compared question sets, for display. */
  setLabels: string[];
}

function readAll(): RecentReport[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (e): e is RecentReport =>
        e &&
        typeof e.id === "string" &&
        typeof e.reportPath === "string" &&
        typeof e.destinationPath === "string",
    );
  } catch {
    return [];
  }
}

function writeAll(entries: RecentReport[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    // localStorage may be unavailable or full; silently ignore.
  }
}

export function listRecentReports(): RecentReport[] {
  return readAll().sort((a, b) => (a.savedAt < b.savedAt ? 1 : -1));
}

export function addRecentReport(
  entry: Omit<RecentReport, "id" | "savedAt"> & { savedAt?: string },
): RecentReport {
  const record: RecentReport = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    savedAt: entry.savedAt ?? new Date().toISOString(),
    reportPath: entry.reportPath,
    destinationPath: entry.destinationPath,
    label: entry.label,
    setLabels: entry.setLabels,
  };
  // Drop any prior entry pointing at the same file so the list dedupes.
  const next = [record, ...readAll().filter((e) => e.reportPath !== record.reportPath)];
  writeAll(next);
  return record;
}

export function removeRecentReport(id: string): void {
  writeAll(readAll().filter((e) => e.id !== id));
}
