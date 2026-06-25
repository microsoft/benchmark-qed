import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AddWorkspaceTabsDialog } from "./components/AddWorkspaceTabsDialog";
import {
  DatasetDownloadDialog,
  type PredefinedDataset,
} from "./components/DatasetDownloadDialog";
import {
  LoadDatasetDialog,
  type LoadDatasetSubmit,
} from "./components/LoadDatasetDialog";
import { FileEditor } from "./components/FileEditor";
import { FolderTree } from "./components/FolderTree";
import type { InitJob } from "./components/InitJobsPanel";
import {
  InitRunDialog,
  type InitRunRequest,
} from "./components/InitRunDialog";
import { RunConfigDialog } from "./components/RunConfigDialog";
import { JobLogViewer } from "./components/JobLogViewer";
import type { RunJob } from "./components/RunJobsBottomTab";
import { MultiConfigDialog, type DetectedConfig } from "./components/MultiConfigDialog";
import { TreeCreateDialog } from "./components/TreeCreateDialog";
import { TreeDeleteDialog } from "./components/TreeDeleteDialog";
import { CopyWorkspaceDialog } from "./components/CopyWorkspaceDialog";
import { RenameWorkspaceDialog } from "./components/RenameWorkspaceDialog";
import { CopyNodeDialog } from "./components/CopyNodeDialog";
import { RenameNodeDialog } from "./components/RenameNodeDialog";
import { WorkspaceActionsMenu } from "./components/WorkspaceActionsMenu";
import { CopilotPanel } from "./components/CopilotPanel";
import {
  EvaluateQuestionsPickerForm,
  type EvaluateQuestionsSubmit,
} from "./components/EvaluateQuestionsPickerDialog";
import { addRecentReport, type RecentReport } from "./recentReports";
import { RunnerBlobFileSource } from "./sources/RunnerBlobFileSource.ts";
import { RunnerPathFileSource } from "./sources/RunnerPathFileSource";
import type {
  FileSource,
  OpenFile,
  SourceKind,
  TreeNode,
  Workspace,
  WorkspaceConfigType,
} from "./types";
import {
  WeatherMoon24Regular,
  WeatherSunny24Regular,
  ChevronRight16Regular,
  ChevronDown16Regular,
  Folder16Regular,
  Cloud16Regular,
  ArrowDownload16Regular,
  Play16Regular,
  ArrowSync16Regular,
  Document16Regular,
  FolderAdd16Regular,
  Dismiss16Regular,
  ClipboardTask16Regular,
  PanelLeft24Regular,
  Copy16Regular,
  Navigation16Regular,
  Edit16Regular,
} from '@fluentui/react-icons';
import { detectKind, detectLanguage } from "./utils/files";
import { ActivityLogPanel, type ActivityLogEntry } from "./components/ActivityLogPanel";
import { JobsPanel } from "./components/JobsPanel";
import "./App.css";

const AUTOSAVE_DELAY_MS = 800;
const INIT_RUNNER_URL = "http://localhost:8787";
const PERSISTED_WORKSPACES_KEY = "workspaces";

/**
 * Skills available from the "✨ AI Assistant" launcher. Each entry maps to a
 * skill directory under `.apm/skills/<name>/SKILL.md`. To add a new skill:
 *   1. Create `.apm/skills/<your-skill>/SKILL.md`.
 *   2. Append an entry here with the skill name, a short user-facing label,
 *      a one-line description, and an initialPrompt that tells the agent to
 *      run the skill end-to-end.
 *
 * The "Execution context: ui" marker is recognized by benchmark-qed-setup;
 * harmless for skills that don't use it.
 */
interface CopilotSkill {
  id: string;
  label: string;
  description: string;
  initialPrompt: string;
}

const COPILOT_SKILLS: CopilotSkill[] = [
  {
    id: "benchmark-qed-setup",
    label: "Setup workspace",
    description: "Initialize a benchmark-qed workspace and generate settings.yaml.",
    initialPrompt:
      "Execution context: ui\n\nRun the benchmark-qed-setup skill end-to-end. Do not display the skill's contents in chat. Go straight into asking me the first question (one at a time) using the skill's ask_user / elicitation flow. Be brief — short prompts only.",
  },
  {
    id: "benchmark-qed-autoq",
    label: "Generate questions (autoq)",
    description: "Generate benchmark questions and assertions from input data.",
    initialPrompt:
      "Execution context: ui\n\nRun the benchmark-qed-autoq skill end-to-end. Do not display the skill's contents in chat. Ask me one question at a time using the skill's ask_user / elicitation flow. Be brief — short prompts only.",
  },
  {
    id: "benchmark-qed-autoe",
    label: "Evaluate RAG outputs (autoe)",
    description: "Score and compare RAG answers with pairwise / reference / assertion metrics.",
    initialPrompt:
      "Execution context: ui\n\nRun the benchmark-qed-autoe skill end-to-end. Do not display the skill's contents in chat. Ask me one question at a time using the skill's ask_user / elicitation flow. Be brief — short prompts only.",
  },
  {
    id: "benchmark-qed-question-quality",
    label: "Evaluate question quality",
    description: "Audit autoq question sets and compare two or more sets head-to-head.",
    initialPrompt:
      "Execution context: ui\n\nRun the benchmark-qed-question-quality skill end-to-end. Do not display the skill's contents in chat. Ask me one question at a time using the skill's ask_user / elicitation flow. Be brief — short prompts only.",
  },
];

const DEFAULT_COPILOT_SKILL_ID = COPILOT_SKILLS[0].id;

/**
 * Build the prompt sent to the benchmark-qed-question-quality skill given
 * the folders the user picked. Shared by the inline picker rendered inside
 * the Copilot popup.
 */
function buildQuestionQualityPrompt(params: EvaluateQuestionsSubmit): string {
  const list = params.entries
    .map((e, i) => `${i + 1}. Label: "${e.label}" — Path: ${e.path}`)
    .join("\n");
  const destFolder = params.destinationPath.replace(/[\\/]+$/, "");
  // Preserve the user's native path style (backslash on Windows, slash on
  // POSIX) so the path the agent receives is the one their shell expects.
  const joinSep = destFolder.includes("\\") ? "\\" : "/";
  const savePath = `${destFolder}${joinSep}${params.reportFilename}`;
  return [
    "Execution context: ui",
    "",
    "Run the benchmark-qed-question-quality skill to evaluate and",
    "compare the following question sets:",
    "",
    list,
    "",
    "For each path, look inside for autoq question JSON files",
    "(activity_global_questions, activity_local_questions,",
    "data_global_questions, data_local_questions). Infer the question",
    "type from the filename or parent folder. Score each question",
    "against the per-type criteria defined in the skill, then produce:",
    "  1. A per-set summary table.",
    "  2. A side-by-side comparison table with a Winner column.",
    "  3. Failure examples grouped by criterion.",
    "  4. A short verdict and 3–6 actionable recommendations.",
    "",
    `Save the final report at exactly this path (use this exact file`,
    `name — do not rename it):`,
    "",
    savePath,
    "",
    "Do NOT write the report (or any intermediate scripts/files)",
    "anywhere else on disk. If you need to run helper scripts, place",
    "them inside the destination folder and clean them up when done.",
    "",
    "Do not display the skill's contents in chat. Only ask me a",
    "follow-up if you genuinely cannot infer a question type or find",
    "the question files. Be brief.",
  ].join("\n");
}

type PersistedWorkspace =
  | {
      kind: "local";
      name: string;
      rootPath: string;
      configType?: WorkspaceConfigType;
    /** Marks this workspace as a parent container for child workspaces. */
    hasChildWorkspaces?: boolean;
    /** Root path of this workspace's parent, when nested as a child. */
    parentRootPath?: string;
      /** When set, marks this workspace as a copy of the workspace whose
       *  rootPath matches this value. Used to nest copies under their source
       *  in the sidebar tree. */
      copyOfRootPath?: string;
    }
  | {
      kind: "blob";
      label: string;
      accountUrl: string;
      containerName: string;
      prefix: string;
    };

function loadPersistedWorkspaces(): PersistedWorkspace[] {
  try {
    const raw = localStorage.getItem(PERSISTED_WORKSPACES_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (w): w is PersistedWorkspace =>
        typeof w === "object" &&
        w !== null &&
        "kind" in w &&
        ((w as { kind: unknown }).kind === "local" ||
          (w as { kind: unknown }).kind === "blob"),
    );
  } catch {
    return [];
  }
}

function savePersistedWorkspaces(workspaces: PersistedWorkspace[]): void {
  try {
    localStorage.setItem(
      PERSISTED_WORKSPACES_KEY,
      JSON.stringify(workspaces),
    );
  } catch {
    // ignore storage errors (quota, private mode, etc.)
  }
}

function workspaceStableKey(ws: {
  sourceKind: SourceKind;
  rootPath?: string;
  name: string;
}): string {
  return ws.rootPath
    ? `${ws.sourceKind}:${ws.rootPath}`
    : `${ws.sourceKind}:${ws.name}`;
}

function normalizeLocalPath(value?: string): string | undefined {
  if (!value) return undefined;
  const raw = value.trim().replace(/\\/g, "/");
  if (!raw) return undefined;

  const driveMatch = raw.match(/^[a-zA-Z]:/);
  const drive = driveMatch?.[0] ?? "";
  let rest = drive ? raw.slice(drive.length) : raw;
  const isAbsolute = rest.startsWith("/");
  rest = rest.replace(/^\/+/, "");

  const parts = rest.split("/");
  const stack: string[] = [];
  for (const part of parts) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (stack.length > 0 && stack[stack.length - 1] !== "..") {
        stack.pop();
      } else if (!isAbsolute) {
        stack.push(part);
      }
      continue;
    }
    stack.push(part);
  }

  if (drive) {
    if (isAbsolute) {
      return stack.length > 0 ? `${drive}/${stack.join("/")}` : `${drive}/`;
    }
    return stack.length > 0 ? `${drive}/${stack.join("/")}` : drive;
  }

  if (isAbsolute) {
    return stack.length > 0 ? `/${stack.join("/")}` : "/";
  }
  return stack.length > 0 ? stack.join("/") : undefined;
}

function localPathIdentity(value?: string): string | undefined {
  const normalized = normalizeLocalPath(value);
  return normalized ? normalized.toLowerCase() : undefined;
}

function hasSameLocalRootPath(a?: string, b?: string): boolean {
  const left = localPathIdentity(a);
  const right = localPathIdentity(b);
  return !!left && !!right && left === right;
}

// True when two local root paths are the same folder, or one is an ancestor of
// the other. Uses normalized, separator-agnostic identities so Windows
// backslash paths (e.g. "C:\\a\\b" vs "C:\\a\\b\\output") match correctly.
function rootPathsRelated(a?: string, b?: string): boolean {
  const left = localPathIdentity(a);
  const right = localPathIdentity(b);
  if (!left || !right) return false;
  return (
    left === right ||
    left.startsWith(`${right}/`) ||
    right.startsWith(`${left}/`)
  );
}

type LocalSourceWithResolvedRoot = {
  getResolvedRootPath?: () => string | undefined;
};

function getResolvedLocalRootPath(source: FileSource): string | undefined {
  if (source.kind !== "local") return undefined;
  const candidate = source as FileSource & LocalSourceWithResolvedRoot;
  if (typeof candidate.getResolvedRootPath !== "function") return undefined;
  return candidate.getResolvedRootPath();
}

function relativePathIdentity(value?: string): string | undefined {
  if (!value) return undefined;
  const normalized = normalizeLocalPath(value);
  if (!normalized) return undefined;
  return normalized
    .replace(/^[a-zA-Z]:/, "")
    .replace(/^\/+/, "")
    .toLowerCase();
}

type JobLike = {
  id: string;
  startedAt: string;
  rootPath?: string;
};

// Sort by most recent first and cap the visible history. We keep all jobs
// (no rootPath dedup) so users can see the full run history per workspace.
function normalizeJobs<T extends JobLike>(jobs: T[]): T[] {
  const sorted = [...jobs].sort(
    (a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt),
  );
  return sorted.slice(0, 100);
}

// Persist finished jobs (succeeded/failed/cancelled) to localStorage so they
// survive backend restarts or local jobs.json being wiped. Running jobs are
// authoritative from the runner and are not persisted client-side.
const PERSISTED_INIT_JOBS_KEY = "init-jobs";
const PERSISTED_RUN_JOBS_KEY = "run-jobs";
// Ids of init jobs whose workspace has already been imported into the
// sidebar (or explicitly dismissed by closing that workspace). Persisted so
// closing a workspace stays closed across reloads instead of being
// resurrected by the init-job auto-import effect.
const PERSISTED_IMPORTED_INIT_JOB_IDS_KEY = "imported-init-job-ids";

function loadImportedInitJobIds(): Set<string> {
  try {
    const raw = localStorage.getItem(PERSISTED_IMPORTED_INIT_JOB_IDS_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? new Set(parsed.filter((v): v is string => typeof v === "string")) : new Set();
  } catch {
    return new Set();
  }
}

function saveImportedInitJobIds(ids: Set<string>): void {
  try {
    localStorage.setItem(
      PERSISTED_IMPORTED_INIT_JOB_IDS_KEY,
      JSON.stringify([...ids]),
    );
  } catch {
    // ignore quota / private mode failures
  }
}
const PERSISTED_JOB_OUTPUT_LIMIT = 50_000; // chars per job
const PERSISTED_JOB_LIMIT = 100;

type JobStatusLike = { status?: string };

function truncateJobOutput(output: unknown): string {
  if (typeof output !== "string") return "";
  if (output.length <= PERSISTED_JOB_OUTPUT_LIMIT) return output;
  const head = output.slice(0, PERSISTED_JOB_OUTPUT_LIMIT - 2000);
  const tail = output.slice(-2000);
  return `${head}\n…[truncated ${
    output.length - PERSISTED_JOB_OUTPUT_LIMIT
  } chars]…\n${tail}`;
}

function loadPersistedJobs<T>(key: string): T[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as T[]) : [];
  } catch {
    return [];
  }
}

function savePersistedJobs<T extends JobLike & JobStatusLike>(
  key: string,
  jobs: T[],
): void {
  try {
    const finished = jobs.filter((j) => j.status && j.status !== "running");
    const trimmed = finished.slice(0, PERSISTED_JOB_LIMIT).map((j) => ({
      ...j,
      output: truncateJobOutput((j as unknown as { output?: unknown }).output),
    }));
    localStorage.setItem(key, JSON.stringify(trimmed));
  } catch {
    // ignore quota / private mode failures
  }
}

// Merge runner-returned jobs with localStorage-persisted jobs. Runner wins on
// id collisions; localStorage fills in jobs the runner has forgotten.
function mergeJobs<T extends JobLike>(remote: T[], persisted: T[]): T[] {
  const seen = new Set(remote.map((j) => j.id));
  const merged = [...remote];
  for (const j of persisted) {
    if (!seen.has(j.id)) merged.push(j);
  }
  return normalizeJobs(merged);
}

type ActiveView = 
  | { type: "file"; file: OpenFile }
  | { type: "run-job"; jobId: string }
  | { type: "init-job"; jobId: string }
  | { type: "copilot" }
  | null;

// Mirrors detectConfigType() in benchmark-UI/scripts/init-runner.mjs so blob
// workspaces can detect configs client-side from yaml content fetched via
// the workspace's runner-backed blob source.
function detectConfigTypeFromYaml(content: string): WorkspaceConfigType | null {
  if (/^\s*base:\s*$|^\s*others:\s*[[\-]|score_min:|score_max:|pairwise/im.test(content)) {
    return "autoe_pairwise";
  }
  if (/^\s*reference:\s*$|^\s*generated:\s*[[\-]|^\s*- name:|reference_base_path/im.test(content)) {
    return "autoe_reference";
  }
  if (/^\s*generated:\s*$|^\s*assertions:\s*$|assertions_path:|pass_threshold:|detect_discovery/im.test(content)) {
    return "autoe_assertion";
  }
  if (/^\s*input:\s*$|^\s*output:/im.test(content)) {
    return "autoq";
  }
  if (/input|output|questions/im.test(content)) {
    return "autoq";
  }
  return null;
}

async function detectBlobConfigs(workspace: Workspace): Promise<DetectedConfig[]> {
  const results: DetectedConfig[] = [];
  // Try to derive a `blob://<container>[/<prefix>]` URI for nicer display.
  // Falls back to the workspace name if we can't introspect the source.
  const blobBase = (() => {
    const src = workspace.source as unknown as {
      kind?: string;
      // RunnerBlobFileSource internals, read defensively.
      containerName?: string;
      prefix?: string;
    };
    if (workspace.sourceKind !== "blob") return workspace.name;
    const container = src.containerName ?? "";
    const prefix = (src.prefix ?? "").replace(/\/+$/, "");
    if (!container) return workspace.name;
    return prefix
      ? `blob://${container}/${prefix}`
      : `blob://${container}`;
  })();

  // Root settings.yaml
  const rootSettings = workspace.rootNodes.find(
    (n) => n.kind === "file" && n.name === "settings.yaml",
  );
  if (rootSettings) {
    try {
      const content = await workspace.source.readFile(rootSettings);
      const cfg = detectConfigTypeFromYaml(content);
      if (cfg) {
        results.push({
          path: `${blobBase}/settings.yaml`,
          name: workspace.name,
          configType: cfg,
          isRoot: true,
          blobSubdir: "",
        });
      }
    } catch {
      // Couldn't read root settings.yaml; skip.
    }
  }
  // One level deep: subdirectories that contain a settings.yaml
  for (const node of workspace.rootNodes) {
    if (node.kind !== "directory") continue;
    try {
      const children = await workspace.source.listChildren(node);
      const settings = children.find(
        (c) => c.kind === "file" && c.name === "settings.yaml",
      );
      if (!settings) continue;
      const content = await workspace.source.readFile(settings);
      const cfg = detectConfigTypeFromYaml(content);
      if (!cfg) continue;
      results.push({
        path: `${blobBase}/${node.name}/settings.yaml`,
        name: node.name,
        configType: cfg,
        isRoot: false,
        blobSubdir: node.name,
      });
    } catch {
      // Skip subdirs we can't read.
    }
  }
  return results;
}

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    (localStorage.getItem("theme") as "dark" | "light") ?? "dark"
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  // Mirror of `workspaces` for callbacks that need to read the latest list
  // without taking a dependency that would cascade-recreate them on every
  // change (e.g. addWorkspace).
  const workspacesRef = useRef<Workspace[]>([]);
  useEffect(() => {
    workspacesRef.current = workspaces;
  }, [workspaces]);
  const [activeView, setActiveView] = useState<ActiveView>(null);
  const [openFiles, setOpenFiles] = useState<OpenFile[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileKey = (f: { workspaceId: string; node: { path: string } }) =>
    `${f.workspaceId}::${f.node.path}`;

  // Keep open files in sync with the active view so file tabs remain visible
  // when the user switches between files or to a job log tab.
  useEffect(() => {
    if (activeView?.type !== "file") return;
    const active = activeView.file;
    setOpenFiles((prev) => {
      const key = fileKey(active);
      const idx = prev.findIndex((f) => fileKey(f) === key);
      if (idx === -1) return [...prev, active];
      const next = prev.slice();
      next[idx] = active;
      return next;
    });
  }, [activeView]);

  const [addWorkspaceDialogOpen, setAddWorkspaceDialogOpen] = useState(false);
  const [initDialogOpen, setInitDialogOpen] = useState(false);
  const [copilotWizardOpen, setCopilotWizardOpen] = useState(false);
  const [copilotSkillId, setCopilotSkillId] = useState<string>(
    DEFAULT_COPILOT_SKILL_ID,
  );
  /**
   * When set, this overrides the skill's default `initialPrompt`. Used by the
   * Evaluate Question Quality flow so the multi-folder picker can hand the
   * agent a fully-formed task instead of asking it to elicit folder counts.
   */
  const [copilotPromptOverride, setCopilotPromptOverride] = useState<
    string | null
  >(null);
  /**
   * When true, the panel is allowed to start its session using the skill's
   * `initialPrompt`. While false (the panel's initial state), the panel
   * renders the in-popup skill picker and waits for the user's choice.
   */
  const [copilotPromptArmed, setCopilotPromptArmed] = useState(false);
  /**
   * When set, the Copilot popup renders the matching inline flow (e.g. the
   * folder picker for Evaluate Question Quality) instead of the skill grid,
   * keeping the user inside the same popup.
   */
  const [copilotInlineFlow, setCopilotInlineFlow] = useState<
    "question-quality" | null
  >(null);

  /**
   * In-flight question-quality report awaiting completion. Populated when
   * the user submits the picker; cleared as soon as the destination file
   * appears on disk (or on cancel / new submission). When set, the
   * onActivitySettled hook polls the runner FS for the file and, once it
   * exists, registers a RecentReport entry and opens the file in a tab.
   */
  const pendingReportRef = useRef<{
    reportPath: string;
    destinationPath: string;
    label: string;
    setLabels: string[];
  } | null>(null);

  const [initSubmitting, setInitSubmitting] = useState(false);
  const [initJobs, setInitJobs] = useState<InitJob[]>(() =>
    normalizeJobs(loadPersistedJobs<InitJob>(PERSISTED_INIT_JOBS_KEY)),
  );
  const importedInitJobIdsRef = useRef<Set<string>>(loadImportedInitJobIds());
  const hadRunningInitJobsRef = useRef(false);

  const [runJobs, setRunJobs] = useState<RunJob[]>(() =>
    normalizeJobs(loadPersistedJobs<RunJob>(PERSISTED_RUN_JOBS_KEY)),
  );
  const hadRunningRunJobsRef = useRef(false);
  const lastActiveRunRootsRef = useRef<Set<string>>(new Set());
  const [openRunJobIds, setOpenRunJobIds] = useState<Set<string>>(new Set());
  const [openInitJobIds, setOpenInitJobIds] = useState<Set<string>>(new Set());
  const [jobsPanelCollapsed, setJobsPanelCollapsed] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [workspaceFilter, setWorkspaceFilter] = useState("");
  // Optional source-kind filter for the sidebar: show all workspaces, or only
  // local / only blob ones.
  const [workspaceKindFilter, setWorkspaceKindFilter] = useState<
    "all" | "local" | "blob"
  >("all");
  // Tracks the initial blob-workspace restore so the sidebar can show a
  // "Loading blob workspaces (N/M)..." indicator instead of looking empty.
  const [blobRestoreProgress, setBlobRestoreProgress] = useState<{
    total: number;
    done: number;
    currentLabel: string | null;
  } | null>(null);
  // Per-workspace counter bumped on explicit create/delete actions to force
  // the FolderTree (and all cached subfolder children) to remount and
  // re-fetch. Persisted expansion state means expanded folders auto-reload.
  const [treeNonces, setTreeNonces] = useState<Record<string, number>>({});
  const bumpTreeNonce = useCallback((workspaceId: string) => {
    setTreeNonces((prev) => ({
      ...prev,
      [workspaceId]: (prev[workspaceId] ?? 0) + 1,
    }));
  }, []);
  // Debounce/remap repeated bumps per workspace to avoid excessive remounts
  // when background polling refreshes the same tree many times in quick
  // succession.
  const treeBumpTimersRef = useRef<Record<string, number>>({});
  const scheduleTreeNonceBump = useCallback(
    (workspaceId: string, delayMs = 250) => {
      const existing = treeBumpTimersRef.current[workspaceId];
      if (existing !== undefined) {
        window.clearTimeout(existing);
      }
      treeBumpTimersRef.current[workspaceId] = window.setTimeout(() => {
        bumpTreeNonce(workspaceId);
        delete treeBumpTimersRef.current[workspaceId];
      }, delayMs);
    },
    [bumpTreeNonce],
  );
  useEffect(() => {
    return () => {
      for (const timerId of Object.values(treeBumpTimersRef.current)) {
        window.clearTimeout(timerId);
      }
      treeBumpTimersRef.current = {};
    };
  }, []);
  const [datasetDialogWorkspaceId, setDatasetDialogWorkspaceId] = useState<string | null>(
    null,
  );
  const [datasetSubmitting, setDatasetSubmitting] = useState(false);
  const [loadDatasetDialogOpen, setLoadDatasetDialogOpen] = useState(false);
  const [loadDatasetSubmitting, setLoadDatasetSubmitting] = useState(false);
  const [configurePrompt, setConfigurePrompt] = useState<
    { workspace: string; name: string; summary: string } | null
  >(null);
  const [runConfigDialogOpen, setRunConfigDialogOpen] = useState(false);
  const [runConfigWorkspaceId, setRunConfigWorkspaceId] = useState<string | null>(
    null,
  );
  const [createDialog, setCreateDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
    kind: "file" | "directory";
    parentNode?: TreeNode;
  }>({
    open: false,
    workspaceId: null,
    kind: "file",
  });
  const [createSubmitting, setCreateSubmitting] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
    node?: TreeNode;
  }>({
    open: false,
    workspaceId: null,
  });
  const [deleteSubmitting, setDeleteSubmitting] = useState(false);
  const [copyDialog, setCopyDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
  }>({ open: false, workspaceId: null });
  const [copySubmitting, setCopySubmitting] = useState(false);
  const [renameDialog, setRenameDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
  }>({ open: false, workspaceId: null });
  const [copyNodeDialog, setCopyNodeDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
    node?: TreeNode;
  }>({ open: false, workspaceId: null });
  const [copyNodeSubmitting, setCopyNodeSubmitting] = useState(false);
  const [renameNodeDialog, setRenameNodeDialog] = useState<{
    open: boolean;
    workspaceId: string | null;
    node?: TreeNode;
  }>({ open: false, workspaceId: null });
  const [renameNodeSubmitting, setRenameNodeSubmitting] = useState(false);
  const [actionsMenuWorkspaceId, setActionsMenuWorkspaceId] = useState<
    string | null
  >(null);
  const [multiConfigDialog, setMultiConfigDialog] = useState<{
    open: boolean;
    folderPath: string;
    configs: DetectedConfig[];
    submitting: boolean;
  }>({
    open: false,
    folderPath: "",
    configs: [],
    submitting: false,
  });
  const [runSubfolderConfigWorkspaceId, setRunSubfolderConfigWorkspaceId] = useState<
    string | null
  >(null);
  const [detectingConfigsWorkspaceIds, setDetectingConfigsWorkspaceIds] =
    useState<Set<string>>(new Set());
  const beginDetectingConfigs = useCallback((id: string) => {
    setDetectingConfigsWorkspaceIds((prev) => {
      if (prev.has(id)) return prev;
      const next = new Set(prev);
      next.add(id);
      return next;
    });
  }, []);
  const endDetectingConfigs = useCallback((id: string) => {
    setDetectingConfigsWorkspaceIds((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }, []);
  const [activityLog, setActivityLog] = useState<ActivityLogEntry[]>([]);
  const [activityLogCollapsed, setActivityLogCollapsed] = useState(false);

  const addActivityLog = useCallback(
    (action: string, details?: string, type: "info" | "success" | "warning" | "error" = "info") => {
      const entry: ActivityLogEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        timestamp: new Date(),
        action,
        details,
        type,
      };
      setActivityLog((prev) => [entry, ...prev].slice(0, 100)); // Keep last 100 entries
    },
    [],
  );

  // Build a "workspace · folder · configType" label for the activity log.
  // - workspace: name of the loaded Workspace whose rootPath matches the
  //   job's rootPath (or is a containing/contained path); omitted if no
  //   match is found or if it would duplicate the folder name.
  // - folder:    basename of the job's rootPath.
  // - configType: the job's run kind (autoq, autoe_pairwise, …).
  const jobActivityLabel = useCallback(
    (rootPath?: string | null, configType?: string | null): string => {
      const folder = rootPath
        ? rootPath.split(/[/\\]/).filter(Boolean).pop()
        : undefined;
      const normalize = (p: string) =>
        p.replace(/[/\\]+$/, "").toLowerCase();
      let wsName: string | undefined;
      if (rootPath) {
        const target = normalize(rootPath);
        let best: { len: number; name: string } | undefined;
        for (const w of workspaces) {
          if (!w.rootPath) continue;
          const wp = normalize(w.rootPath);
          if (!wp) continue;
          const fits =
            target === wp ||
            target.startsWith(`${wp}/`) ||
            wp.startsWith(`${target}/`);
          if (fits && (!best || wp.length > best.len)) {
            best = { len: wp.length, name: w.name };
          }
        }
        wsName = best?.name;
      }
      const parts: string[] = [];
      if (wsName && wsName !== folder) parts.push(wsName);
      if (folder) parts.push(folder);
      if (configType) parts.push(configType);
      return parts.join(" \u00b7 ");
    },
    [workspaces],
  );

  // Track the minimal descriptor needed to re-create each workspace's source
  // across browser sessions. Keyed by workspace.id; mirrored to localStorage.
  const persistedWorkspacesRef = useRef<Map<string, PersistedWorkspace>>(
    new Map(),
  );
  const syncPersistedWorkspaces = useCallback(() => {
    savePersistedWorkspaces([...persistedWorkspacesRef.current.values()]);
  }, []);

  const isMissingPathError = useCallback((error: unknown): boolean => {
    const text = String(error).toLowerCase();
    return (
      text.includes("enoent") ||
      text.includes("no such file or directory") ||
      text.includes("scandir") ||
      text.includes("cannot find the path")
    );
  }, []);

  const prunePersistedLocalByRootPath = useCallback(
    (rootPath?: string): void => {
      if (!rootPath) return;
      const targetPathId = localPathIdentity(rootPath);
      if (!targetPathId) return;
      let removed = false;
      for (const [id, entry] of persistedWorkspacesRef.current) {
        if (
          entry.kind === "local" &&
          localPathIdentity(entry.rootPath) === targetPathId
        ) {
          persistedWorkspacesRef.current.delete(id);
          removed = true;
        }
      }
      if (removed) {
        syncPersistedWorkspaces();
      }
    },
    [syncPersistedWorkspaces],
  );

  const removeWorkspaceSilently = useCallback(
    (workspaceId: string): void => {
      setWorkspaces((prev) => prev.filter((w) => w.id !== workspaceId));
      setOpenFiles((prev) => prev.filter((f) => f.workspaceId !== workspaceId));
      setActiveView((prev) => {
        if (prev?.type === "file" && prev.file.workspaceId === workspaceId) {
          return null;
        }
        return prev;
      });
      setTreeNonces((prev) => {
        if (!(workspaceId in prev)) return prev;
        const next = { ...prev };
        delete next[workspaceId];
        return next;
      });

      const pendingTimer = treeBumpTimersRef.current[workspaceId];
      if (pendingTimer !== undefined) {
        window.clearTimeout(pendingTimer);
        delete treeBumpTimersRef.current[workspaceId];
      }

      if (persistedWorkspacesRef.current.delete(workspaceId)) {
        syncPersistedWorkspaces();
      }
    },
    [syncPersistedWorkspaces],
  );

  /**
  * Look up the persisted blob connection info (account URL + container + prefix) for a
   * workspace. Returns null for non-blob workspaces or when the persisted
  * entry is missing. Used by run/download flows to forward managed-identity
  * blob context to the init-runner so the Python CLI can talk to blob.
   */
  const getBlobInfo = useCallback(
    (
      workspace: Workspace,
    ):
      | { accountUrl: string; containerName: string; prefix: string }
      | null => {
      if (workspace.sourceKind !== "blob") return null;
      const persisted = persistedWorkspacesRef.current.get(workspace.id);
      if (!persisted || persisted.kind !== "blob") return null;
      return {
        accountUrl: persisted.accountUrl,
        containerName: persisted.containerName,
        prefix: persisted.prefix,
      };
    },
    [],
  );

  const addWorkspace = useCallback(
    async (
      name: string,
      source: FileSource,
      options?: {
        rootPath?: string;
        configType?: WorkspaceConfigType;
        hasChildWorkspaces?: boolean;
        parentRootPath?: string;
        persisted?: PersistedWorkspace;
        /** When this workspace is a copy of another local workspace,
         *  the source's rootPath. Surfaces as nesting in the sidebar. */
        copyOfRootPath?: string;
        /** When true, the workspace is hidden from the sidebar and from
         *  workspace dropdowns. Used for transient "open this external
         *  file" cases where the folder is not really part of the user's
         *  benchmark workflow. */
        hiddenFromSidebar?: boolean;
        /** When true, errors are re-thrown instead of being routed to the
         *  global error banner. Used by the Add Workspace dialog so it can
         *  display failures inline and stay open. */
        throwOnError?: boolean;
      },
    ) => {
      try {
        const rootNodes = await source.listChildren();
        const normalizedRootPath = normalizeLocalPath(
          getResolvedLocalRootPath(source) ?? options?.rootPath,
        );
        const normalizedParentRootPath = normalizeLocalPath(
          options?.parentRootPath,
        );
        const normalizedCopyOfRootPath = normalizeLocalPath(
          options?.copyOfRootPath,
        );

        // If the new workspace path is INSIDE an already-loaded local
        // workspace (and not itself an existing workspace, child, copy, or
        // hidden helper), skip creating a duplicate card and refresh the
        // parent workspace so its tree reflects the new content.
        if (
          source.kind === "local" &&
          normalizedRootPath &&
          !normalizedParentRootPath &&
          !normalizedCopyOfRootPath &&
          !options?.hiddenFromSidebar
        ) {
          const newRootId = localPathIdentity(normalizedRootPath);
          const exactMatch = workspacesRef.current.find((w) =>
            hasSameLocalRootPath(w.rootPath, normalizedRootPath),
          );
          if (newRootId && !exactMatch) {
            const ancestor = workspacesRef.current.find((w) => {
              if (w.sourceKind !== "local" || !w.rootPath) return false;
              const ancestorId = localPathIdentity(w.rootPath);
              return !!ancestorId && newRootId.startsWith(`${ancestorId}/`);
            });
            if (ancestor) {
              try {
                const updatedRoot = await ancestor.source.listChildren();
                setWorkspaces((prev) =>
                  prev.map((w) =>
                    w.id === ancestor.id
                      ? {
                          ...w,
                          version: w.version + 1,
                          rootNodes: updatedRoot,
                        }
                      : w,
                  ),
                );
              } catch {
                // Best-effort refresh; ignore failures.
              }
              prunePersistedLocalByRootPath(normalizedRootPath);
              return;
            }
          }
        }

        // Pre-compute the id OUTSIDE the state updater so React Strict Mode's
        // double-invocation of the updater does not register two separate ids
        // in `persistedWorkspacesRef` (the ghost id from the discarded first
        // run would otherwise survive deletion and resurrect on refresh).
        const existingId = normalizedRootPath
          ? workspacesRef.current.find((w) =>
              hasSameLocalRootPath(w.rootPath, normalizedRootPath),
            )
              ?.id
          : undefined;
        const savedId =
          existingId ??
          `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

        setWorkspaces((prev) => {
          const existingIndex = normalizedRootPath
            ? prev.findIndex((w) =>
                hasSameLocalRootPath(w.rootPath, normalizedRootPath),
              )
            : -1;

          if (existingIndex >= 0) {
            const next = [...prev];
            const existing = next[existingIndex];
            next[existingIndex] = {
              ...existing,
              version: existing.version + 1,
              name,
              sourceKind: source.kind,
              configType: options?.configType ?? existing.configType,
              hasChildWorkspaces:
                options?.hasChildWorkspaces ?? existing.hasChildWorkspaces,
              parentRootPath:
                normalizedParentRootPath ?? existing.parentRootPath,
              source,
              rootNodes,
              collapsed: false,
              copyOfRootPath:
                normalizedCopyOfRootPath ?? existing.copyOfRootPath,
            };
            return next;
          }
          return [
            ...prev,
            {
              id: savedId,
              version: 1,
              name,
              sourceKind: source.kind,
              rootPath: normalizedRootPath,
              configType: options?.configType,
              hasChildWorkspaces: options?.hasChildWorkspaces,
              parentRootPath: normalizedParentRootPath,
              source,
              rootNodes,
              collapsed: false,
              copyOfRootPath: normalizedCopyOfRootPath,
              hiddenFromSidebar: options?.hiddenFromSidebar,
            },
          ];
        });

        if (options?.persisted) {
          const persistedEntry =
            options.persisted.kind === "local"
              ? {
                  ...options.persisted,
                  rootPath: normalizedRootPath ?? options.persisted.rootPath,
                  parentRootPath: normalizeLocalPath(
                    options.persisted.parentRootPath,
                  ),
                  copyOfRootPath: normalizeLocalPath(
                    options.persisted.copyOfRootPath,
                  ),
                }
              : options.persisted;
          persistedWorkspacesRef.current.set(savedId, persistedEntry);
          syncPersistedWorkspaces();
        }
      } catch (e) {
        if (
          options?.persisted?.kind === "local" &&
          isMissingPathError(e)
        ) {
          const rootPath =
            normalizeLocalPath(options.persisted.rootPath) ??
            options.persisted.rootPath;
          const existingWorkspaceId = workspacesRef.current.find(
            (w) => hasSameLocalRootPath(w.rootPath, rootPath),
          )?.id;
          if (existingWorkspaceId) {
            removeWorkspaceSilently(existingWorkspaceId);
          }
          prunePersistedLocalByRootPath(rootPath);
          return;
        }
        if (options?.throwOnError) {
          throw e instanceof Error ? e : new Error(String(e));
        }
        setError(`Failed to load ${name}: ${e}`);
      }
    },
    [
      syncPersistedWorkspaces,
      isMissingPathError,
      removeWorkspaceSilently,
      prunePersistedLocalByRootPath,
    ],
  );

  const pickLocalFolderPath = useCallback(async (): Promise<string | null> => {
    setError(null);
    try {
      const res = await fetch(`${INIT_RUNNER_URL}/api/pick-folder`);
      const payload = (await res.json()) as
        | { path?: string; cancelled?: boolean; error?: string };
      if (payload.cancelled) return null;
      if (!res.ok || !payload.path) {
        throw new Error(payload.error ?? "Failed to pick folder.");
      }
      return payload.path;
    } catch (e) {
      setError(
        `Folder picker is unavailable. Start the runner with 'npm run init-runner'. ${String(
          e,
        )}`,
      );
      return null;
    }
  }, []);

  const addLocalWorkspace = useCallback(
    async (data: {
      path: string;
      hasChildWorkspaces: boolean;
      childWorkspacePaths: string[];
    }) => {
      setError(null);
      const { path, hasChildWorkspaces, childWorkspacePaths } = data;
      const name = path.split(/[/\\]/).filter(Boolean).pop() ?? path;
      try {
        await addWorkspace(name, new RunnerPathFileSource(path, INIT_RUNNER_URL), {
          rootPath: path,
          hasChildWorkspaces,
          throwOnError: true,
          persisted: {
            kind: "local",
            name,
            rootPath: path,
            hasChildWorkspaces,
          },
        });

        const isAbsolutePath = (value: string): boolean =>
          /^(?:[a-zA-Z]:[\\/]|\\\\|\/)/.test(value);

        for (const rawChildPath of childWorkspacePaths) {
          const childPath = isAbsolutePath(rawChildPath)
            ? rawChildPath
            : `${path.replace(/[\\/]+$/, "")}/${rawChildPath.replace(/^[\\/]+/, "")}`;
          if (childPath === path) continue;
          const childName = childPath.split(/[/\\]/).filter(Boolean).pop() ?? childPath;
          await addWorkspace(
            childName,
            new RunnerPathFileSource(childPath, INIT_RUNNER_URL),
            {
              rootPath: childPath,
              parentRootPath: path,
              throwOnError: true,
              persisted: {
                kind: "local",
                name: childName,
                rootPath: childPath,
                parentRootPath: path,
              },
            },
          );
        }
        setAddWorkspaceDialogOpen(false);
      } catch (e) {
        // Surface to the dialog so it stays open and shows the error.
        throw new Error(
          `Failed to add local workspace: ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    },
    [addWorkspace],
  );

  // For AddWorkspaceDialog: add blob workspace
  const addBlobWorkspace = useCallback(
    async (data: {
      accountUrl: string;
      containerName: string;
      prefix: string;
      label: string;
    }) => {
      setError(null);
      try {
        await addWorkspace(
          data.label,
          new RunnerBlobFileSource(
            data.accountUrl,
            data.containerName,
            data.prefix,
            INIT_RUNNER_URL,
          ),
          {
            throwOnError: true,
            persisted: {
              kind: "blob",
              label: data.label,
              accountUrl: data.accountUrl,
              containerName: data.containerName,
              prefix: data.prefix,
            },
          },
        );
        setAddWorkspaceDialogOpen(false);
      } catch (e) {
        // Bubble the error up so the dialog can display it inline and the
        // user can correct credentials / paths without re-opening the modal.
        throw new Error(
          `Failed to connect to blob: ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    },
    [addWorkspace],
  );

  // Restore workspaces saved in localStorage on first mount.
  // Run sequentially so the restored order matches the saved order
  // (parallel addWorkspace calls race on async listChildren()).
  const restoredWorkspacesRef = useRef(false);
  useEffect(() => {
    if (restoredWorkspacesRef.current) return;
    restoredWorkspacesRef.current = true;
    const persisted = loadPersistedWorkspaces();
    // Dedupe by identity (rootPath for local, accountUrl+container+prefix for blob). Older
    // builds with the Strict-Mode ref-leak bug could have written ghost
    // duplicates of the same workspace; without this they'd reappear as
    // separate sidebar entries after refresh.
    const seen = new Set<string>();
    const unique = persisted.filter((entry) => {
      const key =
        entry.kind === "local"
          ? `local:${localPathIdentity(entry.rootPath) ?? entry.rootPath}`
          : `blob:${entry.accountUrl}|${entry.containerName}|${entry.prefix}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });

    // Drop persisted local workspaces whose path is inside another
    // persisted local workspace (and that aren't explicit children or
    // copies). Older builds added these as separate top-level cards;
    // we now treat them as part of the parent and remove them on restore.
    const localRootIds = new Map<string, PersistedWorkspace>();
    for (const entry of unique) {
      if (entry.kind !== "local") continue;
      const id = localPathIdentity(entry.rootPath);
      if (id) localRootIds.set(id, entry);
    }
    const restorable = unique.filter((entry) => {
      if (entry.kind !== "local") return true;
      if (entry.parentRootPath || entry.copyOfRootPath) return true;
      const id = localPathIdentity(entry.rootPath);
      if (!id) return true;
      for (const otherId of localRootIds.keys()) {
        if (otherId === id) continue;
        if (id.startsWith(`${otherId}/`)) return false;
      }
      return true;
    });
    if (restorable.length !== persisted.length) {
      // Rewrite localStorage so the cleanup survives the next reload.
      savePersistedWorkspaces(restorable);
    }

    void (async () => {
      const blobEntries = restorable.filter((e) => e.kind === "blob");
      const totalBlobs = blobEntries.length;
      let doneBlobs = 0;
      if (totalBlobs > 0) {
        setBlobRestoreProgress({
          total: totalBlobs,
          done: 0,
          currentLabel: blobEntries[0]?.kind === "blob"
            ? blobEntries[0].label
            : null,
        });
      }
      try {
        for (const entry of restorable) {
          if (entry.kind === "local") {
            await addWorkspace(
              entry.name,
              new RunnerPathFileSource(entry.rootPath, INIT_RUNNER_URL),
              {
                rootPath: entry.rootPath,
                configType: entry.configType,
                hasChildWorkspaces: entry.hasChildWorkspaces,
                parentRootPath: entry.parentRootPath,
                copyOfRootPath: entry.copyOfRootPath,
                persisted: entry,
              },
            );
          } else {
            setBlobRestoreProgress((prev) =>
              prev
                ? { ...prev, currentLabel: entry.label }
                : prev,
            );
            try {
              await addWorkspace(
                entry.label,
                new RunnerBlobFileSource(
                  entry.accountUrl,
                  entry.containerName,
                  entry.prefix,
                  INIT_RUNNER_URL,
                ),
                { persisted: entry },
              );
            } finally {
              doneBlobs += 1;
              setBlobRestoreProgress((prev) =>
                prev ? { ...prev, done: doneBlobs } : prev,
              );
            }
          }
        }
      } finally {
        setBlobRestoreProgress(null);
      }
    })();
  }, [addWorkspace]);

  // Helper functions for opening/closing run job logs
  const handleOpenRunJobLog = useCallback((jobId: string) => {
    setOpenRunJobIds((prev) => new Set(prev).add(jobId));
    setActiveView({ type: "run-job", jobId });
  }, []);

  const handleCloseRunJobLog = useCallback((jobId: string) => {
    setOpenRunJobIds((prev) => {
      const next = new Set(prev);
      next.delete(jobId);
      return next;
    });
    if (activeView?.type === "run-job" && activeView.jobId === jobId) {
      setActiveView(null);
    }
  }, [activeView]);

  const handleOpenInitJobLog = useCallback((jobId: string) => {
    setOpenInitJobIds((prev) => new Set(prev).add(jobId));
    setActiveView({ type: "init-job", jobId });
  }, []);

  const handleCloseInitJobLog = useCallback((jobId: string) => {
    setOpenInitJobIds((prev) => {
      const next = new Set(prev);
      next.delete(jobId);
      return next;
    });
    if (activeView?.type === "init-job" && activeView.jobId === jobId) {
      setActiveView(null);
    }
  }, [activeView]);

  const deleteJob = useCallback(
    async (kind: "init" | "run", jobId: string) => {
      setError(null);
      try {
        const path = kind === "init" ? "init-jobs" : "run-jobs";
        const res = await fetch(`${INIT_RUNNER_URL}/api/${path}/${jobId}`, {
          method: "DELETE",
        });
        if (!res.ok) {
          const payload = (await res
            .json()
            .catch(() => ({}))) as { error?: string };
          throw new Error(payload.error ?? `HTTP ${res.status}`);
        }
        if (kind === "run") {
          setRunJobs((prev) => prev.filter((j) => j.id !== jobId));
          setOpenRunJobIds((prev) => {
            if (!prev.has(jobId)) return prev;
            const next = new Set(prev);
            next.delete(jobId);
            return next;
          });
        } else {
          setInitJobs((prev) => prev.filter((j) => j.id !== jobId));
          setOpenInitJobIds((prev) => {
            if (!prev.has(jobId)) return prev;
            const next = new Set(prev);
            next.delete(jobId);
            return next;
          });
        }
        setActiveView((prev) => {
          if (!prev) return prev;
          if (
            (kind === "run" && prev.type === "run-job" && prev.jobId === jobId) ||
            (kind === "init" && prev.type === "init-job" && prev.jobId === jobId)
          ) {
            return null;
          }
          return prev;
        });
      } catch (e) {
        setError(`Failed to remove job: ${e}`);
      }
    },
    [],
  );

  const submitInitJob = useCallback(async (data: InitRunRequest) => {
    setInitSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${INIT_RUNNER_URL}/api/init-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const payload = (await res.json()) as InitJob | { error: string };
      if (!res.ok) {
        const message = "error" in payload ? payload.error : "Unknown error";
        throw new Error(message);
      }
      setInitJobs((prev) => normalizeJobs([payload as InitJob, ...prev]));
      setInitDialogOpen(false);
      addActivityLog(`Created configuration`, data.configType ?? "unknown", "success");
    } catch (e) {
      setError(
        `Failed to queue init command. Start the runner with 'npm run init-runner'. ${e}`,
      );
      addActivityLog(`Failed to create configuration`, data.configType ?? "unknown", "error");
    } finally {
      setInitSubmitting(false);
    }
  }, [addActivityLog]);

  const handleRunSelectedConfigs = useCallback(
    async (selectedConfigs: DetectedConfig[]) => {
      setMultiConfigDialog((prev) => ({ ...prev, submitting: true }));
      setError(null);
      try {
        for (const config of selectedConfigs) {
          const name = config.name;
          await addWorkspace(
            name,
            new RunnerPathFileSource(config.path, INIT_RUNNER_URL),
            {
              rootPath: config.path,
              configType: config.configType,
              persisted: {
                kind: "local",
                name,
                rootPath: config.path,
                configType: config.configType,
              },
            },
          );
        }
        setMultiConfigDialog({
          open: false,
          folderPath: "",
          configs: [],
          submitting: false,
        });
      } catch (e) {
        setError(`Failed to add workspaces: ${e}`);
        setMultiConfigDialog((prev) => ({ ...prev, submitting: false }));
      }
    },
    [addWorkspace],
  );

  const submitRunJob = useCallback(async (
    workspace: Workspace,
    overrideConfigType?: WorkspaceConfigType,
    blobInfoOverride?: {
      accountUrl: string;
      containerName: string;
      prefix: string;
    },
  ): Promise<boolean> => {
    const configType = overrideConfigType ?? workspace.configType;
    const blobInfo = blobInfoOverride ?? getBlobInfo(workspace);
    if (!configType) {
      setError("This workspace does not have a generated config type to run.");
      return false;
    }
    if (!workspace.rootPath && !blobInfo) {
      setError("This workspace cannot be run.");
      return false;
    }

    const runRoot = blobInfo
      ? `blob://${(() => {
          const container = blobInfo.containerName.trim();
          const prefix = blobInfo.prefix.replace(/^\/+|\/+$/g, "");
          return prefix ? `${container}/${prefix}` : container || "blob";
        })()}`
      : workspace.rootPath;

    setError(null);
    try {
      const requestBody: Record<string, unknown> = { configType };
      if (blobInfo) {
        requestBody.blob = blobInfo;
      } else {
        requestBody.rootPath = workspace.rootPath;
      }
      const res = await fetch(`${INIT_RUNNER_URL}/api/run-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const payload = (await res.json()) as RunJob | { error: string };
      if (!res.ok) {
        const message = "error" in payload ? payload.error : "Unknown error";
        throw new Error(message);
      }
      setWorkspaces((prev) =>
        prev.map((w) =>
          w.id === workspace.id ? { ...w, configType } : w,
        ),
      );
      const job = payload as RunJob;
      setRunJobs((prev) => normalizeJobs([job, ...prev]));
      handleOpenRunJobLog(job.id);
      addActivityLog(
        `Started job`,
        jobActivityLabel(runRoot, configType),
        "success",
      );
      return true;
    } catch (e) {
      setError(
        `Failed to run workspace command. Start the runner with 'npm run init-runner'. ${e}`,
      );
      addActivityLog(
        `Failed to start job`,
        jobActivityLabel(runRoot, configType),
        "error",
      );
      return false;
    }
  }, [handleOpenRunJobLog, addActivityLog, jobActivityLabel, getBlobInfo]);

  const handleRunSubfolderConfig = useCallback(
    async (selectedConfigs: DetectedConfig[]) => {
      setMultiConfigDialog((prev) => ({ ...prev, submitting: true }));
      setError(null);
      try {
        const parent = runSubfolderConfigWorkspaceId
          ? workspaces.find((w) => w.id === runSubfolderConfigWorkspaceId)
          : undefined;
        const parentBlob = parent ? getBlobInfo(parent) : null;

        if (parent && parentBlob) {
          // Blob: each detected config maps to a sub-prefix under the
          // workspace's basePrefix. Build a temp Workspace per config so
          // submitRunJob can pass the right blob payload to the runner.
          for (const cfg of selectedConfigs) {
            const subdir = (cfg.blobSubdir ?? "").replace(/^\/+|\/+$/g, "");
            const subPrefix = subdir
              ? [parentBlob.prefix, subdir].filter(Boolean).join("/")
              : parentBlob.prefix;
            const tempWorkspace: Workspace = {
              id: `temp-blob-${subdir || "root"}`,
              version: 1,
              name: cfg.name,
              sourceKind: "blob",
              configType: cfg.configType,
              source: parent.source,
              rootNodes: [],
              collapsed: false,
            };
            await submitRunJob(tempWorkspace, cfg.configType, {
              accountUrl: parentBlob.accountUrl,
              containerName: parentBlob.containerName,
              prefix: subPrefix,
            });
          }
        } else {
          // Local: each detected config has its own folder path.
          for (const config of selectedConfigs) {
            // Create a temporary workspace object to pass to submitRunJob
            const tempWorkspace: Workspace = {
              id: `temp-${config.path}`,
              version: 1,
              name: config.name,
              sourceKind: "local",
              rootPath: config.path,
              configType: config.configType,
              source: new RunnerPathFileSource(config.path, INIT_RUNNER_URL),
              rootNodes: [],
              collapsed: false,
            };
            await submitRunJob(tempWorkspace);
          }
        }
        setMultiConfigDialog({
          open: false,
          folderPath: "",
          configs: [],
          submitting: false,
        });
        setRunSubfolderConfigWorkspaceId(null);
      } catch (e) {
        setError(`Failed to run configs: ${e}`);
        setMultiConfigDialog((prev) => ({ ...prev, submitting: false }));
      }
    },
    [submitRunJob, runSubfolderConfigWorkspaceId, workspaces, getBlobInfo],
  );

  const submitDatasetDownload = useCallback(
    async (dataset: PredefinedDataset, destinationPath: string) => {
      if (!datasetDialogWorkspaceId) return;
      const workspace = workspaces.find((w) => w.id === datasetDialogWorkspaceId);
      if (!workspace) return;

      const blobInfo = getBlobInfo(workspace);
      if (!workspace.rootPath && !blobInfo) {
        setError("This workspace cannot host downloads.");
        return;
      }

      setDatasetSubmitting(true);
      setError(null);
      try {
        const requestBody: Record<string, unknown> = {
          dataset,
          outputPath: destinationPath,
        };
        if (blobInfo) {
          requestBody.blob = blobInfo;
        } else {
          requestBody.rootPath = workspace.rootPath;
        }
        const res = await fetch(`${INIT_RUNNER_URL}/api/datasets/download`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });
        const payload = (await res.json()) as {
          error?: string;
          outputPath?: string;
          output?: string;
        };
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error(
              "Runner endpoint not found. Restart init-runner to load dataset download support.",
            );
          }
          const details = payload.output ? `\n${payload.output}` : "";
          throw new Error(`${payload.error ?? "Download failed."}${details}`);
        }

        // Reload the workspace tree in place. We can't use `refreshWorkspace`
        // here because it's defined later in the file; `addWorkspace` would
        // duplicate blob workspaces (its dedup keys on rootPath, which blob
        // workspaces don't have).
        try {
          const rootNodes = await workspace.source.listChildren();
          setWorkspaces((prev) =>
            prev.map((w) =>
              w.id === workspace.id
                ? { ...w, version: w.version + 1, rootNodes }
                : w,
            ),
          );
          bumpTreeNonce(workspace.id);
        } catch (e) {
          setError(`Failed to refresh ${workspace.name}: ${e}`);
        }
        setDatasetDialogWorkspaceId(null);
      } catch (e) {
        setError(`Failed to download dataset: ${e}`);
      } finally {
        setDatasetSubmitting(false);
      }
    },
    [datasetDialogWorkspaceId, workspaces, getBlobInfo, bumpTreeNonce],
  );

  const submitLoadDataset = useCallback(
    async (payload: LoadDatasetSubmit) => {
      const { sourceFolder, destinationFolder, workspaceName, subdir } = payload;
      setLoadDatasetSubmitting(true);
      setError(null);
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/datasets/import`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sourcePaths: [sourceFolder],
            destinationFolder,
            subdir,
            flatten: true,
          }),
        });
        const body = (await res.json()) as {
          error?: string;
          imported?: Array<{ source: string; target: string }>;
          skipped?: Array<{ source: string; error: string }>;
          inputFolder?: string;
        };
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error(
              "Runner endpoint /api/datasets/import not found. Restart init-runner.",
            );
          }
          const skippedNote =
            body.skipped && body.skipped.length
              ? `\nSkipped: ${body.skipped
                  .map((s) => `${s.source} (${s.error})`)
                  .join("; ")}`
              : "";
          throw new Error(`${body.error ?? "Import failed."}${skippedNote}`);
        }

        await addWorkspace(
          workspaceName,
          new RunnerPathFileSource(destinationFolder, INIT_RUNNER_URL),
          {
            rootPath: destinationFolder,
            persisted: {
              kind: "local",
              name: workspaceName,
              rootPath: destinationFolder,
            },
          },
        );
        setLoadDatasetDialogOpen(false);

        const skippedSuffix =
          body.skipped && body.skipped.length
            ? ` (${body.skipped.length} skipped)`
            : "";
        setConfigurePrompt({
          workspace: destinationFolder,
          name: workspaceName,
          summary: `Imported ${body.imported?.length ?? 0} entr${(body.imported?.length ?? 0) === 1 ? "y" : "ies"} into ${body.inputFolder}${skippedSuffix}.`,
        });
      } catch (e) {
        setError(`Failed to load dataset: ${e}`);
      } finally {
        setLoadDatasetSubmitting(false);
      }
    },
    [addWorkspace],
  );

  const handleRunWorkspace = useCallback(
    async (workspace: Workspace) => {
      const blobInfo = getBlobInfo(workspace);
      if (!workspace.rootPath && !blobInfo) return;
      if (detectingConfigsWorkspaceIds.has(workspace.id)) return;

      // For blob workspaces, mirror the local detect-configs flow by
      // scanning the workspace's own file source for settings.yaml files
      // (root + one level deep). The runner's /api/detect-configs only
      // walks the local filesystem, so we do this client-side using the
      // already-connected runner-backed blob source.
      if (blobInfo) {
        setError(null);
        beginDetectingConfigs(workspace.id);
        try {
          const detected = await detectBlobConfigs(workspace);
          if (detected.length > 0) {
            setMultiConfigDialog({
              open: true,
              folderPath: workspace.name,
              configs: detected,
              submitting: false,
            });
            setRunSubfolderConfigWorkspaceId(workspace.id);
            return;
          }
        } catch (e) {
          setError(`Failed to detect configs in blob workspace: ${e}`);
        } finally {
          endDetectingConfigs(workspace.id);
        }
        // Fall back to the config-type picker when nothing was detected.
        setRunConfigWorkspaceId(workspace.id);
        setRunConfigDialogOpen(true);
        return;
      }

      // Below: local workspace path (narrowed by the early returns above).
      if (!workspace.rootPath) return;
      const localRootPath = workspace.rootPath;

      // Always prompt for config selection so the user can pick which job to run.
      setError(null);
      beginDetectingConfigs(workspace.id);
      try {
        const detectRes = await fetch(`${INIT_RUNNER_URL}/api/detect-configs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rootPath: localRootPath }),
        });
        const detectPayload = (await detectRes.json()) as {
          configs?: DetectedConfig[];
          error?: string;
        };

        if (detectRes.ok && detectPayload.configs && detectPayload.configs.length > 0) {
          // Show selection dialog so the user can choose the command
          setMultiConfigDialog({
            open: true,
            folderPath: localRootPath,
            configs: detectPayload.configs,
            submitting: false,
          });
          setRunSubfolderConfigWorkspaceId(workspace.id);
          return;
        }

        // No subfolders with configs - show config type dialog
        setRunConfigWorkspaceId(workspace.id);
        setRunConfigDialogOpen(true);
      } catch (e) {
        setError(`Failed to detect configs: ${e}`);
      } finally {
        endDetectingConfigs(workspace.id);
      }
    },
    [
      getBlobInfo,
      detectingConfigsWorkspaceIds,
      beginDetectingConfigs,
      endDetectingConfigs,
    ],
  );

  const handleRunConfigSubmit = useCallback(
    async (configType: WorkspaceConfigType) => {
      if (!runConfigWorkspaceId) return;
      const workspace = workspaces.find((w) => w.id === runConfigWorkspaceId);
      if (!workspace) return;

      const succeeded = await submitRunJob(workspace, configType);
      if (succeeded) {
        setRunConfigDialogOpen(false);
        setRunConfigWorkspaceId(null);
      }
    },
    [runConfigWorkspaceId, submitRunJob, workspaces],
  );

  const cancelRunJob = useCallback(async (jobId: string) => {
    setError(null);
    try {
      const res = await fetch(`${INIT_RUNNER_URL}/api/run-jobs/${jobId}/cancel`, {
        method: "POST",
      });
      const payload = (await res.json()) as { ok?: boolean; error?: string };
      if (!res.ok) {
        throw new Error(payload.error ?? "Failed to cancel job.");
      }
      setRunJobs((prev) =>
        prev.map((job) =>
          job.id === jobId
            ? { ...job, status: "cancelled", endedAt: new Date().toISOString(), exitCode: -1 }
            : job,
        ),
      );
    } catch (e) {
      setError(`Failed to cancel job: ${e}`);
    }
  }, []);

  const rerunJob = useCallback(async (jobId: string) => {
    const existing = runJobs.find((j) => j.id === jobId);
    if (!existing || !existing.rootPath || !existing.configType) {
      setError("Cannot re-run job: missing workspace path or config type.");
      return;
    }
    if (existing.status === "running") return;

    setError(null);
    try {
      const res = await fetch(
        `${INIT_RUNNER_URL}/api/run-jobs/${jobId}/rerun`,
        { method: "POST" },
      );
      const payload = (await res.json()) as RunJob | { error: string };
      if (!res.ok) {
        const message =
          "error" in payload ? payload.error : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const updated = payload as RunJob;
      setRunJobs((prev) =>
        prev.map((j) => (j.id === jobId ? updated : j)),
      );
      // Reset our "last-logged status" so the running → finished transition
      // emits a fresh activity-log entry for this re-run.
      loggedRunJobStatusRef.current.delete(jobId);
      handleOpenRunJobLog(jobId);
      addActivityLog(
        `Re-started job`,
        jobActivityLabel(existing.rootPath, existing.configType),
        "success",
      );
    } catch (e) {
      setError(`Failed to re-run job: ${e}`);
      addActivityLog(`Failed to re-run job`, String(e), "error");
    }
  }, [runJobs, handleOpenRunJobLog, addActivityLog, jobActivityLabel]);

  useEffect(() => {
    let dirty = false;
    for (const job of initJobs) {
      if (job.status !== "succeeded") continue;
      if (!job.rootPath) continue;
      // Blob init jobs write to Azure, not to a usable local folder
      // (rootPath is typically ".", the runner's cwd). Don't auto-add them
      // as LOCAL sidebar workspaces; instead surface the blob config as a new
      // blob workspace. Fall back to inspecting the command string for older
      // jobs / older runners that don't set storageType.
      const isBlobJob =
        job.storageType === "blob" ||
        /--storage-type\s+blob\b/.test(job.command ?? "");
      if (isBlobJob) {
        if (importedInitJobIdsRef.current.has(job.id)) continue;
        importedInitJobIdsRef.current.add(job.id);
        dirty = true;

        // Reconstruct the blob location from the command. Building a blob
        // source needs account-url + container; jobs authenticated via a
        // connection-string only can't be rebuilt here, so we just mark them
        // handled and move on.
        const cmd = job.command ?? "";
        const matchArg = (flag: string): string | undefined => {
          const re = new RegExp(`${flag}\\s+("([^"]*)"|(\\S+))`);
          const m = cmd.match(re);
          return m ? (m[2] ?? m[3]) : undefined;
        };
        const jobContainer = matchArg("--container-name");
        const jobAccountUrl = matchArg("--account-url");
        if (!jobContainer || !jobAccountUrl) continue;
        const jobBaseDir = (matchArg("--base-dir") ?? "").replace(
          /^[/\\]+|[/\\]+$/g,
          "",
        );
        // In blob mode the positional root argument is unused (defaults to
        // "."), so the config lives directly under <base-dir>.
        const jobRoot = (job.rootPath ?? "").replace(/^[/\\]+|[/\\]+$/g, "");
        const configRoot = jobRoot === "." ? "" : jobRoot;
        const newPrefix = [jobBaseDir, configRoot].filter(Boolean).join("/");
        const newPathLc = newPrefix.toLowerCase();
        const accountId = jobAccountUrl.replace(/\/+$/, "");

        // If an open blob workspace already covers this config (exact prefix,
        // an ancestor prefix, or the container root), skip creating a card —
        // the blob-refresh effect re-lists it so the new folder shows nested.
        const coveredByExisting = workspacesRef.current.some((ws) => {
          if (ws.sourceKind !== "blob") return false;
          const info = persistedWorkspacesRef.current.get(ws.id);
          if (!info || info.kind !== "blob") return false;
          if (info.containerName !== jobContainer) return false;
          if (info.accountUrl.replace(/\/+$/, "") !== accountId) return false;
          const wsPrefix = (info.prefix ?? "")
            .replace(/^[/\\]+|[/\\]+$/g, "")
            .toLowerCase();
          return (
            !wsPrefix ||
            wsPrefix === newPathLc ||
            newPathLc.startsWith(`${wsPrefix}/`)
          );
        });
        if (coveredByExisting) continue;

        const label =
          newPrefix.split("/").filter(Boolean).pop() ?? jobContainer;
        void addWorkspace(
          label,
          new RunnerBlobFileSource(
            jobAccountUrl,
            jobContainer,
            newPrefix,
            INIT_RUNNER_URL,
          ),
          {
            configType: job.configType,
            persisted: {
              kind: "blob",
              label,
              accountUrl: jobAccountUrl,
              containerName: jobContainer,
              prefix: newPrefix,
            },
          },
        );
        continue;
      }
      if (importedInitJobIdsRef.current.has(job.id)) continue;

      importedInitJobIdsRef.current.add(job.id);
      dirty = true;
      const name =
        job.rootPath.split(/[/\\]/).filter(Boolean).pop() ?? job.rootPath;
      void addWorkspace(
        name,
        new RunnerPathFileSource(job.rootPath, INIT_RUNNER_URL),
        {
          rootPath: job.rootPath,
          configType: job.configType,
          persisted: {
            kind: "local",
            name,
            rootPath: job.rootPath,
            configType: job.configType,
          },
        },
      );
    }
    if (dirty) saveImportedInitJobIds(importedInitJobIdsRef.current);
  }, [addWorkspace, initJobs]);

  const closeWorkspace = useCallback(
    (id: string) => {
      const closed = workspacesRef.current.find((w) => w.id === id);
      setWorkspaces((prev) => prev.filter((w) => w.id !== id));
      if (activeView?.type === "file" && activeView.file.workspaceId === id) {
        setActiveView(null);
      }
      setOpenFiles((prev) => prev.filter((f) => f.workspaceId !== id));
      if (persistedWorkspacesRef.current.delete(id)) {
        syncPersistedWorkspaces();
      }
      if (closed?.sourceKind === "local" && closed.rootPath) {
        // Catch any orphan persisted entries that share the same rootPath
        // but were keyed under a different (ghost) id in the Map. Without
        // this they'd reappear on reload as a duplicate top-level card.
        prunePersistedLocalByRootPath(closed.rootPath);
        // Mark any init job that produced this workspace as dismissed so
        // the auto-import effect doesn't resurrect it on the next reload.
        const targetId = localPathIdentity(closed.rootPath);
        if (targetId) {
          let dirty = false;
          for (const job of initJobs) {
            if (!job.rootPath) continue;
            if (localPathIdentity(job.rootPath) !== targetId) continue;
            if (importedInitJobIdsRef.current.has(job.id)) continue;
            importedInitJobIdsRef.current.add(job.id);
            dirty = true;
          }
          if (dirty) saveImportedInitJobIds(importedInitJobIdsRef.current);
        }
      }
    },
    [activeView, syncPersistedWorkspaces, prunePersistedLocalByRootPath, initJobs],
  );

  const toggleWorkspace = useCallback((id: string) => {
    setWorkspaces((prev) =>
      prev.map((w) =>
        w.id === id ? { ...w, collapsed: !w.collapsed } : w,
      ),
    );
  }, []);

  const reorderWorkspaces = useCallback(
    (fromId: string, toId: string) => {
      if (fromId === toId) return;
      setWorkspaces((prev) => {
        const fromIndex = prev.findIndex((w) => w.id === fromId);
        const toIndex = prev.findIndex((w) => w.id === toId);
        if (fromIndex < 0 || toIndex < 0) return prev;
        const next = prev.slice();
        const [moved] = next.splice(fromIndex, 1);
        next.splice(toIndex, 0, moved);
        // Rebuild the persisted Map in the new order so saved JSON matches.
        const map = persistedWorkspacesRef.current;
        const reordered = new Map<string, PersistedWorkspace>();
        for (const w of next) {
          const entry = map.get(w.id);
          if (entry) reordered.set(w.id, entry);
        }
        // Preserve any orphan entries (shouldn't happen, but safe).
        for (const [k, v] of map) {
          if (!reordered.has(k)) reordered.set(k, v);
        }
        persistedWorkspacesRef.current = reordered;
        queueMicrotask(syncPersistedWorkspaces);
        return next;
      });
    },
    [syncPersistedWorkspaces],
  );

  const [draggingWorkspaceId, setDraggingWorkspaceId] = useState<string | null>(null);
  const [dragOverWorkspaceId, setDragOverWorkspaceId] = useState<string | null>(null);
  const workspaceRefreshInFlightRef = useRef<Set<string>>(new Set());

  const refreshWorkspace = useCallback(
    async (workspace: Workspace): Promise<boolean> => {
      if (workspaceRefreshInFlightRef.current.has(workspace.id)) {
        return false;
      }
      workspaceRefreshInFlightRef.current.add(workspace.id);
      // Re-list children in place instead of going through addWorkspace.
      // addWorkspace dedups on rootPath, which blob workspaces don't have,
      // so calling it for a refresh would append a duplicate entry.
      try {
        const rootNodes = await workspace.source.listChildren();
        setWorkspaces((prev) =>
          prev.map((w) =>
            w.id === workspace.id
              ? { ...w, version: w.version + 1, rootNodes }
              : w,
          ),
        );
        return true;
      } catch (e) {
        if (workspace.sourceKind === "local" && isMissingPathError(e)) {
          removeWorkspaceSilently(workspace.id);
          if (workspace.rootPath) {
            prunePersistedLocalByRootPath(workspace.rootPath);
          }
          return false;
        }
        setError(`Failed to refresh ${workspace.name}: ${e}`);
        return false;
      } finally {
        workspaceRefreshInFlightRef.current.delete(workspace.id);
      }
    },
    [
      isMissingPathError,
      removeWorkspaceSilently,
      prunePersistedLocalByRootPath,
    ],
  );

  // Keep the sidebar in sync when folders are deleted outside the UI. This
  // runs even while idle so stale local entries disappear without user action.
  useEffect(() => {
    if (runJobs.some((j) => j.status === "running")) return;
    const localWorkspaces = workspaces.filter(
      (w) => w.sourceKind === "local" && !!w.rootPath,
    );
    if (localWorkspaces.length === 0) return;

    let cancelled = false;
    let running = false;

    const sweep = async () => {
      if (running) return;
      running = true;
      try {
        for (const ws of localWorkspaces) {
          if (cancelled) return;
          try {
            await ws.source.listChildren();
          } catch (e) {
            if (cancelled) return;
            if (isMissingPathError(e)) {
              removeWorkspaceSilently(ws.id);
              if (ws.rootPath) {
                prunePersistedLocalByRootPath(ws.rootPath);
              }
            }
          }
        }
      } finally {
        running = false;
      }
    };

    void sweep();
    const timerId = window.setInterval(() => {
      void sweep();
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timerId);
    };
  }, [
    workspaces,
    runJobs,
    isMissingPathError,
    removeWorkspaceSilently,
    prunePersistedLocalByRootPath,
  ]);

  // When a blob init job succeeds, refresh any open blob workspace whose
  // container/account match and whose prefix is an ancestor of the new
  // config folder, so the freshly created folder shows up in the sidebar
  // tree without requiring the user to remove and re-add the workspace.
  const blobInitRefreshedJobIdsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    for (const job of initJobs) {
      if (job.status !== "succeeded") continue;
      if (!job.rootPath) continue;
      const isBlobJob =
        job.storageType === "blob" ||
        /--storage-type\s+blob\b/.test(job.command ?? "");
      if (!isBlobJob) continue;
      if (blobInitRefreshedJobIdsRef.current.has(job.id)) continue;
      blobInitRefreshedJobIdsRef.current.add(job.id);

      const cmd = job.command ?? "";
      const matchArg = (flag: string): string | undefined => {
        const re = new RegExp(`${flag}\\s+("([^"]*)"|(\\S+))`);
        const m = cmd.match(re);
        return m ? (m[2] ?? m[3]) : undefined;
      };
      const jobContainer = matchArg("--container-name");
      const jobAccountUrl = matchArg("--account-url");
      const jobBaseDir = (matchArg("--base-dir") ?? "").replace(
        /^[/\\]+|[/\\]+$/g,
        "",
      );
      // In blob mode the positional root argument is unused (it defaults to
      // "."), so the config is uploaded directly under <base-dir>. Treat the
      // base-dir as the config's container prefix and ignore a "." root.
      const jobRoot = (job.rootPath ?? "").replace(/^[/\\]+|[/\\]+$/g, "");
      const configRoot = jobRoot === "." ? "" : jobRoot;
      const newPath = [jobBaseDir, configRoot]
        .filter(Boolean)
        .join("/")
        .toLowerCase();

      for (const ws of workspacesRef.current) {
        if (ws.sourceKind !== "blob") continue;
        const info = persistedWorkspacesRef.current.get(ws.id);
        if (!info || info.kind !== "blob") continue;
        if (jobContainer && info.containerName !== jobContainer) continue;
        if (
          jobAccountUrl &&
          info.accountUrl.replace(/\/+$/, "") !==
            jobAccountUrl.replace(/\/+$/, "")
        ) {
          continue;
        }
        const wsPrefix = (info.prefix ?? "")
          .replace(/^[/\\]+|[/\\]+$/g, "")
          .toLowerCase();
        if (
          wsPrefix &&
          newPath &&
          !newPath.startsWith(`${wsPrefix}/`) &&
          newPath !== wsPrefix
        ) {
          continue;
        }
        void refreshWorkspace(ws).then((ok) => {
          if (ok) bumpTreeNonce(ws.id);
        });
      }
    }
  }, [initJobs, refreshWorkspace, bumpTreeNonce]);

  const requestCreatePath = useCallback(
    (workspace: Workspace, kind: "file" | "directory", parentNode?: TreeNode) => {
      setCreateDialog({
        open: true,
        workspaceId: workspace.id,
        kind,
        parentNode,
      });
    },
    [],
  );

  const submitCreatePath = useCallback(
    async (name: string) => {
      if (!createDialog.workspaceId) return;
      const workspace = workspaces.find((w) => w.id === createDialog.workspaceId);
      if (!workspace) return;

      const kind = createDialog.kind;
      const parentNode = createDialog.parentNode;

      if (!workspace.source.canWrite()) {
        setError("This workspace is read-only.");
        return;
      }

      if (kind === "file" && !workspace.source.createFile) {
        setError("This workspace source does not support creating files.");
        return;
      }
      if (kind === "directory" && !workspace.source.createDirectory) {
        setError("This workspace source does not support creating folders.");
        return;
      }
      if (kind !== "file" && kind !== "directory") {
        setError("This workspace source does not support creating paths.");
        return;
      }

      const parentPath = parentNode?.path ?? "";
      const trimmedName = name.trim().replace(/^[/\\]+|[/\\]+$/g, "");
      if (!trimmedName) {
        setError("Name is required.");
        return;
      }

      const relPath = parentPath ? `${parentPath}/${trimmedName}` : trimmedName;
      setError(null);
      setCreateSubmitting(true);
      try {
        if (kind === "file") {
          await workspace.source.createFile!(relPath, "");
        } else {
          await workspace.source.createDirectory!(relPath);
        }
        const refreshed = await refreshWorkspace(workspace);
        if (refreshed) {
          bumpTreeNonce(workspace.id);
        }
        setCreateDialog({ open: false, workspaceId: null, kind: "file" });
      } catch (e) {
        setError(`Failed to create ${kind}: ${e}`);
      } finally {
        setCreateSubmitting(false);
      }
    },
    [createDialog, refreshWorkspace, workspaces, bumpTreeNonce],
  );

  const requestDeletePath = useCallback((workspace: Workspace, node: TreeNode) => {
    setDeleteDialog({
      open: true,
      workspaceId: workspace.id,
      node,
    });
  }, []);

  const requestCopyNode = useCallback(
    (workspace: Workspace, node: TreeNode) => {
      if (node.kind !== "directory") {
        setError("Only folders can be copied across workspaces.");
        return;
      }
      setCopyNodeDialog({
        open: true,
        workspaceId: workspace.id,
        node,
      });
    },
    [],
  );

  const submitCopyNode = useCallback(
    async ({
      destWorkspaceId,
      destRelPath,
      newName,
      overwrite,
    }: {
      destWorkspaceId: string;
      destRelPath: string;
      newName: string;
      overwrite: boolean;
    }) => {
      const sourceWorkspace = workspaces.find(
        (w) => w.id === copyNodeDialog.workspaceId,
      );
      const destWorkspace = workspaces.find((w) => w.id === destWorkspaceId);
      const node = copyNodeDialog.node;
      if (!sourceWorkspace?.rootPath || !destWorkspace?.rootPath || !node) {
        setError("Source and destination must be local workspaces.");
        return;
      }
      const srcRoot = sourceWorkspace.rootPath.replace(/[/\\]+$/, "");
      const dstRoot = destWorkspace.rootPath.replace(/[/\\]+$/, "");
      const srcRel = node.path.replace(/^[/\\]+/, "");
      const sourcePath = `${srcRoot}/${srcRel}`;
      const trimmedRel = destRelPath.replace(/^[/\\]+|[/\\]+$/g, "");
      const destPath = trimmedRel
        ? `${dstRoot}/${trimmedRel}/${newName}`
        : `${dstRoot}/${newName}`;
      setError(null);
      setCopyNodeSubmitting(true);
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/fs/copy-path`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sourcePath, destPath, overwrite }),
        });
        if (!res.ok) {
          const payload = (await res.json().catch(() => ({}))) as {
            error?: string;
          };
          throw new Error(payload.error ?? `Copy failed (HTTP ${res.status}).`);
        }
        const refreshed = await refreshWorkspace(destWorkspace);
        if (refreshed) {
          bumpTreeNonce(destWorkspace.id);
        }
        setCopyNodeDialog({ open: false, workspaceId: null });
      } catch (e) {
        setError(`Failed to copy folder: ${e}`);
      } finally {
        setCopyNodeSubmitting(false);
      }
    },
    [copyNodeDialog, workspaces, refreshWorkspace, bumpTreeNonce],
  );

  const requestRenameNode = useCallback(
    (workspace: Workspace, node: TreeNode) => {
      setRenameNodeDialog({
        open: true,
        workspaceId: workspace.id,
        node,
      });
    },
    [],
  );

  const submitRenameNode = useCallback(
    async (newName: string) => {
      const workspace = workspaces.find(
        (w) => w.id === renameNodeDialog.workspaceId,
      );
      const node = renameNodeDialog.node;
      if (!workspace || !node) return;
      if (!workspace.source.canWrite() || !workspace.source.renamePath) {
        setError("This workspace does not support renaming.");
        return;
      }
      setError(null);
      setRenameNodeSubmitting(true);
      try {
        const newRelPath = await workspace.source.renamePath(
          node.path,
          newName,
        );
        // Re-point any open files / active view that were under the renamed
        // node so users don't end up with a stale path after rename.
        const oldPath = node.path.replace(/\\/g, "/");
        const newPath = newRelPath.replace(/\\/g, "/");
        const rewriteNode = (n: TreeNode): TreeNode => {
          if (n.path === oldPath) {
            return { ...n, path: newPath, name: newName };
          }
          if (n.path.startsWith(`${oldPath}/`)) {
            return {
              ...n,
              path: `${newPath}${n.path.slice(oldPath.length)}`,
            };
          }
          return n;
        };
        setOpenFiles((prev) =>
          prev.map((f) =>
            f.workspaceId === workspace.id
              ? { ...f, node: rewriteNode(f.node) }
              : f,
          ),
        );
        setActiveView((prev) => {
          if (
            prev?.type === "file" &&
            prev.file.workspaceId === workspace.id
          ) {
            return {
              ...prev,
              file: { ...prev.file, node: rewriteNode(prev.file.node) },
            };
          }
          return prev;
        });
        const refreshed = await refreshWorkspace(workspace);
        if (refreshed) {
          bumpTreeNonce(workspace.id);
        }
        setRenameNodeDialog({ open: false, workspaceId: null });
      } catch (e) {
        setError(`Failed to rename: ${e}`);
      } finally {
        setRenameNodeSubmitting(false);
      }
    },
    [renameNodeDialog, workspaces, refreshWorkspace, bumpTreeNonce],
  );

  const confirmDeletePath = useCallback(async () => {
    if (!deleteDialog.workspaceId || !deleteDialog.node) return;
    const workspace = workspaces.find((w) => w.id === deleteDialog.workspaceId);
    if (!workspace) return;
    const node = deleteDialog.node;

      if (!workspace.source.canWrite()) {
        setError("This workspace is read-only.");
        return;
      }
      if (!workspace.source.deletePath) {
        setError("This workspace source does not support deleting paths.");
        return;
      }

      setError(null);
      setDeleteSubmitting(true);
      try {
        await workspace.source.deletePath(node.path);
        setActiveView((prev) => {
          if (prev?.type !== "file") return prev;
          if (prev.file.workspaceId !== workspace.id) return prev;
          if (
            prev.file.node.path === node.path ||
            prev.file.node.path.startsWith(`${node.path}/`)
          ) {
            return null;
          }
          return prev;
        });
        setOpenFiles((prev) =>
          prev.filter(
            (f) =>
              f.workspaceId !== workspace.id ||
              (f.node.path !== node.path && !f.node.path.startsWith(`${node.path}/`)),
          ),
        );
        const refreshed = await refreshWorkspace(workspace);
        if (refreshed) {
          bumpTreeNonce(workspace.id);
        }
        setDeleteDialog({ open: false, workspaceId: null, node: undefined });
      } catch (e) {
        setError(`Failed to delete path: ${e}`);
      } finally {
        setDeleteSubmitting(false);
      }
    },
    [deleteDialog, refreshWorkspace, workspaces, bumpTreeNonce],
  );

  const requestCopyWorkspace = useCallback((workspace: Workspace) => {
    if (!workspace.rootPath) {
      setError("Only local workspaces can be copied.");
      return;
    }
    if (workspace.copyOfRootPath) {
      setError("This workspace is already a copy and cannot be copied again.");
      return;
    }
    setCopyDialog({ open: true, workspaceId: workspace.id });
  }, []);

  const requestRenameWorkspace = useCallback((workspace: Workspace) => {
    setRenameDialog({ open: true, workspaceId: workspace.id });
  }, []);

  const submitRenameWorkspace = useCallback(
    (newName: string) => {
      const id = renameDialog.workspaceId;
      if (!id) return;
      const trimmed = newName.trim();
      if (!trimmed) {
        setRenameDialog({ open: false, workspaceId: null });
        return;
      }
      setWorkspaces((prev) =>
        prev.map((w) => (w.id === id ? { ...w, name: trimmed } : w)),
      );
      const persisted = persistedWorkspacesRef.current.get(id);
      if (persisted) {
        const updated: PersistedWorkspace =
          persisted.kind === "local"
            ? { ...persisted, name: trimmed }
            : { ...persisted, label: trimmed };
        persistedWorkspacesRef.current.set(id, updated);
        syncPersistedWorkspaces();
      }
      setRenameDialog({ open: false, workspaceId: null });
    },
    [renameDialog, syncPersistedWorkspaces],
  );

  const submitCopyWorkspace = useCallback(
    async ({
      destPath,
      workspaceName,
      overwrite,
    }: {
      destPath: string;
      workspaceName: string;
      overwrite: boolean;
    }) => {
      if (!copyDialog.workspaceId) return;
      const workspace = workspaces.find((w) => w.id === copyDialog.workspaceId);
      if (!workspace || !workspace.rootPath) return;
      setError(null);
      setCopySubmitting(true);
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/fs/copy-path`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sourcePath: workspace.rootPath,
            destPath,
            overwrite,
          }),
        });
        if (!res.ok) {
          const payload = (await res.json().catch(() => ({}))) as {
            error?: string;
          };
          throw new Error(payload.error ?? `Copy failed (HTTP ${res.status}).`);
        }
        const payload = (await res.json()) as { path?: string };
        const finalPath = payload.path ?? destPath;
        await addWorkspace(
          workspaceName,
          new RunnerPathFileSource(finalPath, INIT_RUNNER_URL),
          {
            rootPath: finalPath,
            configType: workspace.configType,
            copyOfRootPath: workspace.rootPath,
            persisted: {
              kind: "local",
              name: workspaceName,
              rootPath: finalPath,
              configType: workspace.configType,
              copyOfRootPath: workspace.rootPath,
            },
          },
        );
        // Place the new copy directly under its source in the workspace list
        // (it would otherwise be appended at the end).
        setWorkspaces((prev) => {
          const sourceIndex = prev.findIndex((w) => w.id === workspace.id);
          const copyIndex = prev.findIndex((w) => w.rootPath === finalPath);
          if (
            sourceIndex < 0 ||
            copyIndex < 0 ||
            copyIndex === sourceIndex + 1
          ) {
            return prev;
          }
          const next = prev.slice();
          const [moved] = next.splice(copyIndex, 1);
          const insertAt = copyIndex < sourceIndex ? sourceIndex : sourceIndex + 1;
          next.splice(insertAt, 0, moved);
          // Keep persisted order in sync with the new visual order.
          const map = persistedWorkspacesRef.current;
          const reordered = new Map<string, PersistedWorkspace>();
          for (const w of next) {
            const entry = map.get(w.id);
            if (entry) reordered.set(w.id, entry);
          }
          for (const [k, v] of map) {
            if (!reordered.has(k)) reordered.set(k, v);
          }
          persistedWorkspacesRef.current = reordered;
          queueMicrotask(syncPersistedWorkspaces);
          return next;
        });
        setCopyDialog({ open: false, workspaceId: null });
      } catch (e) {
        // Surface the error inline in the dialog so the user sees it next to
        // the Overwrite checkbox; do not blow it up into the global banner.
        setCopySubmitting(false);
        throw e instanceof Error ? e : new Error(String(e));
      } finally {
        setCopySubmitting(false);
      }
    },
    [copyDialog, workspaces, addWorkspace, syncPersistedWorkspaces],
  );

  const handleSelectFile = useCallback(
    async (workspace: Workspace, node: TreeNode) => {
      setError(null);
      const key = `${workspace.id}::${node.path}`;
      // If the file is already open, just activate it (preserve unsaved edits).
      const existing = openFiles.find((f) => fileKey(f) === key);
      if (existing) {
        setActiveView({ type: "file", file: existing });
        return;
      }
      const kind = detectKind(node.name);
      const language = detectLanguage(node.name);
      try {
        const content =
          kind === "unsupported" ? "" : await workspace.source.readFile(node);
        const file: OpenFile = {
          workspaceId: workspace.id,
          node,
          kind,
          language,
          content,
          dirty: false,
        };
        setActiveView({ type: "file", file });
      } catch (e) {
        setError(`Failed to open ${node.name}: ${e}`);
      }
    },
    [openFiles],
  );

  const handleContentChange = useCallback((value: string) => {
    setActiveView((prev) =>
      prev?.type === "file" ? { ...prev, file: { ...prev.file, content: value, dirty: true } } : prev,
    );
  }, []);

  /**
   * Open a previously generated quality report in the editor area. The
   * report file lives at an absolute path on disk; we find (or auto-add)
   * a local workspace whose root contains it, then route through the
   * normal file-select handler.
   */
  const openRecentReport = useCallback(
    async (report: RecentReport) => {
      const reportPath = report.reportPath;
      const dir = report.destinationPath;
      const matches = (w: Workspace): boolean =>
        !!w.rootPath &&
        (reportPath === w.rootPath ||
          reportPath.startsWith(w.rootPath.replace(/[\\/]+$/, "") + "/") ||
          reportPath.startsWith(w.rootPath.replace(/[\\/]+$/, "") + "\\"));

      let ws = workspacesRef.current.find(matches);
      if (!ws) {
        // No real workspace contains the report. Attach a HIDDEN
        // transient workspace anchored at the report's folder so the
        // file can be opened in a tab. It won't show in the sidebar
        // tree or in workspace dropdowns, and it isn't persisted —
        // re-opening from "Recent reports" re-creates it on demand.
        const name = dir.split(/[\\/]/).filter(Boolean).pop() ?? dir;
        await addWorkspace(
          name,
          new RunnerPathFileSource(dir, INIT_RUNNER_URL),
          { rootPath: dir, hiddenFromSidebar: true },
        );
        for (let i = 0; i < 20 && !ws; i++) {
          await new Promise((r) => setTimeout(r, 50));
          ws = workspacesRef.current.find(matches);
        }
      }
      if (!ws) {
        setError(
          `Could not open ${reportPath}. The folder ${dir} could not be loaded.`,
        );
        return;
      }
      const wsRoot = ws.rootPath!.replace(/[\\/]+$/, "");
      const relPath = reportPath.slice(wsRoot.length).replace(/^[\\/]+/, "");
      const name = relPath.split(/[\\/]/).pop() ?? "report.md";
      const node: TreeNode = {
        name,
        path: relPath,
        kind: "file",
      };
      await handleSelectFile(ws, node);
    },
    // handleSelectFile is defined below; including it keeps deps fresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [addWorkspace],
  );

  /**
   * Called whenever the AI Assistant goes idle. If a question-quality
   * report is pending, probe the destination file via the runner FS API
   * and, when it appears, register it as a recent report and open it in
   * a tab. Returns true if it finalized so the caller can stop polling.
   */
  const finalizePendingReport = useCallback(async (): Promise<boolean> => {
    const pending = pendingReportRef.current;
    if (!pending) return false;
    try {
      // Split the report path into root folder + filename. The runner's
      // /api/fs/read endpoint requires a non-empty `path`, so we cannot
      // probe by passing the full file path as `root` with an empty path.
      // Handle both POSIX (/) and Windows (\) separators since the runner
      // returns native paths.
      const sep = Math.max(
        pending.reportPath.lastIndexOf("/"),
        pending.reportPath.lastIndexOf("\\"),
      );
      if (sep < 0) return false;
      const rootDir = pending.reportPath.slice(0, sep);
      const fileName = pending.reportPath.slice(sep + 1);
      if (!rootDir || !fileName) return false;
      const u =
        `${INIT_RUNNER_URL}/api/fs/read` +
        `?root=${encodeURIComponent(rootDir)}` +
        `&path=${encodeURIComponent(fileName)}`;
      const res = await fetch(u);
      if (!res.ok) return false;
    } catch {
      return false;
    }
    pendingReportRef.current = null;
    const record = addRecentReport({
      reportPath: pending.reportPath,
      destinationPath: pending.destinationPath,
      label: pending.label,
      setLabels: pending.setLabels,
    });
    void openRecentReport(record);
    return true;
  }, [openRecentReport]);

  const handleSave = useCallback(async () => {
    if (activeView?.type !== "file") return;
    const file = activeView.file;
    const ws = workspaces.find((w) => w.id === file.workspaceId);
    if (!ws) return;
    if (!ws.source.canWrite()) {
      setError("This workspace is read-only.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      await ws.source.writeFile(file.node, file.content);
      setActiveView({ type: "file", file: { ...file, dirty: false } });
    } catch (e) {
      setError(`Failed to save: ${e}`);
    } finally {
      setSaving(false);
    }
  }, [activeView, workspaces]);

  useEffect(() => {
    if (activeView?.type !== "file" || !activeView.file.dirty) return;
    const file = activeView.file;
    const ws = workspaces.find((w) => w.id === file.workspaceId);
    if (!ws || !ws.source.canWrite()) return;

    const snapshot = {
      workspaceId: file.workspaceId,
      path: file.node.path,
      content: file.content,
      node: file.node,
    };

    const timerId = window.setTimeout(async () => {
      setSaving(true);
      setError(null);
      try {
        await ws.source.writeFile(snapshot.node, snapshot.content);
        setActiveView((prev) => {
          if (prev?.type !== "file") return prev;
          if (prev.file.workspaceId !== snapshot.workspaceId) return prev;
          if (prev.file.node.path !== snapshot.path) return prev;
          if (prev.file.content !== snapshot.content) return prev;
          return { ...prev, file: { ...prev.file, dirty: false } };
        });
      } catch (e) {
        setError(`Failed to save: ${e}`);
      } finally {
        setSaving(false);
      }
    }, AUTOSAVE_DELAY_MS);

    return () => window.clearTimeout(timerId);
  }, [activeView, workspaces]);

  // Mirror finished jobs to localStorage so they survive runner restarts or
  // local jobs.json being wiped.
  useEffect(() => {
    savePersistedJobs(PERSISTED_INIT_JOBS_KEY, initJobs);
  }, [initJobs]);
  useEffect(() => {
    savePersistedJobs(PERSISTED_RUN_JOBS_KEY, runJobs);
  }, [runJobs]);

  // Fetch the full job history from the runner once on mount so any jobs
  // persisted to disk from previous sessions appear in the Jobs panel.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [initRes, runRes] = await Promise.all([
          fetch(`${INIT_RUNNER_URL}/api/init-jobs`),
          fetch(`${INIT_RUNNER_URL}/api/run-jobs`),
        ]);
        if (!cancelled && initRes.ok) {
          const jobs = (await initRes.json()) as InitJob[];
          setInitJobs((prev) => mergeJobs(jobs, prev));
        }
        if (!cancelled && runRes.ok) {
          const jobs = (await runRes.json()) as RunJob[];
          setRunJobs((prev) => mergeJobs(jobs, prev));
        }
      } catch {
        // Runner may not be running yet; ignore.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const hasActive = initJobs.some((j) => j.status === "running");
    if (!hasActive) return;

    const timerId = window.setInterval(async () => {
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/init-jobs`);
        if (!res.ok) return;
        const jobs = (await res.json()) as InitJob[];
        setInitJobs(normalizeJobs(jobs));
      } catch {
        // Ignore temporary polling failures.
      }
    }, 1500);

    return () => window.clearInterval(timerId);
  }, [initJobs]);

  useEffect(() => {
    const hasRunningJobs = initJobs.some((j) => j.status === "running");
    hadRunningInitJobsRef.current = hasRunningJobs;
  }, [initJobs]);

  useEffect(() => {
    const hasActive = runJobs.some((j) => j.status === "running");
    if (!hasActive) return;

    const timerId = window.setInterval(async () => {
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/run-jobs`);
        if (!res.ok) return;
        const jobs = (await res.json()) as RunJob[];
        setRunJobs(normalizeJobs(jobs));
      } catch {
        // Ignore temporary polling failures.
      }
    }, 1500);

    return () => window.clearInterval(timerId);
  }, [runJobs]);

  // Detect run-job status transitions (running → succeeded/failed/cancelled)
  // and emit a corresponding activity log entry exactly once per job.
  const loggedRunJobStatusRef = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    for (const job of runJobs) {
      const prev = loggedRunJobStatusRef.current.get(job.id);
      if (prev === job.status) continue;
      if (job.status === "running") {
        loggedRunJobStatusRef.current.set(job.id, job.status);
        continue;
      }
      const target = jobActivityLabel(job.rootPath, job.configType);
      if (job.status === "succeeded") {
        addActivityLog("Job succeeded", target, "success");
      } else if (job.status === "failed") {
        const lines = (job.output || "")
          .split(/\r?\n/)
          .map((l) => l.replace(/\s+$/g, ""))
          .filter((l) => l.trim().length > 0);
        // Prefer the last Python-style error line (e.g. "KeyError: 'x'").
        const errorLine = [...lines]
          .reverse()
          .find((l) => /^[A-Za-z_][\w.]*Error\b|^Exception\b|^error:/i.test(l.trim()));
        const summary = errorLine ?? lines[lines.length - 1] ?? "";
        const detail = summary
          ? `${target} (exit ${job.exitCode ?? "?"}) — ${summary}`
          : `${target} (exit ${job.exitCode ?? "?"})`;
        addActivityLog("Job failed", detail, "error");
      } else if (job.status === "cancelled") {
        addActivityLog("Job cancelled", `${target} — by the user`, "warning");
      }
      loggedRunJobStatusRef.current.set(job.id, job.status);
    }
  }, [runJobs, addActivityLog, jobActivityLabel]);

  // While any run job is active, periodically re-list the workspaces it
  // targets so files the job creates appear in the sidebar without a
  // manual refresh.
  useEffect(() => {
    const activeRootPaths = new Set(
      runJobs
        .filter((j) => j.status === "running" && j.rootPath)
        .map((j) => j.rootPath!),
    );
    if (activeRootPaths.size > 0) {
      hadRunningRunJobsRef.current = true;
      lastActiveRunRootsRef.current = activeRootPaths;
    }
    if (activeRootPaths.size === 0) return;

    const tick = () => {
      for (const ws of workspaces) {
        if (ws.sourceKind !== "local") continue;
        const wsRoot = ws.rootPath;
        if (!wsRoot) continue;
        // Refresh if the workspace root matches a running job root, or is a
        // parent/child of it (so output written into a sibling folder shows up).
        const match = [...activeRootPaths].some((rp) =>
          rootPathsRelated(wsRoot, rp),
        );
        if (match) {
          // Lightweight refresh while the job is running. Avoid forcing
          // FolderTree remounts on every tick, which can freeze the UI for
          // file-heavy jobs.
          void refreshWorkspace(ws);
        }
      }
    };
    tick();
    const timerId = window.setInterval(tick, 4000);
    return () => window.clearInterval(timerId);
  }, [runJobs, workspaces, refreshWorkspace]);

  // When run jobs finish, do one hard tree refresh so newly created files in
  // expanded subfolders become visible without repeated remount churn.
  useEffect(() => {
    const activeRoots = new Set(
      runJobs
        .filter((j) => j.status === "running" && j.rootPath)
        .map((j) => j.rootPath!),
    );
    if (activeRoots.size > 0) return;
    if (!hadRunningRunJobsRef.current) return;

    const previousRoots = lastActiveRunRootsRef.current;
    hadRunningRunJobsRef.current = false;
    lastActiveRunRootsRef.current = new Set();

    for (const ws of workspaces) {
      if (ws.sourceKind !== "local") continue;
      const wsRoot = ws.rootPath;
      if (!wsRoot) continue;
      const match = [...previousRoots].some((rp) =>
        rootPathsRelated(wsRoot, rp),
      );
      if (!match) continue;
      // Bump the tree nonce unconditionally (not gated on the refresh result).
      // refreshWorkspace returns false when a refresh is already in-flight
      // (the 4s running-tick refresh commonly overlaps job completion). The
      // nonce bump remounts FolderTree, which re-fetches every persisted-
      // expanded subfolder fresh, so newly created deep files become visible.
      void refreshWorkspace(ws).finally(() => {
        scheduleTreeNonceBump(ws.id, 50);
      });
    }
  }, [runJobs, workspaces, refreshWorkspace, scheduleTreeNonceBump]);

  const activeWorkspace = activeView?.type === "file"
    ? workspaces.find((w) => w.id === activeView.file.workspaceId)
    : undefined;

  // The Copilot agent UI lives as an editor tab. Defining it once here lets
  // us render it inside the editor content area and keeps all props in one
  // place so opening/closing the tab is just an `activeView` switch.
  const copilotPanelElement = (
    <CopilotPanel
      open={copilotWizardOpen}
      inline
      initialPrompt={
        copilotPromptOverride ??
        (copilotPromptArmed
          ? (COPILOT_SKILLS.find((s) => s.id === copilotSkillId) ??
              COPILOT_SKILLS[0]).initialPrompt
          : undefined)
      }
      silentInitialPrompt
      skillLabel={
        (COPILOT_SKILLS.find((s) => s.id === copilotSkillId) ??
          COPILOT_SKILLS[0]).id
      }
      skillChoices={
        !copilotPromptArmed && !copilotPromptOverride
          ? COPILOT_SKILLS.map((s) => ({
              id: s.id,
              label: s.label,
              description: s.description,
            }))
          : undefined
      }
      onSkillSelected={(id) => {
        setCopilotSkillId(id);
        if (id === "benchmark-qed-question-quality") {
          setCopilotInlineFlow("question-quality");
          return;
        }
        setCopilotPromptArmed(true);
      }}
      inlinePicker={
        copilotInlineFlow === "question-quality"
          ? {
              title: "Evaluate Question Quality",
              node: (
                <EvaluateQuestionsPickerForm
                  workspaces={workspaces
                    .filter((w) => w.rootPath && !w.hiddenFromSidebar)
                    .map((w) => ({
                      id: w.id,
                      name: w.name,
                      rootPath: w.rootPath ?? "",
                    }))}
                  pickFolder={pickLocalFolderPath}
                  onCancel={() => setCopilotInlineFlow(null)}
                  onOpenReport={(report) => {
                    setCopilotWizardOpen(false);
                    setCopilotInlineFlow(null);
                    setCopilotPromptOverride(null);
                    setCopilotPromptArmed(false);
                    if (activeView?.type === "copilot") setActiveView(null);
                    void openRecentReport(report);
                  }}
                  onSubmit={(params) => {
                    const prompt = buildQuestionQualityPrompt(params);
                    setCopilotPromptOverride(prompt);
                    setCopilotPromptArmed(true);
                    setCopilotInlineFlow(null);
                    // Record the pending report so we can auto-open it
                    // once the agent writes the file. The recent-reports
                    // history entry is added only after the file actually
                    // appears on disk (in finalizePendingReport).
                    const destFolder = params.destinationPath.replace(
                      /[\\/]+$/,
                      "",
                    );
                    const folderName =
                      destFolder.split(/[\\/]/).filter(Boolean).pop() ??
                      destFolder;
                    // Match the user's native path style (see
                    // buildQuestionQualityPrompt).
                    const joinSep = destFolder.includes("\\") ? "\\" : "/";
                    pendingReportRef.current = {
                      reportPath: `${destFolder}${joinSep}${params.reportFilename}`,
                      destinationPath: destFolder,
                      label: `${folderName} / ${params.reportFilename}`,
                      setLabels: params.entries.map((e) => e.label),
                    };
                  }}
                  resetKey={copilotInlineFlow ?? "closed"}
                />
              ),
            }
          : undefined
      }
      onFolderDetected={(folderPath) => {
        const alreadyAdded = workspaces.some(
          (w) => hasSameLocalRootPath(w.rootPath, folderPath),
        );
        if (!alreadyAdded) {
          void addLocalWorkspace({
            path: folderPath,
            hasChildWorkspaces: false,
            childWorkspacePaths: [],
          });
        }
      }}
      onActivitySettled={() => {
        for (const ws of workspaces) {
          if (ws.sourceKind === "local") {
            void refreshWorkspace(ws).then((refreshed) => {
              if (!refreshed) return;
              scheduleTreeNonceBump(ws.id);
            });
          }
        }
        // Try to open a freshly-written quality report (if any).
        void finalizePendingReport();
      }}
      onWorkspaceCreated={({ path, configType }) => {
        // The benchmark-qed-setup skill creates workspaces by running the
        // CLI through its own shell tool, so the runner's /api/init-jobs
        // auto-import effect never fires. The skill emits a hidden
        // `<!-- benchmark-qed:workspace-created -->` marker in its closing
        // message; we pick it up here and register the workspace so it
        // shows up in the sidebar without the user having to add it
        // manually.
        if (!path) return;
        const alreadyAdded = workspacesRef.current.some(
          (w) => hasSameLocalRootPath(w.rootPath, path),
        );
        if (alreadyAdded) return;
        const allowed: ReadonlyArray<WorkspaceConfigType> = [
          "autoq",
          "autoe_pairwise",
          "autoe_reference",
          "autoe_assertion",
        ];
        const typed =
          configType && (allowed as readonly string[]).includes(configType)
            ? (configType as WorkspaceConfigType)
            : undefined;
        const name = path.split(/[/\\]/).filter(Boolean).pop() ?? path;
        void addWorkspace(
          name,
          new RunnerPathFileSource(path, INIT_RUNNER_URL),
          {
            rootPath: path,
            configType: typed,
            persisted: {
              kind: "local",
              name,
              rootPath: path,
              configType: typed,
            },
          },
        );
      }}
      onLogEvent={(action, details, type) =>
        addActivityLog(action, details, type)
      }
      onBackToOptions={() => {
        // Reset skill-selection state so CopilotPanel renders the picker
        // again. The panel itself archives the prior transcript so the
        // user keeps history of the finished run.
        setCopilotPromptOverride(null);
        setCopilotPromptArmed(false);
        setCopilotInlineFlow(null);
        pendingReportRef.current = null;
      }}
      onClose={() => {
        setCopilotWizardOpen(false);
        setCopilotPromptOverride(null);
        setCopilotPromptArmed(false);
        setCopilotInlineFlow(null);
        pendingReportRef.current = null;
        if (activeView?.type === "copilot") setActiveView(null);
      }}
    />
  );

  const sidebarEntries = useMemo(() => {
    const visibleWorkspaces = workspaces.filter((w) => !w.hiddenFromSidebar);
    const childrenByParent = new Map<string, Workspace[]>();
    for (const w of visibleWorkspaces) {
      const parentRootPath = w.parentRootPath ?? w.copyOfRootPath;
      if (!parentRootPath) continue;
      const arr = childrenByParent.get(parentRootPath) ?? [];
      arr.push(w);
      childrenByParent.set(parentRootPath, arr);
    }

    const visited = new Set<string>();
    const ordered: Array<{
      ws: Workspace;
      depth: number;
      parentName?: string;
      isCopyChild: boolean;
    }> = [];
    const visit = (
      ws: Workspace,
      depth: number,
      parentName?: string,
      isCopyChild = false,
    ) => {
      if (visited.has(ws.id)) return;
      visited.add(ws.id);
      ordered.push({ ws, depth, parentName, isCopyChild });
      if (!ws.rootPath) return;
      for (const child of childrenByParent.get(ws.rootPath) ?? []) {
        visit(child, depth + 1, ws.name, child.copyOfRootPath === ws.rootPath);
      }
    };

    for (const w of visibleWorkspaces) {
      const linkedParentRoot = w.parentRootPath ?? w.copyOfRootPath;
      const parentLoaded =
        !!linkedParentRoot &&
        visibleWorkspaces.some((p) => p.rootPath === linkedParentRoot);
      if (!linkedParentRoot || !parentLoaded) {
        visit(w, 0);
      }
    }

    const normalizedFilter = workspaceFilter.trim().toLowerCase();
    const kindOrdered =
      workspaceKindFilter === "all"
        ? ordered
        : ordered.filter(({ ws }) => ws.sourceKind === workspaceKindFilter);

    if (!normalizedFilter) {
      return {
        visibleCount: visibleWorkspaces.length,
        filteredCount: kindOrdered.length,
        filtered: kindOrdered,
      };
    }

    const includeIds = new Set<string>();
    for (const { ws } of kindOrdered) {
      const haystack = [
        ws.name,
        ws.rootPath ?? "",
        ws.configType ?? "",
        ws.parentRootPath ?? "",
        ws.copyOfRootPath ?? "",
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(normalizedFilter)) continue;

      let current: Workspace | undefined = ws;
      while (current) {
        if (includeIds.has(current.id)) break;
        includeIds.add(current.id);
        const parentRoot: string | undefined =
          current.parentRootPath ?? current.copyOfRootPath;
        if (!parentRoot) break;
        current = visibleWorkspaces.find(
          (candidate) => candidate.rootPath === parentRoot,
        );
      }
    }

    const filtered = kindOrdered.filter(({ ws }) => includeIds.has(ws.id));
    return {
      visibleCount: visibleWorkspaces.length,
      filteredCount: filtered.length,
      filtered,
    };
  }, [workspaces, workspaceFilter, workspaceKindFilter]);

  const hiddenDirectoryPathIdsByWorkspace = useMemo(() => {
    const visibleLocalWorkspaces = workspaces.filter(
      (w) => !w.hiddenFromSidebar && w.sourceKind === "local" && !!w.rootPath,
    );
    const byWorkspaceId = new Map<string, Set<string>>();

    for (const ws of visibleLocalWorkspaces) {
      const wsRootId = localPathIdentity(ws.rootPath);
      const hidden = new Set<string>();
      if (!wsRootId) {
        byWorkspaceId.set(ws.id, hidden);
        continue;
      }

      for (const other of visibleLocalWorkspaces) {
        if (other.id === ws.id) continue;
        const otherRootId = localPathIdentity(other.rootPath);
        if (!otherRootId) continue;
        if (!otherRootId.startsWith(`${wsRootId}/`)) continue;
        const rel = relativePathIdentity(otherRootId.slice(wsRootId.length + 1));
        if (rel) hidden.add(rel);
      }

      // Fallback for parent workspaces that explicitly opt into child
      // workspaces: hide top-level directory names that match basenames of
      // loaded child workspace roots, even when absolute prefix comparison
      // is insufficient due to path-shape differences.
      if (ws.hasChildWorkspaces) {
        for (const other of visibleLocalWorkspaces) {
          if (other.id === ws.id) continue;
          const basename = other.rootPath?.split(/[\\/]/).filter(Boolean).pop();
          const relBase = relativePathIdentity(basename);
          if (relBase) hidden.add(relBase);
        }
      }

      byWorkspaceId.set(ws.id, hidden);
    }

    return byWorkspaceId;
  }, [workspaces]);

  return (
    <div className="app-root">
      <nav className="navbar">
        <span className="navbar-title">Benchmark-QED Dashboard</span>
        <button
          className="navbar-theme-toggle"
          onClick={() => {
            const newTheme = theme === "dark" ? "light" : "dark";
            setTheme(newTheme);
          }}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <WeatherMoon24Regular /> : <WeatherSunny24Regular />}
        </button>
      </nav>
      <div className="dashboard" style={{ display: "flex" }}>
      {sidebarCollapsed ? (
        <aside className="sidebar collapsed" aria-label="Workspaces (collapsed)">
          <button
            type="button"
            className="sidebar-collapse-toggle"
            onClick={() => setSidebarCollapsed(false)}
            title="Expand workspaces sidebar"
            aria-label="Expand workspaces sidebar"
          >
            <PanelLeft24Regular />
          </button>
        </aside>
      ) : (
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-header-top">
            <span className="sidebar-header-title">Workspaces</span>
            <button
              type="button"
              className="sidebar-collapse-toggle inline"
              onClick={() => setSidebarCollapsed(true)}
              title="Collapse sidebar"
              aria-label="Collapse sidebar"
            >
              <PanelLeft24Regular />
            </button>
          </div>
          <div className="ai-skill-trigger-wrap">
            <button
              className="open-btn"
              onClick={() => {
                // If the Copilot tab is already open, just focus it so we
                // don't blow away an in-progress session / transcript.
                // Only fresh opens reset the skill picker state.
                if (copilotWizardOpen) {
                  setActiveView({ type: "copilot" });
                  return;
                }
                setCopilotSkillId(DEFAULT_COPILOT_SKILL_ID);
                setCopilotPromptOverride(null);
                setCopilotPromptArmed(false);
                setCopilotInlineFlow(null);
                setCopilotWizardOpen(true);
                setActiveView({ type: "copilot" });
              }}
              title="Launch the embedded Copilot agent"
            >
              ✨ AI Assistant
            </button>
          </div>
          <button className="open-btn secondary" onClick={() => setInitDialogOpen(true)}>
            + Create Configuration
          </button>
          <button className="open-btn secondary" onClick={() => setAddWorkspaceDialogOpen(true)}>
            + Add Workspace
          </button>
          <label className="sidebar-filter" aria-label="Filter workspaces">
            <div className="sidebar-filter-input-wrap">
              <input
                type="text"
                value={workspaceFilter}
                onChange={(e) => setWorkspaceFilter(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape" && workspaceFilter) {
                    e.preventDefault();
                    setWorkspaceFilter("");
                  }
                }}
                placeholder="Filter workspaces..."
              />
              {workspaceFilter && (
                <button
                  type="button"
                  className="sidebar-filter-clear"
                  onClick={() => setWorkspaceFilter("")}
                  aria-label="Clear workspace filter"
                  title="Clear"
                >
                  <Dismiss16Regular />
                </button>
              )}
            </div>
          </label>
          <div
            className="sidebar-filter-kind"
            role="group"
            aria-label="Filter by storage type"
          >
            {(["all", "local", "blob"] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                className={`sidebar-filter-kind-btn${
                  workspaceKindFilter === kind ? " active" : ""
                }${kind === "blob" ? " blob" : ""}`}
                aria-pressed={workspaceKindFilter === kind}
                onClick={() => setWorkspaceKindFilter(kind)}
                title={
                  kind === "all"
                    ? "Show all workspaces"
                    : kind === "local"
                      ? "Show local workspaces only"
                      : "Show blob storage workspaces only"
                }
              >
                {kind === "all" ? "All" : kind === "local" ? "Local" : "Blob"}
              </button>
            ))}
          </div>
          <div className="ws-count">
            {workspaceFilter.trim() || workspaceKindFilter !== "all"
              ? `${sidebarEntries.filteredCount} of ${sidebarEntries.visibleCount}`
              : `${sidebarEntries.visibleCount} workspace${sidebarEntries.visibleCount === 1 ? "" : "s"}`}
          </div>
        </div>

        <div className="sidebar-body">
          {blobRestoreProgress && (
            <div
              className="blob-restore-banner"
              role="status"
              aria-live="polite"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 10px",
                margin: "6px 8px",
                borderRadius: 6,
                background: "rgba(80, 140, 220, 0.12)",
                border: "1px solid rgba(80, 140, 220, 0.40)",
                fontSize: 12,
              }}
            >
              <ArrowSync16Regular className="spin" />
              <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                <span>
                  Loading blob workspaces ({blobRestoreProgress.done}/
                  {blobRestoreProgress.total})...
                </span>
                {blobRestoreProgress.currentLabel && (
                  <span
                    style={{
                      opacity: 0.75,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    title={blobRestoreProgress.currentLabel}
                  >
                    {blobRestoreProgress.currentLabel}
                  </span>
                )}
              </div>
            </div>
          )}
          {workspaces.length === 0 ? (
            blobRestoreProgress ? null : (
              <div className="empty-tree">
                No workspaces opened. Add a local folder or an Azure Blob container.
              </div>
            )
          ) : (
            (() => {
              if (sidebarEntries.filtered.length === 0) {
                return (
                  <div className="empty-tree">
                    {workspaceFilter.trim() ? (
                      <>
                        No {workspaceKindFilter !== "all" ? `${workspaceKindFilter} ` : ""}
                        workspaces match <code>{workspaceFilter.trim()}</code>.
                      </>
                    ) : (
                      <>No {workspaceKindFilter} workspaces.</>
                    )}
                  </div>
                );
              }

              return sidebarEntries.filtered.map(({ ws, depth, parentName, isCopyChild }) => (
              <div
                key={ws.id}
                className={`workspace${
                  depth > 0 ? " workspace-child" : ""
                }${
                  draggingWorkspaceId === ws.id ? " workspace-dragging" : ""
                }${
                  dragOverWorkspaceId === ws.id &&
                  draggingWorkspaceId &&
                  draggingWorkspaceId !== ws.id
                    ? " workspace-drag-over"
                    : ""
                }`}
                style={depth > 0 ? { paddingLeft: depth * 14 } : undefined}
                onDragOver={(e) => {
                  if (!draggingWorkspaceId || draggingWorkspaceId === ws.id) return;
                  e.preventDefault();
                  e.dataTransfer.dropEffect = "move";
                  if (dragOverWorkspaceId !== ws.id) setDragOverWorkspaceId(ws.id);
                }}
                onDragLeave={(e) => {
                  // Only clear when leaving the workspace container itself.
                  if (e.currentTarget === e.target && dragOverWorkspaceId === ws.id) {
                    setDragOverWorkspaceId(null);
                  }
                }}
                onDrop={(e) => {
                  if (!draggingWorkspaceId) return;
                  e.preventDefault();
                  reorderWorkspaces(draggingWorkspaceId, ws.id);
                  setDraggingWorkspaceId(null);
                  setDragOverWorkspaceId(null);
                }}
              >
                <div
                  className="workspace-header"
                  draggable
                  onDragStart={(e) => {
                    setDraggingWorkspaceId(ws.id);
                    e.dataTransfer.effectAllowed = "move";
                    try {
                      e.dataTransfer.setData("text/plain", ws.id);
                    } catch {
                      // Some browsers throw if setData is called without a type; ignore.
                    }
                  }}
                  onDragEnd={() => {
                    setDraggingWorkspaceId(null);
                    setDragOverWorkspaceId(null);
                  }}
                  title="Drag to reorder"
                >
                  <button
                    className="ws-toggle"
                    onClick={() => toggleWorkspace(ws.id)}
                    title={ws.collapsed ? "Expand" : "Collapse"}
                  >
                    {ws.collapsed ? <ChevronRight16Regular /> : <ChevronDown16Regular />}
                  </button>
                  <span className={`ws-kind ws-kind-${ws.sourceKind}`}>
                    {ws.sourceKind === "blob" ? <Cloud16Regular /> : <Folder16Regular />}
                  </span>
                  {parentName && isCopyChild && (
                    <span
                      className="ws-copy-indicator"
                      title={`Copy of ${parentName}`}
                      aria-label={`Copy of ${parentName}`}
                    >
                      <Copy16Regular />
                    </span>
                  )}
                  <span
                    className="ws-name"
                    title={
                      parentName && isCopyChild
                        ? `${ws.name} — copy of ${parentName}`
                        : parentName
                          ? `${ws.name} — child of ${parentName}`
                          : ws.name
                    }
                  >
                    {ws.name}
                  </span>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => setDatasetDialogWorkspaceId(ws.id)}
                    title={
                      ws.rootPath || ws.sourceKind === "blob"
                        ? "Download predefined inputs"
                        : "Unavailable for this workspace"
                    }
                    aria-label="Download predefined inputs"
                    disabled={!ws.rootPath && ws.sourceKind !== "blob"}
                  >
                    <ArrowDownload16Regular />
                  </button>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => void handleRunWorkspace(ws)}
                    title={
                      detectingConfigsWorkspaceIds.has(ws.id)
                        ? "Detecting configs…"
                        : ws.configType
                          ? `Run ${ws.configType}`
                          : "Run workspace"
                    }
                    aria-label="Run workspace"
                    aria-busy={detectingConfigsWorkspaceIds.has(ws.id)}
                    disabled={
                      (!ws.rootPath && ws.sourceKind !== "blob") ||
                      detectingConfigsWorkspaceIds.has(ws.id)
                    }
                  >
                    {detectingConfigsWorkspaceIds.has(ws.id) ? (
                      <ArrowSync16Regular className="spin" />
                    ) : (
                      <Play16Regular />
                    )}
                  </button>
                  <div className="ws-actions-trigger-wrap">
                    <button
                      className="ws-run ws-icon-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActionsMenuWorkspaceId((prev) =>
                          prev === ws.id ? null : ws.id,
                        );
                      }}
                      title="Folder actions"
                      aria-label="Folder actions"
                      aria-haspopup="menu"
                      aria-expanded={actionsMenuWorkspaceId === ws.id}
                    >
                      <Navigation16Regular />
                    </button>
                    <WorkspaceActionsMenu
                      open={actionsMenuWorkspaceId === ws.id}
                      onClose={() => setActionsMenuWorkspaceId(null)}
                      items={[
                        {
                          key: "new-file",
                          label: "New file",
                          icon: <Document16Regular />,
                          onClick: () => requestCreatePath(ws, "file"),
                          disabled:
                            !ws.source.canWrite() || !ws.source.createFile,
                          disabledTitle: "Workspace is read-only",
                        },
                        {
                          key: "new-folder",
                          label: "New folder",
                          icon: <FolderAdd16Regular />,
                          onClick: () => requestCreatePath(ws, "directory"),
                          disabled:
                            !ws.source.canWrite() || !ws.source.createDirectory,
                          disabledTitle: "Workspace is read-only",
                        },
                        {
                          key: "copy",
                          label: "Copy workspace…",
                          icon: <Copy16Regular />,
                          onClick: () => requestCopyWorkspace(ws),
                          disabled: !ws.rootPath || !!ws.copyOfRootPath,
                          disabledTitle: ws.copyOfRootPath
                            ? "This workspace is already a copy"
                            : "Copy is only available for local workspaces",
                        },
                        {
                          key: "rename",
                          label: "Rename workspace…",
                          icon: <Edit16Regular />,
                          onClick: () => requestRenameWorkspace(ws),
                        },
                      ]}
                    />
                  </div>
                  <button
                    className="ws-close"
                    onClick={() => closeWorkspace(ws.id)}
                    title="Close workspace"
                  >
                    <Dismiss16Regular />
                  </button>
                </div>
                {!ws.collapsed && (
                  <div className="workspace-tree">
                    <FolderTree
                      key={`${ws.id}-${treeNonces[ws.id] ?? 0}`}
                      workspaceKey={workspaceStableKey(ws)}
                      source={ws.source}
                      nodes={ws.rootNodes}
                      hiddenDirectoryPathIds={hiddenDirectoryPathIdsByWorkspace.get(ws.id)}
                      onSelectFile={(n) => handleSelectFile(ws, n)}
                      onCreateFile={(parent) => requestCreatePath(ws, "file", parent)}
                      onCreateFolder={(parent) =>
                        requestCreatePath(ws, "directory", parent)
                      }
                      onDeleteNode={(node) => requestDeletePath(ws, node)}
                      onCopyNode={
                        ws.sourceKind === "local"
                          ? (node) => requestCopyNode(ws, node)
                          : undefined
                      }
                      onRenameNode={
                        ws.source.renamePath
                          ? (node) => requestRenameNode(ws, node)
                          : undefined
                      }
                      selectedPath={
                        activeView?.type === "file" && activeView.file.workspaceId === ws.id
                          ? activeView.file.node.path
                          : undefined
                      }
                    />
                  </div>
                )}
              </div>
              ));
            })()
          )}
        </div>
      </aside>
      )}

      <JobsPanel
        initJobs={initJobs}
        runJobs={runJobs}
        selectedJob={
          activeView?.type === "run-job"
            ? { kind: "run", id: activeView.jobId }
            : activeView?.type === "init-job"
              ? { kind: "init", id: activeView.jobId }
              : null
        }
        onOpenInitJob={handleOpenInitJobLog}
        onOpenRunJob={handleOpenRunJobLog}
        onCancelRunJob={cancelRunJob}
        onRerunJob={rerunJob}
        onDeleteJob={deleteJob}
        workspaceNameByRoot={Object.fromEntries(
          workspaces
            .filter((w) => !!w.rootPath && !w.hiddenFromSidebar)
            .map((w) => [w.rootPath as string, w.name]),
        )}
        collapsed={jobsPanelCollapsed}
        onToggleCollapsed={() => setJobsPanelCollapsed((v) => !v)}
      />

      <main className="main">
        {error && <div className="error-banner">{error}</div>}
        {configurePrompt && (
          <div className="info-banner configure-banner">
            <div className="configure-banner-text">
              <strong>{configurePrompt.summary}</strong>
              <span>
                Configure <code>{configurePrompt.name}</code> with Copilot to
                generate <code>settings.yaml</code> using the{" "}
                <code>benchmark-qed-setup</code> skill.
              </span>
            </div>
            <div className="configure-banner-actions">
              <button
                className="btn btn-primary"
                onClick={() => {
                  const params = new URLSearchParams({
                    workspace: configurePrompt.workspace,
                    dataset: configurePrompt.name,
                  });
                  window.location.href = `vscode://benchmark-qed.bridge/run-skill?${params.toString()}`;
                }}
              >
                Configure with Copilot
              </button>
              <button
                className="btn"
                onClick={() => setConfigurePrompt(null)}
              >
                Dismiss
              </button>
            </div>
          </div>
        )}
        
        {/* Top-level tabs for files and job logs */}
        <div className="editor-tabs">
          {/* File tabs */}
          {openFiles.map((file) => {
            const ws = workspaces.find((w) => w.id === file.workspaceId);
            const pathParts = file.node.path.split(/[/\\]/).filter(Boolean);
            const parentSegments = pathParts.slice(0, -1);
            const parentPath = parentSegments.join("/");
            const contextLabel = ws?.name
              ? parentPath
                ? `${ws.name}/${parentPath}`
                : ws.name
              : parentPath;
            const isActive =
              activeView?.type === "file" &&
              activeView.file.workspaceId === file.workspaceId &&
              activeView.file.node.path === file.node.path;
            const key = fileKey(file);
            return (
              <button
                key={key}
                className={`editor-tab ${isActive ? "active" : ""}`}
                type="button"
                onClick={() => setActiveView({ type: "file", file })}
                title={ws ? `${ws.name}/${file.node.path}` : file.node.path}
              >
                <Document16Regular className="tab-icon" />
                {file.node.name}
                {contextLabel && (
                  <span className="tab-context">{contextLabel}</span>
                )}
                {file.dirty && <span className="tab-dirty">●</span>}
                <button
                  type="button"
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenFiles((prev) => {
                      const next = prev.filter((f) => fileKey(f) !== key);
                      if (isActive) {
                        const fallback = next[next.length - 1];
                        setActiveView(
                          fallback ? { type: "file", file: fallback } : null,
                        );
                      }
                      return next;
                    });
                  }}
                  title="Close file"
                >
                  <Dismiss16Regular />
                </button>
              </button>
            );
          })}
          
          {/* Run job tabs */}
          {Array.from(openRunJobIds).map((jobId) => {
            const job = runJobs.find((j) => j.id === jobId);
            if (!job) return null;
            const folderName = job.rootPath
              ? job.rootPath.split(/[/\\]/).filter(Boolean).pop()
              : undefined;
            const configLabel = job.configType ?? "job";
            const label = folderName ? `${folderName} · ${configLabel}` : configLabel;
            return (
              <button
                key={jobId}
                type="button"
                className={`editor-tab ${activeView?.type === "run-job" && activeView.jobId === jobId ? "active" : ""}`}
                onClick={() => setActiveView({ type: "run-job", jobId })}
                title={`${label} (${job.status})\n${job.command}`}
              >
                <ClipboardTask16Regular className="tab-icon" />
                {label}
                <span className={`tab-job-status init-status-${job.status}`}>
                  {job.status}
                </span>
                <button
                  type="button"
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCloseRunJobLog(jobId);
                  }}
                  title="Close job log"
                >
                  <Dismiss16Regular />
                </button>
              </button>
            );
          })}

          {/* Init job tabs */}
          {Array.from(openInitJobIds).map((jobId) => {
            const job = initJobs.find((j) => j.id === jobId);
            if (!job) return null;
            const folderName = job.rootPath
              ? job.rootPath.split(/[/\\]/).filter(Boolean).pop()
              : undefined;
            const configLabel = job.configType ?? "init";
            const isBlobJob =
              job.storageType === "blob" ||
              /--storage-type\s+blob\b/.test(job.command ?? "");
            const displayName = isBlobJob ? undefined : folderName;
            const label = displayName
              ? `${displayName} · ${configLabel}`
              : `init · ${configLabel}`;
            return (
              <button
                key={jobId}
                type="button"
                className={`editor-tab ${activeView?.type === "init-job" && activeView.jobId === jobId ? "active" : ""}`}
                onClick={() => setActiveView({ type: "init-job", jobId })}
                title={`${label} (${job.status})\n${job.command}`}
              >
                <ClipboardTask16Regular className="tab-icon" />
                {label}
                <span className={`tab-job-status init-status-${job.status}`}>
                  {job.status}
                </span>
                <button
                  type="button"
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCloseInitJobLog(jobId);
                  }}
                  title="Close job log"
                >
                  <Dismiss16Regular />
                </button>
              </button>
            );
          })}

          {/* Copilot AI Assistant tab (singleton) */}
          {copilotWizardOpen && (
            <button
              key="__copilot__"
              type="button"
              className={`editor-tab ${activeView?.type === "copilot" ? "active" : ""}`}
              onClick={() => setActiveView({ type: "copilot" })}
              title="Copilot AI Assistant"
            >
              <span className="tab-icon" aria-hidden>✨</span>
              AI Assistant
              <button
                type="button"
                className="tab-close"
                onClick={(e) => {
                  e.stopPropagation();
                  setCopilotWizardOpen(false);
                  setCopilotPromptOverride(null);
                  setCopilotPromptArmed(false);
                  setCopilotInlineFlow(null);
                  if (activeView?.type === "copilot") {
                    setActiveView(null);
                  }
                }}
                title="Close Copilot"
              >
                <Dismiss16Regular />
              </button>
            </button>
          )}
        </div>
        
        {/* Editor content */}
        {activeView?.type === "file" ? (
          <FileEditor
            file={activeView.file}
            workspaceName={activeWorkspace?.name}
            readOnly={activeWorkspace ? !activeWorkspace.source.canWrite() : false}
            onContentChange={handleContentChange}
            onSave={handleSave}
            saving={saving}
            theme={theme}
          />
        ) : activeView?.type === "run-job" ? (
          <JobLogViewer
            job={runJobs.find((j) => j.id === activeView.jobId)!}
            onCancel={cancelRunJob}
            onRerun={rerunJob}
          />
        ) : activeView?.type === "init-job" ? (
          (() => {
            const initJob = initJobs.find((j) => j.id === activeView.jobId);
            if (!initJob) return null;
            // InitJob is structurally compatible with RunJob's required fields
            // for the viewer (its status is a subset). No-op cancel because
            // init jobs aren't cancellable.
            return (
              <JobLogViewer
                job={initJob as unknown as RunJob}
                onCancel={() => {}}
              />
            );
          })()
        ) : activeView?.type === "copilot" ? (
          // Render nothing here when the Copilot tab is active — the
          // persistent pane below covers the editor area. Keeping it
          // mounted across tab switches preserves the running session and
          // transcript so the user can hop to a file and come back.
          null
        ) : (
          <div className="empty-state">
            <h2>No file or job open</h2>
            <p>Open a folder on the left and select a file to view or edit it, or run a job to see its logs.</p>
          </div>
        )}
        {/* Persistent Copilot pane. Mounted whenever the tab exists; only
            visible when the Copilot tab is the active view. This is what
            keeps the agent session alive when the user opens a file. */}
        {copilotWizardOpen && (
          <div
            className="copilot-tab-pane"
            style={{
              display: activeView?.type === "copilot" ? "flex" : "none",
            }}
          >
            {copilotPanelElement}
          </div>
        )}
      </main>
      <ActivityLogPanel
        entries={activityLog}
        onClear={() => setActivityLog([])}
        collapsed={activityLogCollapsed}
        onToggleCollapsed={() => setActivityLogCollapsed(!activityLogCollapsed)}
      />

      <AddWorkspaceTabsDialog
        open={addWorkspaceDialogOpen}
        onClose={() => setAddWorkspaceDialogOpen(false)}
        onPickLocalFolder={pickLocalFolderPath}
        onAddLocal={addLocalWorkspace}
        onAddBlob={addBlobWorkspace}
      />
      <InitRunDialog
        open={initDialogOpen}
        submitting={initSubmitting}
        onClose={() => setInitDialogOpen(false)}
        onSubmit={submitInitJob}
      />
      <RunConfigDialog
        open={runConfigDialogOpen}
        onClose={() => {
          setRunConfigDialogOpen(false);
          setRunConfigWorkspaceId(null);
        }}
        onSubmit={handleRunConfigSubmit}
      />
      <DatasetDownloadDialog
        open={datasetDialogWorkspaceId !== null}
        workspaceName={
          workspaces.find((w) => w.id === datasetDialogWorkspaceId)?.name ?? "workspace"
        }
        workspaceRootPath={
          workspaces.find((w) => w.id === datasetDialogWorkspaceId)?.rootPath
        }
        isBlob={
          workspaces.find((w) => w.id === datasetDialogWorkspaceId)?.sourceKind ===
          "blob"
        }
        submitting={datasetSubmitting}
        onClose={() => setDatasetDialogWorkspaceId(null)}
        onSubmit={submitDatasetDownload}
      />
      <LoadDatasetDialog
        open={loadDatasetDialogOpen}
        submitting={loadDatasetSubmitting}
        onClose={() => setLoadDatasetDialogOpen(false)}
        onSubmit={submitLoadDataset}
      />
      <TreeCreateDialog
        open={createDialog.open}
        kind={createDialog.kind}
        parentNode={createDialog.parentNode}
        submitting={createSubmitting}
        onClose={() => setCreateDialog({ open: false, workspaceId: null, kind: "file" })}
        onSubmit={submitCreatePath}
      />
      <TreeDeleteDialog
        open={deleteDialog.open}
        node={deleteDialog.node}
        submitting={deleteSubmitting}
        onClose={() => setDeleteDialog({ open: false, workspaceId: null, node: undefined })}
        onConfirm={confirmDeletePath}
      />
      {(() => {
        const ws = workspaces.find((w) => w.id === copyDialog.workspaceId);
        const sourcePath = ws?.rootPath ?? "";
        const defaultParent = sourcePath
          ? sourcePath.includes("/")
            ? sourcePath.slice(0, sourcePath.lastIndexOf("/"))
            : sourcePath.slice(0, sourcePath.lastIndexOf("\\"))
          : "";
        return (
          <CopyWorkspaceDialog
            open={copyDialog.open && !!ws}
            sourceName={ws?.name ?? ""}
            sourcePath={sourcePath}
            defaultParentDir={defaultParent}
            submitting={copySubmitting}
            pickFolder={pickLocalFolderPath}
            onClose={() => setCopyDialog({ open: false, workspaceId: null })}
            onSubmit={submitCopyWorkspace}
          />
        );
      })()}
      {(() => {
        const ws = workspaces.find((w) => w.id === renameDialog.workspaceId);
        return (
          <RenameWorkspaceDialog
            open={renameDialog.open && !!ws}
            currentName={ws?.name ?? ""}
            onClose={() => setRenameDialog({ open: false, workspaceId: null })}
            onSubmit={submitRenameWorkspace}
          />
        );
      })()}
      {(() => {
        const sourceWs = workspaces.find(
          (w) => w.id === copyNodeDialog.workspaceId,
        );
        const destWorkspaces = workspaces.filter(
          (w) =>
            w.sourceKind === "local" &&
            !!w.rootPath &&
            !w.hiddenFromSidebar,
        );
        return (
          <CopyNodeDialog
            open={copyNodeDialog.open && !!sourceWs && !!copyNodeDialog.node}
            sourceWorkspace={sourceWs}
            sourceNodePath={copyNodeDialog.node?.path}
            sourceNodeName={copyNodeDialog.node?.name}
            destWorkspaces={destWorkspaces}
            submitting={copyNodeSubmitting}
            onClose={() =>
              setCopyNodeDialog({ open: false, workspaceId: null })
            }
            onSubmit={submitCopyNode}
          />
        );
      })()}
      {(() => {
        const ws = workspaces.find(
          (w) => w.id === renameNodeDialog.workspaceId,
        );
        const node = renameNodeDialog.node;
        if (!ws || !node) return null;
        const parentPath = node.path.includes("/")
          ? node.path.slice(0, node.path.lastIndexOf("/"))
          : "";
        return (
          <RenameNodeDialog
            open={renameNodeDialog.open}
            kind={node.kind === "directory" ? "directory" : "file"}
            currentName={node.name}
            parentPath={parentPath}
            submitting={renameNodeSubmitting}
            onClose={() =>
              setRenameNodeDialog({ open: false, workspaceId: null })
            }
            onSubmit={submitRenameNode}
          />
        );
      })()}
      <MultiConfigDialog
        open={multiConfigDialog.open}
        folderPath={multiConfigDialog.folderPath}
        configs={multiConfigDialog.configs}
        submitting={multiConfigDialog.submitting}
        mode={runSubfolderConfigWorkspaceId ? "run" : "add"}
        onClose={() => {
          setMultiConfigDialog({
            open: false,
            folderPath: "",
            configs: [],
            submitting: false,
          });
          setRunSubfolderConfigWorkspaceId(null);
        }}
        onRunSelected={
          runSubfolderConfigWorkspaceId
            ? handleRunSubfolderConfig
            : handleRunSelectedConfigs
        }
      />
    </div>
    </div>
  );
}
