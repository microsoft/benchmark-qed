import { useCallback, useEffect, useRef, useState } from "react";
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
import { CopilotPanel } from "./components/CopilotPanel";
import { BlobFileSource } from "./sources/BlobFileSource";
import { RunnerPathFileSource } from "./sources/RunnerPathFileSource";
import type {
  FileSource,
  OpenFile,
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
  Document16Regular,
  FolderAdd16Regular,
  Dismiss16Regular,
  ClipboardTask16Regular,
} from '@fluentui/react-icons';
import { detectKind, detectLanguage } from "./utils/files";
import { ActivityLogPanel, type ActivityLogEntry } from "./components/ActivityLogPanel";
import "./App.css";

const AUTOSAVE_DELAY_MS = 800;
const INIT_RUNNER_URL = "http://localhost:8787";

type JobLike = {
  id: string;
  startedAt: string;
  rootPath?: string;
};

function normalizeJobs<T extends JobLike>(jobs: T[]): T[] {
  const sorted = [...jobs].sort(
    (a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt),
  );
  const seenRootPaths = new Set<string>();
  const result: T[] = [];

  for (const job of sorted) {
    if (job.rootPath) {
      if (seenRootPaths.has(job.rootPath)) continue;
      seenRootPaths.add(job.rootPath);
    }
    result.push(job);
    if (result.length >= 10) break;
  }

  return result;
}

type ActiveView = 
  | { type: "file"; file: OpenFile }
  | { type: "run-job"; jobId: string }
  | null;

export default function App() {
  const [theme, setTheme] = useState<"dark" | "light">(() =>
    (localStorage.getItem("theme") as "dark" | "light") ?? "dark"
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
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

  const [initSubmitting, setInitSubmitting] = useState(false);
  const [initJobs, setInitJobs] = useState<InitJob[]>([]);
  const importedInitJobIdsRef = useRef<Set<string>>(new Set());
  const hadRunningInitJobsRef = useRef(false);

  const [runJobs, setRunJobs] = useState<RunJob[]>([]);
  const [openRunJobIds, setOpenRunJobIds] = useState<Set<string>>(new Set());
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

  const addWorkspace = useCallback(
    async (
      name: string,
      source: FileSource,
      options?: { rootPath?: string; configType?: WorkspaceConfigType },
    ) => {
      try {
        const rootNodes = await source.listChildren();

        setWorkspaces((prev) => {
          const existingIndex = options?.rootPath
            ? prev.findIndex((w) => w.rootPath === options.rootPath)
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
              source,
              rootNodes,
              collapsed: false,
            };
            return next;
          }

          const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
          return [
            ...prev,
            {
              id,
              version: 1,
              name,
              sourceKind: source.kind,
              rootPath: options?.rootPath,
              configType: options?.configType,
              source,
              rootNodes,
              collapsed: false,
            },
          ];
        });
      } catch (e) {
        setError(`Failed to load ${name}: ${e}`);
      }
    },
    [],
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
    async (path: string) => {
      setAddWorkspaceDialogOpen(false);
      setError(null);
      try {
        const name = path.split(/[/\\]/).filter(Boolean).pop() ?? path;
        await addWorkspace(name, new RunnerPathFileSource(path, INIT_RUNNER_URL), {
          rootPath: path,
        });
      } catch (e) {
        setError(`Failed to add local workspace: ${e}`);
      }
    },
    [addWorkspace],
  );

  // For AddWorkspaceDialog: add blob workspace
  const addBlobWorkspace = useCallback(
    async (data: { sasUrl: string; prefix: string; label: string }) => {
      setAddWorkspaceDialogOpen(false);
      setError(null);
      try {
        await addWorkspace(data.label, new BlobFileSource(data.sasUrl, data.prefix));
      } catch (e) {
        setError(`Failed to connect to blob: ${e}`);
      }
    },
    [addWorkspace],
  );

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
            { rootPath: config.path, configType: config.configType },
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
  ): Promise<boolean> => {
    const configType = overrideConfigType ?? workspace.configType;
    if (!workspace.rootPath || !configType) {
      setError("This workspace does not have a generated config type to run.");
      return false;
    }

    setError(null);
    try {
      const res = await fetch(`${INIT_RUNNER_URL}/api/run-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rootPath: workspace.rootPath,
          configType,
        }),
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
        `${workspace.name} · ${configType}`,
        "success",
      );
      return true;
    } catch (e) {
      setError(
        `Failed to run workspace command. Start the runner with 'npm run init-runner'. ${e}`,
      );
      addActivityLog(`Failed to start job`, workspace.name, "error");
      return false;
    }
  }, [handleOpenRunJobLog, addActivityLog]);

  const handleRunSubfolderConfig = useCallback(
    async (selectedConfigs: DetectedConfig[]) => {
      setMultiConfigDialog((prev) => ({ ...prev, submitting: true }));
      setError(null);
      try {
        // Run each selected config
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
    [submitRunJob],
  );

  const submitDatasetDownload = useCallback(
    async (dataset: PredefinedDataset, destinationPath: string) => {
      if (!datasetDialogWorkspaceId) return;
      const workspace = workspaces.find((w) => w.id === datasetDialogWorkspaceId);
      if (!workspace?.rootPath) {
        setError("This workspace is not backed by a local path.");
        return;
      }

      setDatasetSubmitting(true);
      setError(null);
      try {
        const res = await fetch(`${INIT_RUNNER_URL}/api/datasets/download`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rootPath: workspace.rootPath,
            dataset,
            outputPath: destinationPath,
          }),
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

        await addWorkspace(workspace.name, workspace.source, {
          rootPath: workspace.rootPath,
          configType: workspace.configType,
        });
        setDatasetDialogWorkspaceId(null);
      } catch (e) {
        setError(`Failed to download dataset: ${e}`);
      } finally {
        setDatasetSubmitting(false);
      }
    },
    [addWorkspace, datasetDialogWorkspaceId, workspaces],
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
          { rootPath: destinationFolder },
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
      if (!workspace.rootPath) return;

      // Always prompt for config selection so the user can pick which job to run.
      setError(null);
      try {
        const detectRes = await fetch(`${INIT_RUNNER_URL}/api/detect-configs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rootPath: workspace.rootPath }),
        });
        const detectPayload = (await detectRes.json()) as {
          configs?: DetectedConfig[];
          error?: string;
        };

        if (detectRes.ok && detectPayload.configs && detectPayload.configs.length > 0) {
          // Show selection dialog so the user can choose the command
          setMultiConfigDialog({
            open: true,
            folderPath: workspace.rootPath,
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
      }
    },
    [],
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
    const job = runJobs.find((j) => j.id === jobId);
    if (!job || !job.rootPath || !job.configType) {
      setError("Cannot re-run job: missing workspace path or config type.");
      return;
    }
    const existing = workspaces.find(
      (w) => w.sourceKind === "local" && w.rootPath === job.rootPath,
    );
    const workspace: Workspace =
      existing ?? {
        id: `temp-rerun-${job.id}`,
        version: 1,
        name:
          job.rootPath.split(/[/\\]/).filter(Boolean).pop() ?? job.rootPath,
        sourceKind: "local",
        rootPath: job.rootPath,
        configType: job.configType,
        source: new RunnerPathFileSource(job.rootPath, INIT_RUNNER_URL),
        rootNodes: [],
        collapsed: false,
      };
    await submitRunJob(workspace, job.configType);
  }, [runJobs, workspaces, submitRunJob]);

  useEffect(() => {
    for (const job of initJobs) {
      if (job.status !== "succeeded") continue;
      if (!job.rootPath) continue;
      if (importedInitJobIdsRef.current.has(job.id)) continue;

      importedInitJobIdsRef.current.add(job.id);
      const name =
        job.rootPath.split(/[/\\]/).filter(Boolean).pop() ?? job.rootPath;
      void addWorkspace(
        name,
        new RunnerPathFileSource(job.rootPath, INIT_RUNNER_URL),
        { rootPath: job.rootPath, configType: job.configType },
      );
    }
  }, [addWorkspace, initJobs]);

  const closeWorkspace = useCallback(
    (id: string) => {
      setWorkspaces((prev) => prev.filter((w) => w.id !== id));
      if (activeView?.type === "file" && activeView.file.workspaceId === id) {
        setActiveView(null);
      }
      setOpenFiles((prev) => prev.filter((f) => f.workspaceId !== id));
    },
    [activeView],
  );

  const toggleWorkspace = useCallback((id: string) => {
    setWorkspaces((prev) =>
      prev.map((w) =>
        w.id === id ? { ...w, collapsed: !w.collapsed } : w,
      ),
    );
  }, []);

  const refreshWorkspace = useCallback(
    async (workspace: Workspace) => {
      await addWorkspace(workspace.name, workspace.source, {
        rootPath: workspace.rootPath,
        configType: workspace.configType,
      });
    },
    [addWorkspace],
  );

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
        await refreshWorkspace(workspace);
        setCreateDialog({ open: false, workspaceId: null, kind: "file" });
      } catch (e) {
        setError(`Failed to create ${kind}: ${e}`);
      } finally {
        setCreateSubmitting(false);
      }
    },
    [createDialog, refreshWorkspace, workspaces],
  );

  const requestDeletePath = useCallback((workspace: Workspace, node: TreeNode) => {
    setDeleteDialog({
      open: true,
      workspaceId: workspace.id,
      node,
    });
  }, []);

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
        await refreshWorkspace(workspace);
        setDeleteDialog({ open: false, workspaceId: null, node: undefined });
      } catch (e) {
        setError(`Failed to delete path: ${e}`);
      } finally {
        setDeleteSubmitting(false);
      }
    },
    [deleteDialog, refreshWorkspace, workspaces],
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
      const label = job.configType ?? "job";
      const folder = job.rootPath
        ? job.rootPath.split(/[/\\]/).filter(Boolean).pop()
        : undefined;
      const target = folder ? `${folder} · ${label}` : label;
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
  }, [runJobs, addActivityLog]);

  // While any run job is active, periodically re-list the workspaces it
  // targets so files the job creates appear in the sidebar without a
  // manual refresh.
  useEffect(() => {
    const activeRootPaths = new Set(
      runJobs
        .filter((j) => j.status === "running" && j.rootPath)
        .map((j) => j.rootPath!),
    );
    if (activeRootPaths.size === 0) return;

    const tick = () => {
      for (const ws of workspaces) {
        if (ws.sourceKind !== "local") continue;
        const wsRoot = ws.rootPath;
        if (!wsRoot) continue;
        // Refresh if the workspace root matches a running job root, or is a
        // parent/child of it (so output written into a sibling folder shows up).
        const match = [...activeRootPaths].some(
          (rp) =>
            wsRoot === rp ||
            rp.startsWith(`${wsRoot}/`) ||
            wsRoot.startsWith(`${rp}/`),
        );
        if (match) void refreshWorkspace(ws);
      }
    };
    tick();
    const timerId = window.setInterval(tick, 4000);
    return () => window.clearInterval(timerId);
  }, [runJobs, workspaces, refreshWorkspace]);

  const activeWorkspace = activeView?.type === "file"
    ? workspaces.find((w) => w.id === activeView.file.workspaceId)
    : undefined;

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
      <aside className="sidebar">
        <div className="sidebar-header">
          <button
            className="open-btn"
            onClick={() => setCopilotWizardOpen(true)}
            title="Run the benchmark-qed-setup skill in an embedded Copilot agent"
          >
            ✨ AI Assistant
          </button>
          <button className="open-btn secondary" onClick={() => setInitDialogOpen(true)}>
            + Create Configuration
          </button>
          <button className="open-btn secondary" onClick={() => setAddWorkspaceDialogOpen(true)}>
            + Add Workspace
          </button>
          <div className="ws-count">
            {workspaces.length} workspace{workspaces.length === 1 ? "" : "s"}
          </div>
        </div>

        <div className="sidebar-body">
          {workspaces.length === 0 ? (
            <div className="empty-tree">
              No workspaces opened. Add a local folder or an Azure Blob container.
            </div>
          ) : (
            workspaces.map((ws) => (
              <div key={ws.id} className="workspace">
                <div className="workspace-header">
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
                  <span className="ws-name" title={ws.name}>
                    {ws.name}
                  </span>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => setDatasetDialogWorkspaceId(ws.id)}
                    title={ws.rootPath ? "Download predefined inputs" : "Unavailable for this workspace"}
                    aria-label="Download predefined inputs"
                    disabled={!ws.rootPath}
                  >
                    <ArrowDownload16Regular />
                  </button>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => void handleRunWorkspace(ws)}
                    title={ws.configType ? `Run ${ws.configType}` : "Choose run type"}
                    aria-label="Run workspace"
                    disabled={!ws.rootPath}
                  >
                    <Play16Regular />
                  </button>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => requestCreatePath(ws, "file")}
                    title="Create file in workspace root"
                    aria-label="Create file in workspace root"
                    disabled={!ws.source.canWrite() || !ws.source.createFile}
                  >
                    <Document16Regular />
                  </button>
                  <button
                    className="ws-run ws-icon-btn"
                    onClick={() => requestCreatePath(ws, "directory")}
                    title="Create folder in workspace root"
                    aria-label="Create folder in workspace root"
                    disabled={!ws.source.canWrite() || !ws.source.createDirectory}
                  >
                    <FolderAdd16Regular />
                  </button>
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
                      key={`${ws.id}-${ws.version}`}
                      source={ws.source}
                      nodes={ws.rootNodes}
                      onSelectFile={(n) => handleSelectFile(ws, n)}
                      onCreateFile={(parent) => requestCreatePath(ws, "file", parent)}
                      onCreateFolder={(parent) =>
                        requestCreatePath(ws, "directory", parent)
                      }
                      onDeleteNode={(node) => requestDeletePath(ws, node)}
                      selectedPath={
                        activeView?.type === "file" && activeView.file.workspaceId === ws.id
                          ? activeView.file.node.path
                          : undefined
                      }
                    />
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </aside>

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
                className="btn primary"
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
        ) : (
          <div className="empty-state">
            <h2>No file or job open</h2>
            <p>Open a folder on the left and select a file to view or edit it, or run a job to see its logs.</p>
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
      <CopilotPanel
        open={copilotWizardOpen}
        initialPrompt={"Execution context: ui\n\nRun the benchmark-qed-setup skill end-to-end. Do not display the skill's contents in chat. Go straight into asking me the first question (one at a time) using the skill's ask_user / elicitation flow. Be brief — short prompts only."}
        silentInitialPrompt
        onFolderDetected={(folderPath) => {
          // The agent typically asks for a root path; once we see one and it
          // exists on disk, register it as a workspace so it shows up in the
          // sidebar tree. Dedup against existing workspaces.
          const alreadyAdded = workspaces.some(
            (w) => w.rootPath === folderPath,
          );
          if (!alreadyAdded) {
            void addLocalWorkspace(folderPath);
          }
        }}
        onActivitySettled={() => {
          // Re-list every local workspace so files the agent just created or
          // moved appear in the sidebar without a manual reload.
          for (const ws of workspaces) {
            if (ws.sourceKind === "local") void refreshWorkspace(ws);
          }
        }}
        onLogEvent={(action, details, type) =>
          addActivityLog(action, details, type)
        }
        onClose={() => setCopilotWizardOpen(false)}
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
