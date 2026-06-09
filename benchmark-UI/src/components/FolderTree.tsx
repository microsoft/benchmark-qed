import { useEffect, useState } from "react";
import {
  Document16Regular,
  DocumentAdd16Regular,
  FolderAdd16Regular,
  Dismiss16Regular,
  ChevronRight16Regular,
  ChevronDown16Regular,
  Copy16Regular,
  Edit16Regular,
} from "@fluentui/react-icons";
import type { FileSource, TreeNode } from "../types";

const EXPANDED_STORAGE_PREFIX = "tree-expanded:";

function expandedStorageKey(workspaceKey: string, path: string): string {
  return `${EXPANDED_STORAGE_PREFIX}${workspaceKey}::${path}`;
}

function readPersistedExpanded(workspaceKey: string, path: string): boolean {
  try {
    return localStorage.getItem(expandedStorageKey(workspaceKey, path)) === "1";
  } catch {
    return false;
  }
}

function writePersistedExpanded(
  workspaceKey: string,
  path: string,
  expanded: boolean,
): void {
  try {
    const key = expandedStorageKey(workspaceKey, path);
    if (expanded) {
      localStorage.setItem(key, "1");
    } else {
      localStorage.removeItem(key);
    }
  } catch {
    // ignore storage errors (quota, private mode, etc.)
  }
}

interface Props {
  source: FileSource;
  nodes: TreeNode[];
  hiddenDirectoryPathIds?: Set<string>;
  onSelectFile: (node: TreeNode) => void;
  selectedPath?: string;
  onCreateFile?: (parent?: TreeNode) => void;
  onCreateFolder?: (parent?: TreeNode) => void;
  onDeleteNode?: (node: TreeNode) => void;
  onCopyNode?: (node: TreeNode) => void;
  onRenameNode?: (node: TreeNode) => void;
  parentNode?: TreeNode;
  workspaceKey: string;
}

export function FolderTree({
  source,
  nodes,
  hiddenDirectoryPathIds,
  onSelectFile,
  selectedPath,
  onCreateFile,
  onCreateFolder,
  onDeleteNode,
  onCopyNode,
  onRenameNode,
  parentNode,
  workspaceKey,
}: Props) {
  const normalizeNodePathId = (path: string): string => {
    const parts = path
      .replace(/\\/g, "/")
      .replace(/^\/+/, "")
      .split("/");
    const stack: string[] = [];
    for (const part of parts) {
      if (!part || part === ".") continue;
      if (part === "..") {
        if (stack.length > 0 && stack[stack.length - 1] !== "..") {
          stack.pop();
        }
        continue;
      }
      stack.push(part);
    }
    return stack.join("/").toLowerCase();
  };

  const visibleNodes =
    hiddenDirectoryPathIds && hiddenDirectoryPathIds.size > 0
      ? nodes.filter((n) => {
          if (n.kind !== "directory") return true;
          return !hiddenDirectoryPathIds.has(normalizeNodePathId(n.path));
        })
      : nodes;

  return (
    <ul className="tree">
      {visibleNodes.map((n) => (
        <TreeItem
          key={n.path}
          node={n}
          source={source}
          onSelectFile={onSelectFile}
          selectedPath={selectedPath}
          onCreateFile={onCreateFile}
          onCreateFolder={onCreateFolder}
          onDeleteNode={onDeleteNode}
          onCopyNode={onCopyNode}
          onRenameNode={onRenameNode}
          parentNode={parentNode}
          hiddenDirectoryPathIds={hiddenDirectoryPathIds}
          workspaceKey={workspaceKey}
        />
      ))}
    </ul>
  );
}

function TreeItem({
  node,
  source,
  onSelectFile,
  selectedPath,
  onCreateFile,
  onCreateFolder,
  onDeleteNode,
  onCopyNode,
  onRenameNode,
  parentNode,
  hiddenDirectoryPathIds,
  workspaceKey,
}: {
  node: TreeNode;
  source: FileSource;
  onSelectFile: (node: TreeNode) => void;
  selectedPath?: string;
  onCreateFile?: (parent?: TreeNode) => void;
  onCreateFolder?: (parent?: TreeNode) => void;
  onDeleteNode?: (node: TreeNode) => void;
  onCopyNode?: (node: TreeNode) => void;
  onRenameNode?: (node: TreeNode) => void;
  parentNode?: TreeNode;
  hiddenDirectoryPathIds?: Set<string>;
  workspaceKey: string;
}) {
  const [expanded, setExpanded] = useState(() =>
    node.kind === "directory"
      ? readPersistedExpanded(workspaceKey, node.path)
      : false,
  );
  const [children, setChildren] = useState<TreeNode[] | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-load children when the node mounts already expanded (restored from
  // localStorage) so the persisted state is reflected on first render.
  useEffect(() => {
    if (node.kind !== "directory" || !expanded || children !== undefined) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    source
      .listChildren(node)
      .then((c) => {
        if (!cancelled) setChildren(c);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        // Always clear the loading flag, even if `cancelled` is true. The
        // cleanup that sets `cancelled` runs synchronously when React re-runs
        // this effect after `setChildren` resolves, but `.finally` is the
        // continuation of that same promise — without this we'd be stuck on
        // "Loading..." forever despite the children being populated.
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [expanded, children, node, source]);

  const toggle = async () => {
    if (node.kind !== "directory") return;
    const next = !expanded;
    setExpanded(next);
    writePersistedExpanded(workspaceKey, node.path, next);
    if (next && !children) {
      setLoading(true);
      setError(null);
      try {
        const c = await source.listChildren(node);
        setChildren(c);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    }
  };

  const isSelected = selectedPath === node.path;
  const isDirectory = node.kind === "directory";
  const createParent = isDirectory ? node : parentNode;

  return (
    <li>
      <div
        className={`tree-row ${isSelected ? "selected" : ""}`}
        onClick={() => {
          if (isDirectory) {
            void toggle();
          } else {
            onSelectFile(node);
          }
        }}
      >
        <span className="tree-icon">
          {isDirectory ? (
            expanded ? (
              <ChevronDown16Regular className="tree-chevron expanded" />
            ) : (
              <ChevronRight16Regular className="tree-chevron" />
            )
          ) : (
            <Document16Regular />
          )}
        </span>
        <span className="tree-name">{node.name}</span>
        {(onCreateFile || onCreateFolder || onDeleteNode || onCopyNode || onRenameNode) && (
          <span className="tree-actions">
            {isDirectory && onCreateFile && (
              <button
                type="button"
                className="tree-action-btn"
                title="Create file"
                aria-label="Create file"
                onClick={(event) => {
                  event.stopPropagation();
                  onCreateFile(createParent);
                }}
              >
                <DocumentAdd16Regular />
              </button>
            )}
            {isDirectory && onCreateFolder && (
              <button
                type="button"
                className="tree-action-btn"
                title="Create folder"
                aria-label="Create folder"
                onClick={(event) => {
                  event.stopPropagation();
                  onCreateFolder(createParent);
                }}
              >
                <FolderAdd16Regular />
              </button>
            )}
            {isDirectory && onCopyNode && (
              <button
                type="button"
                className="tree-action-btn"
                title="Copy folder to another workspace"
                aria-label="Copy folder"
                onClick={(event) => {
                  event.stopPropagation();
                  onCopyNode(node);
                }}
              >
                <Copy16Regular />
              </button>
            )}
            {onRenameNode && (
              <button
                type="button"
                className="tree-action-btn"
                title="Rename"
                aria-label="Rename"
                onClick={(event) => {
                  event.stopPropagation();
                  onRenameNode(node);
                }}
              >
                <Edit16Regular />
              </button>
            )}
            {onDeleteNode && (
              <button
                type="button"
                className="tree-action-btn tree-action-danger"
                title="Delete"
                aria-label="Delete"
                onClick={(event) => {
                  event.stopPropagation();
                  onDeleteNode(node);
                }}
              >
                <Dismiss16Regular />
              </button>
            )}
          </span>
        )}
      </div>
      {isDirectory && expanded && (
        <div className="tree-children">
          {loading && <div className="tree-loading">Loading...</div>}
          {error && <div className="tree-error">{error}</div>}
          {children && (
            <FolderTree
              source={source}
              nodes={children}
              hiddenDirectoryPathIds={hiddenDirectoryPathIds}
              onSelectFile={onSelectFile}
              selectedPath={selectedPath}
              onCreateFile={onCreateFile}
              onCreateFolder={onCreateFolder}
              onDeleteNode={onDeleteNode}
              onCopyNode={onCopyNode}
              onRenameNode={onRenameNode}
              parentNode={node}
              workspaceKey={workspaceKey}
            />
          )}
        </div>
      )}
    </li>
  );
}
