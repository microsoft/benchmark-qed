export type SourceKind = "local" | "blob";

export type WorkspaceConfigType =
  | "autoq"
  | "autoe_pairwise"
  | "autoe_reference"
  | "autoe_assertion";

export interface TreeNode {
  name: string;
  path: string; // path relative to workspace root
  kind: "file" | "directory";
  // Opaque handle/ref used by the underlying source.
  ref?: unknown;
}

export type EditorKind =
  | "markdown"
  | "csv"
  | "code"
  | "unsupported";

export interface OpenFile {
  workspaceId: string;
  node: TreeNode;
  kind: EditorKind;
  language: string; // monaco language id
  content: string;
  dirty: boolean;
}

/** Abstraction over a local folder or a remote blob container. */
export interface FileSource {
  readonly kind: SourceKind;
  /** List immediate children of the given node, or root if omitted. */
  listChildren(node?: TreeNode): Promise<TreeNode[]>;
  readFile(node: TreeNode): Promise<string>;
  writeFile(node: TreeNode, content: string): Promise<void>;
  createFile?(path: string, content?: string): Promise<void>;
  createDirectory?(path: string): Promise<void>;
  deletePath?(path: string): Promise<void>;
  /** Rename a file or folder. `newName` is a leaf name (no slashes). */
  renamePath?(path: string, newName: string): Promise<string>;
  canWrite(): boolean;
}

export interface Workspace {
  id: string;
  version: number;
  name: string;
  sourceKind: SourceKind;
  rootPath?: string;
  configType?: WorkspaceConfigType;
  /** Marks a workspace that acts as a parent container for child workspaces. */
  hasChildWorkspaces?: boolean;
  /** Optional parent workspace rootPath for generic workspace nesting. */
  parentRootPath?: string;
  source: FileSource;
  rootNodes: TreeNode[];
  collapsed: boolean;
  /**
   * When this workspace was created by copying another local workspace, the
   * source workspace's rootPath. Used by the sidebar to visually nest copies
   * underneath their original. Stable across reloads because rootPaths are
   * persisted, while in-memory `id`s are regenerated on every session.
   */
  copyOfRootPath?: string;
  /**
   * When true, this workspace exists only to host an externally opened
   * file (e.g. a previously generated quality report). It is NOT shown in
   * the sidebar tree and NOT offered in workspace dropdowns. Such
   * workspaces should also be created with `persisted: undefined` so
   * they vanish on reload.
   */
  hiddenFromSidebar?: boolean;
}
