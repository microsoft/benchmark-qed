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
  canWrite(): boolean;
}

export interface Workspace {
  id: string;
  version: number;
  name: string;
  sourceKind: SourceKind;
  rootPath?: string;
  configType?: WorkspaceConfigType;
  source: FileSource;
  rootNodes: TreeNode[];
  collapsed: boolean;
}
