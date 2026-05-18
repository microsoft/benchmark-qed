import type { FileSource, TreeNode } from "../types";
import { sortNodes } from "../utils/files";

interface LocalRef {
  handle: FileSystemFileHandle | FileSystemDirectoryHandle;
}

export class LocalFileSource implements FileSource {
  readonly kind = "local" as const;
  private root: FileSystemDirectoryHandle;

  constructor(root: FileSystemDirectoryHandle) {
    this.root = root;
  }

  async listChildren(node?: TreeNode): Promise<TreeNode[]> {
    const dir = node
      ? ((node.ref as LocalRef).handle as FileSystemDirectoryHandle)
      : this.root;
    const basePath = node?.path ?? "";
    const entries: TreeNode[] = [];
    const values = (
      dir as unknown as {
        values: () => AsyncIterable<
          FileSystemFileHandle | FileSystemDirectoryHandle
        >;
      }
    ).values();
    for await (const handle of values) {
      entries.push({
        name: handle.name,
        path: basePath ? `${basePath}/${handle.name}` : handle.name,
        kind: handle.kind,
        ref: { handle } satisfies LocalRef,
      });
    }
    return sortNodes(entries);
  }

  async readFile(node: TreeNode): Promise<string> {
    const handle = (node.ref as LocalRef).handle as FileSystemFileHandle;
    const file = await handle.getFile();
    return file.text();
  }

  async writeFile(node: TreeNode, content: string): Promise<void> {
    const handle = (node.ref as LocalRef).handle as FileSystemFileHandle & {
      createWritable: () => Promise<{
        write(data: string): Promise<void>;
        close(): Promise<void>;
      }>;
    };
    const writable = await handle.createWritable();
    await writable.write(content);
    await writable.close();
  }

  canWrite(): boolean {
    return true;
  }
}
