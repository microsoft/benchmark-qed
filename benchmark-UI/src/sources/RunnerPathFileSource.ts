import type { FileSource, TreeNode } from "../types";
import { sortNodes } from "../utils/files";

interface FsListResponse {
  root?: string;
  path?: string;
  nodes: TreeNode[];
}

export class RunnerPathFileSource implements FileSource {
  readonly kind = "local" as const;
  private rootPath: string;
  private baseUrl: string;
  private resolvedRootPath?: string;

  constructor(rootPath: string, baseUrl: string) {
    this.rootPath = rootPath;
    this.baseUrl = baseUrl;
  }

  getResolvedRootPath(): string | undefined {
    return this.resolvedRootPath;
  }

  private async safeFetch(url: string, init?: RequestInit): Promise<Response> {
    try {
      return await fetch(url, init);
    } catch {
      throw new Error(
        `Cannot reach local runner at ${this.baseUrl}. Start it with 'npm run init-runner'.`,
      );
    }
  }

  async listChildren(node?: TreeNode): Promise<TreeNode[]> {
    const relPath = node?.path ?? "";
    const params = new URLSearchParams({
      root: this.rootPath,
      path: relPath,
    });
    const res = await this.safeFetch(
      `${this.baseUrl}/api/fs/list?${params.toString()}`,
    );
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to list files.");
    }
    const payload = (await res.json()) as FsListResponse;
    if (typeof payload.root === "string" && payload.root.trim()) {
      this.resolvedRootPath = payload.root;
      this.rootPath = payload.root;
    }
    return sortNodes(payload.nodes);
  }

  async readFile(node: TreeNode): Promise<string> {
    const params = new URLSearchParams({
      root: this.rootPath,
      path: node.path,
    });
    const res = await this.safeFetch(
      `${this.baseUrl}/api/fs/read?${params.toString()}`,
    );
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to read file.");
    }
    const payload = (await res.json()) as { content?: string };
    return payload.content ?? "";
  }

  async writeFile(node: TreeNode, content: string): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/fs/write`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: this.rootPath,
        path: node.path,
        content,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to write file.");
    }
  }

  async createFile(path: string, content = ""): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/fs/create-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: this.rootPath,
        path,
        content,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to create file.");
    }
  }

  async createDirectory(path: string): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/fs/create-folder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: this.rootPath,
        path,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to create folder.");
    }
  }

  async deletePath(path: string): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/fs/delete`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: this.rootPath,
        path,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to delete path.");
    }
  }

  async renamePath(path: string, newName: string): Promise<string> {
    const res = await this.safeFetch(`${this.baseUrl}/api/fs/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        root: this.rootPath,
        path,
        newName,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to rename path.");
    }
    const payload = (await res.json()) as { path?: string };
    return payload.path ?? newName;
  }

  canWrite(): boolean {
    return true;
  }
}
