import type { FileSource, TreeNode } from "../types";
import { sortNodes } from "../utils/files";

interface BlobListResponse {
  nodes: TreeNode[];
}

export class RunnerBlobFileSource implements FileSource {
  readonly kind = "blob" as const;
  readonly accountUrl: string;
  readonly containerName: string;
  readonly prefix: string;
  private baseUrl: string;

  constructor(
    accountUrl: string,
    containerName: string,
    prefix: string,
    baseUrl: string,
  ) {
    this.accountUrl = accountUrl.replace(/\/+$/, "");
    this.containerName = containerName.trim();
    this.prefix = prefix.trim().replace(/^\/+|\/+$/g, "");
    this.baseUrl = baseUrl;
  }

  private buildParams(path = ""): URLSearchParams {
    return new URLSearchParams({
      accountUrl: this.accountUrl,
      containerName: this.containerName,
      prefix: this.prefix,
      path,
    });
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
    const params = this.buildParams(node?.path ?? "");
    const res = await this.safeFetch(
      `${this.baseUrl}/api/blob/list?${params.toString()}`,
    );
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to list blobs.");
    }
    const payload = (await res.json()) as BlobListResponse;
    return sortNodes(payload.nodes);
  }

  async readFile(node: TreeNode): Promise<string> {
    const params = this.buildParams(node.path);
    const res = await this.safeFetch(
      `${this.baseUrl}/api/blob/read?${params.toString()}`,
    );
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to read blob.");
    }
    const payload = (await res.json()) as { content?: string };
    return payload.content ?? "";
  }

  async writeFile(node: TreeNode, content: string): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/blob/write`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        accountUrl: this.accountUrl,
        containerName: this.containerName,
        prefix: this.prefix,
        path: node.path,
        content,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to write blob.");
    }
  }

  async createFile(path: string, content = ""): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/blob/create-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        accountUrl: this.accountUrl,
        containerName: this.containerName,
        prefix: this.prefix,
        path,
        content,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to create blob.");
    }
  }

  async deletePath(path: string): Promise<void> {
    const res = await this.safeFetch(`${this.baseUrl}/api/blob/delete`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        accountUrl: this.accountUrl,
        containerName: this.containerName,
        prefix: this.prefix,
        path,
      }),
    });
    if (!res.ok) {
      const payload = (await res.json()) as { error?: string };
      throw new Error(payload.error ?? "Failed to delete blob path.");
    }
  }

  canWrite(): boolean {
    return true;
  }
}