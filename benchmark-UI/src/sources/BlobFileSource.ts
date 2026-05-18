import { ContainerClient } from "@azure/storage-blob";
import type { FileSource, TreeNode } from "../types";
import { sortNodes } from "../utils/files";

export class BlobFileSource implements FileSource {
  readonly kind = "blob" as const;
  private client: ContainerClient;
  private writable: boolean;
  private prefix: string;

  constructor(sasUrl: string, prefix = "") {
    this.client = new ContainerClient(sasUrl);
    const sp = new URL(sasUrl).searchParams.get("sp") ?? "";
    this.writable = /w/i.test(sp) || /c/i.test(sp);
    this.prefix = normalizePrefix(prefix);
  }

  private fullPath(node?: TreeNode): string {
    const rel = node?.path ?? "";
    const joined = `${this.prefix}${rel}`;
    return joined === "" ? "" : joined.endsWith("/") ? joined : `${joined}/`;
  }

  async listChildren(node?: TreeNode): Promise<TreeNode[]> {
    const fullPrefix = this.fullPath(node);
    const basePath = node?.path ?? "";
    const entries: TreeNode[] = [];
    const iter = this.client.listBlobsByHierarchy("/", { prefix: fullPrefix });
    for await (const item of iter) {
      if (item.kind === "prefix") {
        const full = item.name; // e.g. "<prefix>/foo/"
        const trimmed = full.endsWith("/") ? full.slice(0, -1) : full;
        const name = trimmed.slice(fullPrefix.length);
        if (!name) continue;
        entries.push({
          name,
          path: basePath ? `${basePath}/${name}` : name,
          kind: "directory",
        });
      } else {
        const full = item.name;
        const name = full.slice(fullPrefix.length);
        if (!name) continue;
        entries.push({
          name,
          path: basePath ? `${basePath}/${name}` : name,
          kind: "file",
        });
      }
    }
    return sortNodes(entries);
  }

  async readFile(node: TreeNode): Promise<string> {
    const blobName = `${this.prefix}${node.path}`;
    const blob = this.client.getBlobClient(blobName);
    const resp = await blob.download();
    const body = await resp.blobBody;
    if (!body) return "";
    return body.text();
  }

  async writeFile(node: TreeNode, content: string): Promise<void> {
    if (!this.writable) {
      throw new Error("This workspace is read-only.");
    }
    const blobName = `${this.prefix}${node.path}`;
    const blob = this.client.getBlockBlobClient(blobName);
    await blob.upload(content, new Blob([content]).size, {
      blobHTTPHeaders: { blobContentType: "text/plain; charset=utf-8" },
    });
  }

  canWrite(): boolean {
    return this.writable;
  }
}

function normalizePrefix(p?: string): string {
  if (!p) return "";
  if (p.endsWith("/")) return p;
  return `${p}/`;
}
