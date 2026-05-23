import { ContainerClient } from "@azure/storage-blob";
import type { FileSource, TreeNode } from "../types";
import { sortNodes } from "../utils/files";

export class BlobFileSource implements FileSource {
  readonly kind = "blob" as const;
  private client: ContainerClient;
  private writable: boolean;
  private deletable: boolean;
  private prefix: string;

  constructor(sasUrl: string, prefix = "") {
    this.client = new ContainerClient(sasUrl);
    const sp = new URL(sasUrl).searchParams.get("sp") ?? "";
    this.writable = /w/i.test(sp) || /c/i.test(sp);
    this.deletable = /d/i.test(sp);
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

  /**
   * Delete a blob or, if the path corresponds to a virtual folder, every
   * blob beneath that prefix. Azure has no real folders — they only exist
   * as common prefixes — so removing a "directory" means iterating and
   * deleting each underlying blob.
   *
   * Requires `sp=d` on the SAS and `DELETE` in the storage account's CORS
   * allowed methods. The most actionable Azure errors are re-thrown
   * verbatim so the UI banner can show them.
   */
  async deletePath(path: string): Promise<void> {
    if (!this.deletable) {
      throw new Error(
        "This blob SAS does not allow delete (need 'sp=d' and 'DELETE' in CORS).",
      );
    }
    const blobName = `${this.prefix}${path}`;

    // First try as a single blob. `deleteIfExists` returns `succeeded`
    // false when the blob does not exist, which is how we detect that the
    // path is actually a folder prefix.
    const single = this.client.getBlobClient(blobName);
    try {
      const res = await single.deleteIfExists();
      if (res.succeeded) return;
    } catch (err) {
      // Real Azure errors (auth, CORS-after-the-fact) bubble up. A 404 is
      // already swallowed by deleteIfExists.
      throw err;
    }

    // Treat as a folder: iterate all blobs whose name starts with the
    // path + "/" and delete them one by one. Azure's `flat` listing
    // returns descendants at any depth, so this also covers nested files.
    const folderPrefix = blobName.endsWith("/") ? blobName : `${blobName}/`;
    let deleted = 0;
    for await (const item of this.client.listBlobsFlat({
      prefix: folderPrefix,
    })) {
      await this.client.getBlobClient(item.name).deleteIfExists();
      deleted++;
    }
    if (deleted === 0) {
      // Nothing matched — the path didn't refer to anything in the
      // container. Treat as no-op (consistent with `deleteIfExists`).
    }
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
