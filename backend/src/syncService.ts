import { randomUUID } from "crypto";
import fs from "fs";
import path from "path";
import { db } from "./db";
import { config } from "./config";
import { extractTextFromBuffer, chunkText } from "./textExtraction";
import { generateEmbeddingsForDocument } from "./embeddingService";

export class SyncService {
  async syncOnce(): Promise<{ documentsSynced: number }> {
    if (!config.localLaesFolder) {
      throw new Error(
        "LOCAL_LAES_FOLDER is not configured. Set it in backend/.env to a local folder synced from OneDrive.",
      );
    }

    const root = config.localLaesFolder;
    if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
      throw new Error(`LOCAL_LAES_FOLDER does not exist or is not a directory: ${root}`);
    }

    const files = this.walkFolder(root);

    let count = 0;
    for (const filePath of files) {
      const relPath = path.relative(root, filePath);
      if (!this.isSupportedFile(filePath)) continue;
      // Use the full path as a stable drive_item_id surrogate.
      await this.ingestLocalFile(filePath, relPath);
      count += 1;
    }

    return { documentsSynced: count };
  }

  private walkFolder(root: string): string[] {
    const results: string[] = [];

    const entries = fs.readdirSync(root, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(root, entry.name);
      if (entry.isDirectory()) {
        results.push(...this.walkFolder(fullPath));
      } else if (entry.isFile()) {
        results.push(fullPath);
      }
    }

    return results;
  }

  private isSupportedFile(filePath: string): boolean {
    const lower = filePath.toLowerCase();
    return [".pdf", ".docx", ".pptx", ".txt", ".md"].some((ext) => lower.endsWith(ext));
  }

  private async ingestLocalFile(fullPath: string, relativePath: string): Promise<void> {
    const stat = fs.statSync(fullPath);
    const name = path.basename(fullPath);
    const driveItemId = fullPath;
    const webUrl = fullPath; // For now, just show the local path as a reference.
    const size = stat.size;
    const lastModified = stat.mtime.toISOString();
    const parentPath = path.dirname(relativePath);

    const existing = db
      .prepare("SELECT id, last_modified FROM documents WHERE drive_item_id = ?")
      .get(driveItemId) as { id: string; last_modified: string } | undefined;

    if (existing && existing.last_modified === lastModified) {
      return;
    }

    const buffer = fs.readFileSync(fullPath);
    const text = extractTextFromBuffer(buffer);
    const { chunks } = chunkText(text);

    const docId = existing?.id ?? randomUUID();

    const tx = db.transaction(() => {
      db.prepare(
        `
        INSERT INTO documents (id, drive_item_id, name, path, web_url, last_modified, size)
        VALUES (@id, @drive_item_id, @name, @path, @web_url, @last_modified, @size)
        ON CONFLICT(id) DO UPDATE SET
          drive_item_id = excluded.drive_item_id,
          name = excluded.name,
          path = excluded.path,
          web_url = excluded.web_url,
          last_modified = excluded.last_modified,
          size = excluded.size
      `,
      ).run({
        id: docId,
        drive_item_id: driveItemId,
        name,
        path: parentPath,
        web_url: webUrl,
        last_modified: lastModified,
        size,
      });

      db.prepare("DELETE FROM chunks WHERE document_id = ?").run(docId);
      db.prepare(
        "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
      ).run(docId);

      const insertChunk = db.prepare(
        `INSERT INTO chunks (id, document_id, chunk_index, text, location)
         VALUES (@id, @document_id, @chunk_index, @text, @location)`,
      );

      chunks.forEach((chunkTextValue, index) => {
        insertChunk.run({
          id: randomUUID(),
          document_id: docId,
          chunk_index: index,
          text: chunkTextValue,
          location: "",
        });
      });
    });

    tx();

    if (text.trim().length > 0) {
      await generateEmbeddingsForDocument(docId);
    }
  }
}

