import OpenAI from "openai";
import { db } from "./db";
import { config } from "./config";
import { AnswerSource } from "./types";

const openai = new OpenAI({
  apiKey: config.groqApiKey,
  baseURL: "https://api.groq.com/openai/v1",
});

type Vector = number[];

async function embedText(text: string): Promise<Vector> {
  const response = await openai.embeddings.create({
    model: "nomic-embed-text-v1.5",
    input: text,
  });

  return response.data[0]?.embedding ?? [];
}

export async function generateEmbeddingsForDocument(documentId: string): Promise<void> {
  const chunks = db
    .prepare(
      `SELECT id, text
       FROM chunks
       WHERE document_id = ?`,
    )
    .all(documentId) as { id: string; text: string }[];

  const insertEmbedding = db.prepare(
    `INSERT OR REPLACE INTO embeddings (chunk_id, embedding)
     VALUES (@chunk_id, @embedding)`,
  );

  for (const chunk of chunks) {
    const vector = await embedText(chunk.text);
    insertEmbedding.run({
      chunk_id: chunk.id,
      embedding: JSON.stringify(vector),
    });
  }
}

function cosineSimilarity(a: Vector, b: Vector): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const length = Math.min(a.length, b.length);

  for (let i = 0; i < length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function semanticSearch(
  query: string,
  topK = 8,
): Promise<(AnswerSource & { chunkText: string })[]> {
  const queryEmbedding = await embedText(query);

  const rows = db
    .prepare(
      `SELECT
         e.chunk_id,
         e.embedding,
         c.text as chunk_text,
         c.document_id,
         d.name as document_name,
         d.web_url as web_url
       FROM embeddings e
       JOIN chunks c ON c.id = e.chunk_id
       JOIN documents d ON d.id = c.document_id`,
    )
    .all() as {
    chunk_id: string;
    embedding: string;
    chunk_text: string;
    document_id: string;
    document_name: string;
    web_url: string;
  }[];

  const scored = rows.map((row) => {
    const vector = JSON.parse(row.embedding) as Vector;
    const score = cosineSimilarity(queryEmbedding, vector);
    return { score, row };
  });

  scored.sort((a, b) => b.score - a.score);

  return scored.slice(0, topK).map(({ row }) => ({
    documentId: row.document_id,
    documentName: row.document_name,
    webUrl: row.web_url,
    snippet: row.chunk_text,
    chunkText: row.chunk_text,
  }));
}

