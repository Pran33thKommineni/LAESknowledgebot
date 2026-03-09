export interface DocumentRecord {
  id: string;
  driveItemId: string;
  name: string;
  path: string;
  webUrl: string;
  lastModified: string;
  size: number;
}

export interface ChunkRecord {
  id: string;
  documentId: string;
  chunkIndex: number;
  text: string;
  location?: string;
}

export interface AnswerSource {
  documentId: string;
  documentName: string;
  webUrl: string;
  snippet: string;
}

export interface Answer {
  text: string;
  sources: AnswerSource[];
}

