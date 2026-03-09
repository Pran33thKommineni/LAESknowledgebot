// Minimal text extraction stub. For a real deployment you can plug in
// PDF/DOCX/PPTX parsers. For now, treat all files as UTF-8 text where possible.

export function extractTextFromBuffer(buffer: Buffer): string {
  try {
    const text = buffer.toString("utf8");
    return text;
  } catch {
    return "";
  }
}

export function chunkText(
  text: string,
  maxChars = 1500,
  overlap = 200,
): { chunks: string[] } {
  const cleaned = text.replace(/\r\n/g, "\n");
  const chunks: string[] = [];

  let start = 0;
  while (start < cleaned.length) {
    const end = Math.min(start + maxChars, cleaned.length);
    chunks.push(cleaned.slice(start, end));
    if (end === cleaned.length) break;
    start = end - overlap;
    if (start < 0) start = 0;
  }

  return { chunks };
}

