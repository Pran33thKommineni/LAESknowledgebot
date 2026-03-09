import OpenAI from "openai";
import { config } from "./config";
import { Answer } from "./types";
import { semanticSearch } from "./embeddingService";

const openai = new OpenAI({
  apiKey: config.groqApiKey,
  baseURL: "https://api.groq.com/openai/v1",
});

export async function answerQuestion(question: string): Promise<Answer> {
  const matches = await semanticSearch(question, 6);

  const contextBlocks = matches
    .map(
      (m, idx) =>
        `Source ${idx + 1} - ${m.documentName} (${m.webUrl}):\n${m.snippet}`,
    )
    .join("\n\n");

  const systemPrompt =
    "You are a helpful assistant for the LAES enhanced technical studio. " +
    "Answer the user's question using ONLY the provided sources. " +
    "If the sources do not contain the answer, say you do not know rather than guessing.";

  const userPrompt = `Question:\n${question}\n\nSources:\n${contextBlocks}\n\nAnswer in a clear, concise way and refer to specific sources when relevant.`;

  const completion = await openai.chat.completions.create({
    model: "llama-3.1-8b-instant",
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
  });

  const text = completion.choices[0]?.message?.content ?? "";

  return {
    text,
    sources: matches.map((m) => ({
      documentId: m.documentId,
      documentName: m.documentName,
      webUrl: m.webUrl,
      snippet: m.snippet,
    })),
  };
}

