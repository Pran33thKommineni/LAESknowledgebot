import express from "express";
import cors from "cors";
import { json } from "express";
import { config } from "./config";
import { SyncService } from "./syncService";
import { answerQuestion } from "./qaService";

const app = express();
app.use(
  cors({
    origin: config.port === 4000 ? "http://localhost:5173" : "*",
  }),
);
app.use(json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.post("/admin/sync", async (_req, res) => {
  try {
    const syncService = new SyncService();
    const result = await syncService.syncOnce();
    res.json({ ok: true, ...result });
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error("Sync failed", err);
    res.status(500).json({ ok: false, error: err.message ?? "Sync failed" });
  }
});

app.post("/api/ask", async (req, res) => {
  try {
    const { question } = req.body as { question?: string };
    if (!question || !question.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }
    const answer = await answerQuestion(question);
    return res.json(answer);
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error("QA failed", err);
    return res.status(500).json({ error: err.message ?? "QA failed" });
  }
});

app.listen(config.port, () => {
  // eslint-disable-next-line no-console
  console.log(`Backend listening on port ${config.port}`);
});


