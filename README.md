# LAES Knowledge Bot

A small full-stack app that lets you ask questions about your LAES enhanced technical studio materials, without manually browsing folders.

## Architecture

- **Frontend**: React + TypeScript (Vite) single-page app in `frontend/`.
- **Backend**: Node + TypeScript (Express) API in `backend/`.
- **Storage**: SQLite database (`better-sqlite3`) storing documents, text chunks, and embeddings.
- **Integrations**:
  - Local filesystem sync (pointing at a folder that can be synced from OneDrive).
  - Groq (OpenAI-compatible API) for embeddings and question answering.

## Setup

1. **Backend**

   ```bash
   cd backend
   cp .env.example .env
   # Fill in your Groq API key and LOCAL_LAES_FOLDER path
   npm install
   npm run dev
   ```

2. **Frontend**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Usage flow**

- Open the frontend (Vite will print the URL, typically `http://localhost:5173`).
- Click **"Sync OneDrive"** (now syncing from your `LOCAL_LAES_FOLDER`) to ingest documents.
- Ask questions in the chat; the bot retrieves relevant chunks and answers with citations back to the original files.

