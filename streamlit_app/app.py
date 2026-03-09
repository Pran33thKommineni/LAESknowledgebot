import io
import math
import os
from dataclasses import dataclass
from typing import List

import openai
import streamlit as st
from docx import Document as DocxDocument
from pypdf import PdfReader


@dataclass
class Chunk:
  text: str
  source_name: str


@dataclass
class EmbeddedChunk(Chunk):
  embedding: List[float]


def get_groq_client() -> openai.OpenAI:
  # Prefer Streamlit secrets in cloud, fall back to env locally.
  api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
  if not api_key:
    api_key = os.getenv("GROQ_API_KEY", "")

  if not api_key:
    st.stop()

  client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
  )
  return client


def extract_text_from_file(upload) -> str:
  name = upload.name.lower()
  data = upload.read()

  if name.endswith(".txt") or name.endswith(".md"):
    return data.decode("utf-8", errors="ignore")

  if name.endswith(".pdf"):
    pdf = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)

  if name.endswith(".docx"):
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

  # Fallback: try utf-8
  return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
  cleaned = text.replace("\r\n", "\n")
  chunks: List[str] = []
  start = 0
  while start < len(cleaned):
    end = min(start + max_chars, len(cleaned))
    chunks.append(cleaned[start:end])
    if end == len(cleaned):
      break
    start = max(0, end - overlap)
  return chunks


def embed_texts(client: openai.OpenAI, texts: List[str]) -> List[List[float]]:
  if not texts:
    return []
  resp = client.embeddings.create(
    model="nomic-embed-text-v1.5",
    input=texts,
  )
  return [item.embedding for item in resp.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
  length = min(len(a), len(b))
  dot = 0.0
  na = 0.0
  nb = 0.0
  for i in range(length):
    ai = a[i]
    bi = b[i]
    dot += ai * bi
    na += ai * ai
    nb += bi * bi
  if na == 0 or nb == 0:
    return 0.0
  return dot / (math.sqrt(na) * math.sqrt(nb))


def answer_question(client: openai.OpenAI, question: str, embedded_chunks: List[EmbeddedChunk]) -> str:
  if not embedded_chunks:
    return "I don't have any documents yet. Please upload some LAES materials first."

  query_emb = embed_texts(client, [question])[0]

  scored = [
    (cosine_similarity(query_emb, ch.embedding), ch)
    for ch in embedded_chunks
  ]
  scored.sort(key=lambda x: x[0], reverse=True)
  top = [c for score, c in scored[:6] if score > 0]

  if not top:
    return "I couldn't find anything relevant in the uploaded documents."

  context = []
  for idx, ch in enumerate(top, start=1):
    context.append(f"Source {idx} - {ch.source_name}:\n{ch.text}")
  context_block = "\n\n".join(context)

  system = (
    "You are a helpful assistant for the LAES enhanced technical studio. "
    "Use ONLY the provided sources to answer the question. "
    "If the sources don't contain the answer, say you don't know."
  )
  user = f"Question:\n{question}\n\nSources:\n{context_block}\n\nAnswer clearly and concisely."

  chat = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
  )
  return chat.choices[0].message.content or ""


def main() -> None:
  st.set_page_config(
    page_title="LAES Knowledge Bot (Streamlit)",
    page_icon="🎓",
    layout="wide",
  )

  st.title("LAES Knowledge Bot – Streamlit")
  st.markdown(
    "Upload LAES enhanced technical studio materials (PDF, DOCX, TXT, MD) "
    "and ask questions without digging through folders."
  )

  client = get_groq_client()

  if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks: List[EmbeddedChunk] = []

  with st.sidebar:
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
      "Drag in LAES materials",
      type=["pdf", "docx", "txt", "md"],
      accept_multiple_files=True,
    )
    if st.button("Process uploads") and uploaded_files:
      all_chunks: List[Chunk] = []
      for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text.strip():
          continue
        for ch in chunk_text(text):
          all_chunks.append(Chunk(text=ch, source_name=f.name))

      with st.spinner("Embedding chunks with Groq..."):
        embeddings = embed_texts(client, [c.text for c in all_chunks])
        st.session_state.embedded_chunks = [
          EmbeddedChunk(text=c.text, source_name=c.source_name, embedding=emb)
          for c, emb in zip(all_chunks, embeddings)
        ]
      st.success(f"Processed {len(st.session_state.embedded_chunks)} chunks from {len(uploaded_files)} file(s).")

    st.markdown("---")
    st.caption(
      "Tip: For a class demo on Streamlit Cloud, put `GROQ_API_KEY` in `st.secrets` "
      "and push this `streamlit_app` folder to GitHub."
    )

  col_left, col_right = st.columns([2, 1])

  with col_left:
    st.subheader("Ask a question")
    question = st.text_input("Question about your LAES materials")
    if st.button("Ask") and question.strip():
      with st.spinner("Thinking with your documents..."):
        answer = answer_question(client, question.strip(), st.session_state.embedded_chunks)
      st.markdown("#### Answer")
      st.write(answer)

  with col_right:
    st.subheader("Loaded sources")
    if st.session_state.embedded_chunks:
      names = sorted({c.source_name for c in st.session_state.embedded_chunks})
      st.write(f"{len(names)} file(s), {len(st.session_state.embedded_chunks)} chunks.")
      for n in names:
        st.markdown(f"- {n}")
    else:
      st.write("No documents loaded yet.")


if __name__ == "__main__":
  main()

