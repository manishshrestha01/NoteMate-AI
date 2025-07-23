# backend/api/index.py

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from pathlib import Path
import shutil
import os
import uuid
import logging
import json
from mangum import Mangum

# LangChain & others
import numpy as np
import fitz  # PyMuPDF
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import GoogleSearchAPIWrapper
from llama_cpp import Llama

# Init app
app = FastAPI()
handler = Mangum(app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
NOTES_DIR = Path("/tmp/generated_notes")
NOTES_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path("/tmp/uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings & vector store
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="/tmp/chroma_db", embedding_function=embeddings)

# Google Search
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID"
Google_Search = GoogleSearchAPIWrapper()

# Local LLM
_llama = None
def get_llama():
    global _llama
    if _llama: return _llama
    _llama = Llama(model_path="/tmp/phi-2.Q4_0.gguf", n_ctx=2048, n_gpu_layers=-1)
    return _llama

# Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# -----------------------------
# Pydantic models
class NoteLink(BaseModel):
    topic: str
    title: str
    snippet: str
    url: str
    similarity_score: float

class NoteLinksResponse(BaseModel):
    notes: List[NoteLink]

class GenerateNotesRequest(BaseModel):
    topics: List[str]
    syllabus_id: Optional[str] = None
    source_type: Literal["syllabus", "internet"] = "syllabus"

# -----------------------------
@app.get("/")
async def root():
    return {"message": "✅ NoteMate AI backend running on Vercel!"}

@app.post("/process-syllabus/")
async def process_syllabus(file: UploadFile = File(...)):
    syllabus_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{syllabus_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text_from_pdf(str(file_path))
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vector_store.add_texts(chunks, metadatas=[{"syllabus_id": syllabus_id}] * len(chunks))
    vector_store.persist()
    return {"message": "Syllabus processed", "syllabus_id": syllabus_id, "chunks": len(chunks)}

@app.post("/generate-notes/", response_model=NoteLinksResponse)
async def generate_notes(request: GenerateNotesRequest):
    if not request.topics:
        raise HTTPException(status_code=400, detail="No topics provided.")
    generated_links = []
    llama = get_llama()

    for topic in request.topics:
        # Search
        search_results = Google_Search.results(topic, num_results=10)
        # Compute similarity
        syllabus_text = topic
        if request.syllabus_id:
            docs = vector_store.similarity_search(topic, k=3, filter={"syllabus_id": request.syllabus_id})
            if docs: syllabus_text = " ".join([d.page_content for d in docs])
        topic_emb = embeddings.embed_query(syllabus_text)
        scored = []
        for res in search_results:
            txt = res.get('title', '') + " " + res.get('snippet', '')
            emb = embeddings.embed_query(txt)
            score = cosine_similarity(topic_emb, emb)
            scored.append((score, res))
        # Sort & validate with LLM
        scored.sort(reverse=True, key=lambda x: x[0])
        top = []
        for score, res in scored[:10]:
            snippet = res.get('snippet', '')
            prompt = f"""
Is this snippet useful for topic '{topic}'? Answer yes or no.

"{snippet}"
"""
            answer = llama(prompt, stop=["```", "</s>"]).strip().lower()
            if "yes" in answer:
                top.append(NoteLink(
                    topic=topic,
                    title=res.get('title', ''),
                    snippet=snippet,
                    url=res.get('link', ''),
                    similarity_score=round(score, 3)
                ))
            if len(top) >= 3: break
        # Fallback if not enough
        if len(top) < 3:
            for score, res in scored:
                top.append(NoteLink(
                    topic=topic,
                    title=res.get('title', ''),
                    snippet=res.get('snippet', ''),
                    url=res.get('link', ''),
                    similarity_score=round(score, 3)
                ))
                if len(top) >= 3: break
        generated_links.extend(top)
    return NoteLinksResponse(notes=generated_links)
