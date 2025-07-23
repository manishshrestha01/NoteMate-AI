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

# Import your other modules (make sure paths work if you have modules like embeddings, vector_store, etc.)

# Initialize FastAPI app
app = FastAPI(
    title="NoteMate Backend - Syllabus & Notes Processor",
    description="API for uploading and processing PDF syllabuses, and generating study notes.",
    version="0.1.0"
)

# Enable CORS (optional; adjust origins)
origins = [
    "*",  # replace with your frontend domain for production
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to save generated notes
NOTES_DIR = Path("generated_notes")
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Define your Pydantic models (simplified)
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

# --------------------------
# Dummy / placeholder endpoints to test deployment
@app.get("/")
async def root():
    return {"message": "✅ NoteMate Backend running on Vercel + FastAPI + Mangum!"}

@app.post("/generate-notes/", response_model=NoteLinksResponse)
async def generate_notes(request: GenerateNotesRequest):
    """
    Dummy version: always returns 3 fake links for each topic.
    Replace this with real logic: embeddings, Google Search, cosine similarity, etc.
    """
    generated_links = []
    for topic in request.topics:
        generated_links.extend([
            NoteLink(
                topic=topic,
                title=f"Example Note Title 1 for {topic}",
                snippet=f"Snippet about {topic}.",
                url=f"https://example.com/{topic}-1",
                similarity_score=0.95
            ),
            NoteLink(
                topic=topic,
                title=f"Example Note Title 2 for {topic}",
                snippet=f"Another snippet about {topic}.",
                url=f"https://example.com/{topic}-2",
                similarity_score=0.93
            ),
            NoteLink(
                topic=topic,
                title=f"Example Note Title 3 for {topic}",
                snippet=f"Yet another snippet about {topic}.",
                url=f"https://example.com/{topic}-3",
                similarity_score=0.90
            )
        ])
    return NoteLinksResponse(notes=generated_links)

@app.post("/process-syllabus/")
async def process_syllabus(file: UploadFile = File(...)):
    """
    Dummy upload handler: just saves file temporarily and returns fake syllabus_id.
    Replace with your real PDF processing, vector store logic, and extraction.
    """
    syllabus_id = str(uuid.uuid4())
    tmp_path = Path(f"/tmp/{syllabus_id}_{file.filename}")
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved file to {tmp_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()  # clean up
    return JSONResponse({
        "message": "File processed (dummy)",
        "filename": file.filename,
        "syllabus_id": syllabus_id,
        "units_data": []  # placeholder
    })

# --------------------------
# Mangum handler for Vercel / Lambda
handler = Mangum(app)
