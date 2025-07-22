from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict, Any, Literal
import os
import shutil
import magic
from pathlib import Path
import logging
import re
import json
import uuid
import numpy as np # Added for cosine_similarity
import sys # Added for error handling insights

from transformers import AutoTokenizer # Unused but kept if you plan to use it

# Document Loaders for text extraction
import fitz  # PyMuPDF for better PDF text extraction

# LangChain components for text splitting, embeddings, vector store, LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import GoogleSearchAPIWrapper # Added missing import

# TinyLlama for syllabus extraction
from llama_cpp import Llama
from functools import lru_cache # For singleton model loading

import requests  # For Ollama API call

NOTES_DIR = Path("generated_notes")
NOTES_DIR.mkdir(parents=True, exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for AI Extraction Output (Syllabus Structure) ---
class SubUnit(BaseModel):
    title: str = Field(description="The title of the sub-unit, e.g., '1.1 Types', '2.3 Interprocess Communication'.")
    content: str = Field(description="The full textual content of this sub-unit.")

class Unit(BaseModel):
    title: str = Field(description="The title of the main unit, e.g., 'Unit 1: Introduction', 'Chapter 2: Processes'.")
    content: str = Field(description="The full textual content of this main unit, excluding content explicitly within its sub-units. Can be empty if all content is within sub-units.")
    sub_units: List[SubUnit] = Field(default_factory=list, description="A list of sub-units within this main unit.")

class SyllabusStructure(BaseModel):
    units_data: List[Unit] = Field(description="A list of main units found in the syllabus, each with its title, content, and nested sub-units.")

# --- Pydantic Schemas for Generated Notes Link Output ---
class NoteLink(BaseModel):
    topic: str
    title: str
    snippet: str
    url: str
    similarity_score: float

class NoteLinksResponse(BaseModel):
    notes: List[NoteLink]

# --- Pydantic Schema for Generate Notes Request ---
class GenerateNotesRequest(BaseModel):
    topics: List[str] = Field(description="List of topic titles for which to generate notes.")
    syllabus_id: Optional[str] = Field(
        None,
        description="The unique ID of the syllabus to retrieve context from. Required if source_type is 'syllabus'."
    )
    source_type: Literal["syllabus", "internet"] = Field(
        "syllabus",
        description="Specifies where to find the notes: 'syllabus' (from uploaded PDF) or 'internet' (using Google Search)."
    )

# --- Settings Management ---
class Settings(BaseSettings):
    chroma_db_path: str = "./chroma_db"
    supported_file_types: List[str] = ["application/pdf"]
    upload_dir: str = "uploaded_files"

    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Ensure directories exist
try:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured '{settings.upload_dir}' and '{settings.chroma_db_path}' directories exist.")
except OSError as e:
    logger.critical(f"Failed to create necessary directories: {e}. Please check permissions.")
    exit(1)

# --- LangChain Components Initialization ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("SentenceTransformerEmbeddings model 'all-MiniLM-L6-v2' initialized.")

vector_store = Chroma(persist_directory=settings.chroma_db_path, embedding_function=embeddings)
logger.info(f"ChromaDB initialized, persistent directory: {settings.chroma_db_path}")

# --- Initialize Google Search Tool ---
Google_Search = None
if settings.google_api_key and settings.google_cse_id:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["GOOGLE_CSE_ID"] = settings.google_cse_id
    try:
        Google_Search = GoogleSearchAPIWrapper()
        logger.info("Google Search API Wrapper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Google Search API Wrapper: {e}. Check GOOGLE_API_KEY and GOOGLE_CSE_ID.", exc_info=True)
        Google_Search = None
else:
    logger.warning("Google API Key or CSE ID missing. Internet search functionality will be unavailable.")

app = FastAPI(
    title="NoteMate Backend - Syllabus & Notes Processor",
    description="API for uploading and processing PDF syllabuses, and generating study notes from selected chapters using AI.",
    version="0.1.0"
)

# Configure CORS to allow requests from your React frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def get_file_type(file_path: str) -> str:
    return magic.from_file(file_path, mime=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path} with fitz: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {e}")
    return text

def preprocess_syllabus_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    merged_lines = []
    buffer = ""
    for line in lines:
        if not line.strip():
            if buffer:
                merged_lines.append(buffer)
                buffer = ""
            continue
        # If buffer is not empty and current line is a continuation (doesn't start with Unit/Chapter/Module/number)
        if buffer and not re.match(r'^(Unit|Chapter|Module|\d+\.\d+)', line.strip(), re.IGNORECASE):
            buffer += " " + line.strip()
        else:
            if buffer:
                merged_lines.append(buffer)
            buffer = line.strip()
    if buffer:
        merged_lines.append(buffer)
    return "\n".join(merged_lines)

# Singleton loader for TinyLlama
_llama_instance: Optional[Llama] = None

@lru_cache(maxsize=1)
def get_llama_model() -> Optional[Llama]:
    global _llama_instance
    if _llama_instance is not None:
        return _llama_instance

    # Only use Phi-2
    phi2_path = str(Path(__file__).parent / "models/phi-2.Q4_0.gguf")
    if Path(phi2_path).exists():
        model_path = phi2_path
        logger.info(f"Using Phi-2 model for extraction: {model_path}")
    else:
        logger.critical(f"Phi-2 model not found in models/. Please add phi-2.Q4_0.gguf.")
        return None

    try:
        _llama_instance = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=True,
        )
        logger.info(f"Model loaded successfully from {model_path}.")
        return _llama_instance
    except Exception as e:
        logger.critical(f"Failed to load model: {e}", exc_info=True)
        _llama_instance = None
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

#Hero of the code: extract_syllabus_structure_with_local_model
def extract_syllabus_structure_with_local_model(syllabus_text: str, llm_instance: Optional[Llama]) -> SyllabusStructure:
    """
    Hybrid extraction: regex to get units/sub-units,
    and remove unwanted trailing sections like 'List of Tutorials', etc.
    Uses local LLM fallback if available.
    """
    logger.info("Starting hybrid syllabus extraction (regex + cleaning unwanted sections).")

    # Define sections after which we should cut
    unwanted_patterns = [
        "List of Tutorials", "List of Practical", "Evaluation System and Students’ Responsibilities",
        "Evaluation System", "Students’ Responsibilities", "Prescribed Books and References",
        "Text Books", "References", "Practical Works", "Practical Work", "Practical"
    ]

    # Step 1: Cut text if unwanted section appears
    for pattern in unwanted_patterns:
        idx = syllabus_text.lower().find(pattern.lower())
        if idx != -1:
            syllabus_text = syllabus_text[:idx]
            break

    # Step 2: Find units
    unit_regex = re.compile(r'^(Unit|Module|Chapter)\s+[\dIVXLCDM]+.*$', re.MULTILINE | re.IGNORECASE)
    unit_matches = list(unit_regex.finditer(syllabus_text))

    units = []

    for idx, match in enumerate(unit_matches):
        unit_title = match.group().strip()
        start = match.end()
        end = unit_matches[idx + 1].start() if idx + 1 < len(unit_matches) else len(syllabus_text)
        unit_content = syllabus_text[start:end].strip()

        # Remove trailing unwanted content in unit_content too
        for pattern in unwanted_patterns:
            cut = unit_content.lower().find(pattern.lower())
            if cut != -1:
                unit_content = unit_content[:cut].strip()
                break

        # Step 3: Find sub-units
        subunit_regex = re.compile(r'^\s*(\d+\.\d+)\s*(.+)', re.MULTILINE)
        sub_units = []
        for sub_match in subunit_regex.finditer(unit_content):
            sub_number = sub_match.group(1).strip()
            sub_title_text = sub_match.group(2).strip()
            # Optional clean: remove trailing commentary like ● Understand
            sub_title_clean = re.split(r'●|•|- Understand|- Review', sub_title_text)[0].strip()
            sub_units.append(SubUnit(title=f"{sub_number} {sub_title_clean}", content=""))  # Only title, empty content

        units.append(Unit(title=unit_title, content="", sub_units=sub_units))

    logger.info(f"Regex extracted {len(units)} units.")

    # Step 4: fallback / optional refinement with local LLM
    if llm_instance:
        try:
            prompt = f"""
Extract only the syllabus structure as JSON.
Ignore sections after:
- List of Tutorials
- List of Practical
- Evaluation System
- Students’ Responsibilities
- Prescribed Books and References

Return format:
{{ "units_data": [ {{ "title": "...", "content": "", "sub_units": [ {{ "title": "...", "content": "" }} ] }} ] }}

Text:
---
{syllabus_text}
---
"""
            output = llm_instance(prompt, stop=["```", "</s>"])
            data = json.loads(output)
            units_data = data.get("units_data", [])

            if units_data:
                refined = [
                    Unit(
                        title=u.get("title", ""),
                        content=u.get("content", ""),
                        sub_units=[
                            SubUnit(title=s.get("title", ""), content=s.get("content", ""))
                            for s in u.get("sub_units", [])
                        ],
                    )
                    for u in units_data
                ]
                logger.info(f"Local LLM extracted {len(refined)} refined units.")
                return SyllabusStructure(units_data=refined)
        except Exception as e:
            logger.warning(f"Local LLM extraction failed: {e}")

    logger.info("Using regex extracted syllabus structure as fallback.")
    return SyllabusStructure(units_data=units)


def process_document_and_add_to_chroma(file_path: Path, filename: str, syllabus_id: str):
    file_type = get_file_type(str(file_path))
    logger.info(f"Processing file: {filename} with detected type: {file_type}")

    if file_type not in settings.supported_file_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_type}. Only PDF files are supported."
        )

    raw_text = extract_text_from_pdf(str(file_path))
    clean_text = preprocess_syllabus_text(raw_text)

    if not raw_text.strip():
        logger.warning(f"Extracted no meaningful text from {filename}. Skipping further processing.")
        return {"message": "File processed, but no significant text was extracted.", "filename": filename, "units_data": [], "syllabus_id": syllabus_id}

    logger.info(f"Raw text extracted from {filename} (first 500 chars):\n{raw_text[:500]}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_text(raw_text)
    logger.info(f"Split {filename} into {len(texts)} chunks for vector store.")

    try:
        metadatas = [{"syllabus_id": syllabus_id, "source_filename": filename} for _ in texts]
        vector_store.add_texts(texts, metadatas=metadatas)
        vector_store.persist()
        logger.info(f"Successfully added {len(texts)} chunks from {filename} to ChromaDB with syllabus_id: {syllabus_id}.")
    except Exception as e:
        logger.error(f"Error adding chunks to ChromaDB for {filename} (syllabus_id: {syllabus_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document for knowledge base: {e}")

    # Ensure LLM is loaded before attempting to extract syllabus structure
    llm_instance = get_llama_model()
    if llm_instance is None:
        logger.error("Local LLM model is not available for syllabus structure extraction.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local LLM model failed to load during startup. Please check server logs for details."
        )

    try:
        syllabus_structure_obj = extract_syllabus_structure_with_local_model(clean_text, llm_instance=llm_instance)
        parsed_units_data = syllabus_structure_obj.units_data
        logger.info(f"Local LLM extracted {len(parsed_units_data)} main units from {filename} (syllabus_id: {syllabus_id}).")
        return {
            "message": "File processed, added to knowledge base, and syllabus structure extracted.",
            "filename": filename,
            "chunks_added": len(texts),
            "units_data": [unit.model_dump() for unit in parsed_units_data],
            "syllabus_id": syllabus_id
        }
    except HTTPException as e: # Re-raise if it's an HTTPException from deep within
        raise e
    except Exception as e:
        logger.error(f"Failed to extract syllabus structure using Local LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract syllabus structure using Local LLM: {e}")

# --- API Endpoints ---

@app.post("/process-syllabus/", summary="Upload and process a syllabus (PDF) to build the knowledge base and extract chapters.")
async def process_syllabus(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Please upload a PDF file."
        )

    current_syllabus_id = str(uuid.uuid4())
    file_location = Path(settings.upload_dir) / f"{current_syllabus_id}_{file.filename}"

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{file.filename}' saved to '{file_location}' with syllabus_id: {current_syllabus_id}")

        result = process_document_and_add_to_chroma(file_location, file.filename, current_syllabus_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.error(f"Error processing syllabus file (syllabus_id: {current_syllabus_id}): {e}", exc_info=True)
        # Check if it's already an HTTPException (e.g., from process_document_and_add_to_chroma)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process syllabus: {e}")
    finally:
        if file_location.exists():
            os.remove(file_location)
            logger.info(f"Temporary file '{file.filename}' (syllabus_id: {current_syllabus_id}) removed.")

@app.post(
    "/generate-notes/",
    response_model=NoteLinksResponse,
    summary="Generate top 3 best note links by comparing syllabus context with internet search"
)
async def generate_notes(request: GenerateNotesRequest):
    topics = request.topics
    syllabus_id = request.syllabus_id
    source_type = request.source_type

    if not topics:
        raise HTTPException(status_code=400, detail="No topics provided.")
    if source_type == "syllabus" and not syllabus_id:
        raise HTTPException(status_code=400, detail="syllabus_id is required when using syllabus source.")
    if Google_Search is None:
        raise HTTPException(status_code=500, detail="Google Search API is not initialized. Check GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file.")

    generated_links = []

    for topic in topics:
        try:
            # Step 1: get syllabus context (if syllabus_id provided)
            syllabus_context_text = topic # Default to topic itself if no syllabus context
            if source_type == "syllabus" and syllabus_id:
                docs = vector_store.similarity_search(
                    query=topic,
                    k=3, # Retrieve top 3 relevant chunks
                    filter={"syllabus_id": syllabus_id} # Filter by the specific syllabus
                )
                if docs:
                    syllabus_context_text = "\n".join([doc.page_content for doc in docs]).strip()
                else:
                    logger.warning(f"No relevant chunks found in ChromaDB for syllabus_id {syllabus_id} and topic '{topic}'. Using topic as context.")


            # Compute embedding for syllabus context
            syllabus_embedding = embeddings.embed_query(syllabus_context_text)

            # Step 2: run programmable search (get results)
            search_results_raw = Google_Search.results(topic, num_results=10) # Get up to 10 results
            
            # Filter out potentially bad results (e.g., no link or snippet)
            search_results = [
                res for res in search_results_raw 
                if res.get('link') and res.get('snippet')
            ]

            scored_results = []
            for result in search_results:
                external_text = f"{result.get('title', '')} {result.get('snippet', '')}"
                external_embedding = embeddings.embed_query(external_text)
                score = cosine_similarity(syllabus_embedding, external_embedding)
                scored_results.append((score, result))

            # Step 3: sort results by similarity descending
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Step 4: take top 3 results
            top_results = scored_results[:3]

            # Step 5: add them to response list
            if top_results:
                for score, res in top_results:
                    generated_links.append(NoteLink(
                        topic=topic,
                        title=res.get('title', f"Result for {topic}"),
                        snippet=res.get('snippet', ''),
                        url=res.get('link', '#'),
                        similarity_score=round(score, 3)
                    ))
            else:
                # If no good search results were found or filtered out
                generated_links.append(NoteLink(
                    topic=topic,
                    title=f"No relevant internet notes found for {topic}",
                    snippet="No highly similar content found online after search.",
                    url="#",
                    similarity_score=0
                ))

        except Exception as e:
            logger.error(f"Error generating note for topic '{topic}': {e}", exc_info=True)
            generated_links.append(NoteLink(
                topic=topic,
                title=f"Error generating note for {topic}",
                snippet=f"Unexpected error: {e}",
                url="#error",
                similarity_score=0
            ))

    return NoteLinksResponse(notes=generated_links)

    topics = request.topics
    syllabus_id = request.syllabus_id
    source_type = request.source_type

    if not topics:
        raise HTTPException(status_code=400, detail="No topics provided.")
    if source_type == "syllabus" and not syllabus_id:
        raise HTTPException(status_code=400, detail="syllabus_id is required when source_type is 'syllabus'.")

    llm = get_llama_model()
    if llm is None:
        raise HTTPException(status_code=503, detail="Local LLM is not available.")

    result_links = []

    for topic in topics:
        try:
            # 1. Get context
            syllabus_context = topic
            if source_type == "syllabus" and syllabus_id:
                docs = vector_store.similarity_search(
                    query=topic, k=3, filter={"syllabus_id": syllabus_id}
                )
                if docs:
                    syllabus_context = "\n".join([doc.page_content for doc in docs]).strip()
                else:
                    syllabus_context = "This is a computer science topic. Generate helpful study notes."

            # 2. Build prompt
            prompt = f"""
You are a subject expert writing study notes.

Instructions:
- Write clear, structured notes in paragraph form.
- Start with a definition or overview.
- Explain subtopics or concepts with short examples.
- Avoid lists or markdown formatting unless necessary.
- Write in a tone suited for a university-level student.
- Do NOT include any meta-text like "Your task is..." or "Textbook:".

Context (optional for reference):
{syllabus_context}

Please begin the notes below:
"""

            # 3. Call LLM
            try:
                response = llm(prompt, stop=["```", "</s>"])
            except Exception as e:
                raise ValueError(f"LLM call failed: {e}")

            # 4. Handle both str and dict formats
            if isinstance(response, dict) and "choices" in response:
                output = response["choices"][0]["text"]
            elif isinstance(response, str):
                output = response
            else:
                raise ValueError("Unexpected LLM response format")

            output = output.strip()

            # 5. Fallback for short or empty output
            if len(output.split()) < 20:
                logger.warning(f"Short or empty output for topic '{topic}':\n{output}")
                output = f"(⚠️ Auto-generated note is too brief or unavailable for topic: '{topic}')"

            # 6. Save to file
            note_id = str(uuid.uuid4())
            filename = f"{note_id}.md"
            note_path = NOTES_DIR / filename

            with open(note_path, "w", encoding="utf-8") as f:
                f.write(f"# Notes for {topic}\n\n{output}\n")

            # 7. Add to result
            result_links.append({
                "topic": topic,
                "note_id": note_id,
                "download_url": f"/download-note/{note_id}"
            })

        except Exception as e:
            logger.error(f"Failed to generate notes for topic '{topic}': {e}", exc_info=True)
            result_links.append({
                "topic": topic,
                "note_id": None,
                "error": str(e)
            })

    return result_links
    
    topics = request.topics
    syllabus_id = request.syllabus_id
    source_type = request.source_type

    if not topics:
        raise HTTPException(status_code=400, detail="No topics provided.")
    if source_type == "syllabus" and not syllabus_id:
        raise HTTPException(status_code=400, detail="syllabus_id is required when source_type is 'syllabus'.")

    llm = get_llama_model()
    if llm is None:
        raise HTTPException(status_code=503, detail="Local LLM is not available.")

    result_links = []

    for topic in topics:
        try:
            # 1. Get context
            syllabus_context = topic
            if source_type == "syllabus" and syllabus_id:
                docs = vector_store.similarity_search(
                    query=topic, k=3, filter={"syllabus_id": syllabus_id}
                )
                if docs:
                    syllabus_context = "\n".join([doc.page_content for doc in docs]).strip()

            # 2. Build prompt
            prompt = f"""
You are a subject expert writing study notes.

Instructions:
- Write clear, structured notes in paragraph form.
- Start with a definition or overview.
- Explain subtopics or concepts with short examples.
- Avoid lists or markdown formatting unless necessary.
- Write in a tone suited for a university-level student.
- Do NOT include any meta-text like "Your task is..." or "Textbook:".

Context (optional for reference):
{syllabus_context}

Please begin the notes below:
"""

            # 3. Generate using LLM
            try:
                response = llm(prompt, stop=["```", "</s>"])
            except Exception as e:
                raise ValueError(f"LLM call failed: {e}")

            # 4. Handle response format
            if isinstance(response, dict) and "choices" in response:
                output = response["choices"][0]["text"]
            elif isinstance(response, str):
                output = response
            else:
                raise ValueError("Unexpected LLM response format")

            output = output.strip()

            # Optional: Guard against very short/incomplete notes
            if len(output.split()) < 20:
                raise ValueError("LLM generated output is too short or empty.")

            # 5. Save to file
            note_id = str(uuid.uuid4())
            filename = f"{note_id}.md"
            note_path = NOTES_DIR / filename

            with open(note_path, "w", encoding="utf-8") as f:
                f.write(f"# Notes for {topic}\n\n{output}\n")

            # 6. Return link
            result_links.append({
                "topic": topic,
                "note_id": note_id,
                "download_url": f"/download-note/{note_id}"
            })

        except Exception as e:
            logger.error(f"Failed to generate notes for topic '{topic}': {e}", exc_info=True)
            result_links.append({
                "topic": topic,
                "note_id": None,
                "error": str(e)
            })

    return result_links


@app.post("/generate-ai-notes/")
async def generate_ai_notes(request: GenerateNotesRequest):
    topics = request.topics
    syllabus_id = request.syllabus_id
    source_type = request.source_type

    if not topics:
        raise HTTPException(status_code=400, detail="No topics provided.")
    if source_type == "syllabus" and not syllabus_id:
        raise HTTPException(status_code=400, detail="syllabus_id is required when source_type is 'syllabus'.")

    result_links = []

    for topic in topics:
        try:
            # 1. Get context
            syllabus_context = topic
            if source_type == "syllabus" and syllabus_id:
                docs = vector_store.similarity_search(
                    query=topic, k=3, filter={"syllabus_id": syllabus_id}
                )
                if docs:
                    syllabus_context = "\n".join([doc.page_content for doc in docs]).strip()
                else:
                    syllabus_context = "This is a computer science topic. Generate helpful study notes."

            # 2. Build prompt
            prompt = f"""
You are a subject expert writing study notes.

Instructions:
- Write clear, structured notes in paragraph form.
- Start with a definition or overview.
- Explain subtopics or concepts with short examples.
- Avoid lists or markdown formatting unless necessary.
- Write in a tone suited for a university-level student.
- Do NOT include any meta-text like "Your task is..." or "Textbook:".

Context (optional for reference):
{syllabus_context}

Please begin the notes below:
"""

            # 3. Call Ollama Llama 3
            output = call_ollama_llama3(prompt)

            output = output.strip()

            # 4. Fallback for short or empty output
            if len(output.split()) < 20:
                logger.warning(f"Short or empty output for topic '{topic}':\n{output}")
                output = f"(⚠️ Auto-generated note is too brief or unavailable for topic: '{topic}')"

            # 5. Save to file
            note_id = str(uuid.uuid4())
            filename = f"{note_id}.md"
            note_path = NOTES_DIR / filename

            with open(note_path, "w", encoding="utf-8") as f:
                f.write(f"# Notes for {topic}\n\n{output}\n")

            # 6. Add to result
            result_links.append({
                "topic": topic,
                "note_id": note_id,
                "download_url": f"/download-note/{note_id}"
            })

        except Exception as e:
            logger.error(f"Failed to generate notes for topic '{topic}': {e}", exc_info=True)
            result_links.append({
                "topic": topic,
                "note_id": None,
                "error": str(e)
            })

    return result_links

@app.get("/download-note/{note_id}", summary="Download a generated note by ID")
def download_note(note_id: str):
    file_path = NOTES_DIR / f"{note_id}.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found.")
    return FileResponse(file_path, media_type="text/markdown", filename=f"{note_id}.md")


@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "NoteMate Backend is running. Access /docs for API documentation."}

def call_ollama_llama3(prompt: str, model: str = "llama3", base_url: str = "http://localhost:11434/api/generate") -> str:
    """
    Calls Ollama's Llama 3 API with the given prompt and returns the generated text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(base_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # Ollama returns the result in 'response' or 'message' or 'text'
        if "response" in data:
            return data["response"].strip()
        elif "message" in data:
            return data["message"].strip()
        elif "text" in data:
            return data["text"].strip()
        else:
            raise ValueError(f"Unexpected Ollama response format: {data}")
    except Exception as e:
        logger.error(f"Ollama Llama 3 call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama Llama 3 call failed: {e}")