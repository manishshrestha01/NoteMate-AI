# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
import numpy as np

# Document Loaders for text extraction
from PyPDF2 import PdfReader

# LangChain components for text splitting, embeddings, vector store, LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as LCBaseModel
from pydantic import Field as LCField

# Google Search Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for AI Extraction Output (Syllabus Structure) ---
class SubUnit(LCBaseModel):
    title: str = LCField(description="The title of the sub-unit, e.g., '1.1 Types', '2.3 Interprocess Communication'.")
    content: str = LCField(description="The full textual content of this sub-unit.")

class Unit(LCBaseModel):
    title: str = LCField(description="The title of the main unit, e.g., 'Unit 1: Introduction', 'Chapter 2: Processes'.")
    content: str = LCField(description="The full textual content of this main unit, excluding content explicitly within its sub-units. Can be empty if all content is within sub-units.")
    sub_units: List[SubUnit] = LCField(default_factory=list, description="A list of sub-units within this main unit.")

class SyllabusStructure(LCBaseModel):
    units_data: List[Unit] = LCField(description="A list of main units found in the syllabus, each with its title, content, and nested sub-units.")

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

    gemini_api_key: Optional[str] = None
    gemini_extraction_model_name: str = "gemini-1.5-flash"
    gemini_notes_generation_model_name: str = "gemini-1.5-flash"

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

# Initialize Gemini LLM for Extraction
extraction_llm = None
if settings.gemini_api_key:
    extraction_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_extraction_model_name,
        temperature=0.1,
        google_api_key=settings.gemini_api_key,
        convert_system_message_to_human=True
    )
    logger.info(f"Extraction LLM (Gemini) initialized with model: {settings.gemini_extraction_model_name}")
else:
    logger.error("Gemini API key is missing. AI syllabus extraction will be disabled.")

# Initialize Gemini LLM for Notes Generation (can be the same model or different)
notes_generation_llm = None
if settings.gemini_api_key:
    notes_generation_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_notes_generation_model_name,
        temperature=0.7,
        google_api_key=settings.gemini_api_key,
        convert_system_message_to_human=True
    )
    logger.info(f"Notes Generation LLM (Gemini) initialized with model: {settings.gemini_notes_generation_model_name}")
else:
    logger.error("Gemini API key is missing. AI note generation will be disabled.")

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
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {e}")
    return text

def extract_syllabus_structure_with_llm(syllabus_text: str) -> SyllabusStructure:
    if extraction_llm is None:
        raise HTTPException(
            status_code=500,
            detail="AI extraction LLM (Gemini) is not initialized. Please ensure GEMINI_API_KEY is set in .env."
        )

    prompt_template = """
    You are an expert syllabus parser. Your task is to extract the main units and their sub-units from the provided syllabus text.
    Identify main units by clear headers like "Unit X:", "Chapter Y:", or "Module Z:".
    Within each main unit, identify sub-units by numerical patterns like "1.1", "1.2.3", etc.

    For each unit and sub-unit, extract its exact title and its full textual content.
    The 'content' field for a unit/sub-unit should include all text from its title line up to the next unit/sub-unit title, or the end of its enclosing section.
    If a unit/sub-unit has no further specific content (e.g., only contains nested sub-units), its 'content' field should be an empty string.

    If no units or sub-units are found, return an empty list for 'units_data'.

    Syllabus Text:
    ---
    {syllabus_text}
    ---
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    extraction_chain = prompt | extraction_llm.with_structured_output(SyllabusStructure)

    try:
        syllabus_structure = extraction_chain.invoke({"syllabus_text": syllabus_text})
        logger.info("Successfully parsed syllabus structure with Gemini LLM.")
        return syllabus_structure

    except ValidationError as e:
        logger.error(f"Pydantic validation failed for LLM output: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM extraction failed: Invalid structure from AI. Error: {e.errors()}")
    except Exception as e:
        logger.error(f"Unexpected error during Gemini LLM extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gemini LLM extraction failed: {e}")

def process_document_and_add_to_chroma(file_path: Path, filename: str, syllabus_id: str):
    file_type = get_file_type(str(file_path))
    logger.info(f"Processing file: {filename} with detected type: {file_type}")

    if file_type not in settings.supported_file_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_type}. Only PDF files are supported."
        )

    raw_text = extract_text_from_pdf(str(file_path))

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

    try:
        if extraction_llm is None:
            raise HTTPException(
                status_code=500,
                detail="AI extraction LLM (Gemini) is not initialized. Please ensure GEMINI_API_KEY is set in .env."
            )
        syllabus_structure_obj = extract_syllabus_structure_with_llm(raw_text)
        parsed_units_data = syllabus_structure_obj.units_data
        logger.info(f"LLM extracted {len(parsed_units_data)} main units from {filename} (syllabus_id: {syllabus_id}).")
        return {
            "message": "File processed, added to knowledge base, and syllabus structure extracted.",
            "filename": filename,
            "chunks_added": len(texts),
            "units_data": [unit.model_dump() for unit in parsed_units_data],
            "syllabus_id": syllabus_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to extract syllabus structure using LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract syllabus structure using AI (Gemini): {e}")

# Cosine similarity helper
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
        raise HTTPException(status_code=500, detail="Google Search API is not initialized. Check GOOGLE_API_KEY and GOOGLE_CSE_ID.")

    generated_links = []

    for topic in topics:
        try:
            # Step 1: get syllabus context (if syllabus_id provided)
            syllabus_text = topic
            if source_type == "syllabus" and syllabus_id:
                docs = vector_store.similarity_search(
                    query=topic,
                    k=3,
                    filter={"syllabus_id": syllabus_id}
                )
                syllabus_text = "\n".join([doc.page_content for doc in docs]).strip() or topic

            # Compute embedding for syllabus context
            syllabus_embedding = embeddings.embed_query(syllabus_text)

            # Step 2: run programmable search (get results)
            search_results = Google_Search.results(topic, num_results=3)
            search_results = search_results[:10]  # get max 10 results for scoring

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
            for score, res in top_results:
                generated_links.append(NoteLink(
                    topic=topic,
                    title=res.get('title', f"Result for {topic}"),
                    snippet=res.get('snippet', ''),
                    url=res.get('link', '#'),
                    similarity_score=round(score, 3)
                ))

            # If no results found
            if not top_results:
                generated_links.append(NoteLink(
                    topic=topic,
                    title=f"No good match found for {topic}",
                    snippet="No relevant content found online.",
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

@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "NoteMate Backend is running. Access /docs for API documentation."}
