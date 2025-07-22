from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import shutil
import magic
from pathlib import Path
import logging
import re
import uuid

# Document Loader
import fitz  # PyMuPDF

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic models ---
class SubUnit(BaseModel):
    title: str
    content: str = ""


class Unit(BaseModel):
    title: str
    content: str = ""
    sub_units: List[SubUnit] = []


class SyllabusStructure(BaseModel):
    units_data: List[Unit]


# --- Settings ---
UPLOAD_DIR = "uploaded_files"
CHROMA_DB_DIR = "chroma_db"

Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(CHROMA_DB_DIR).mkdir(exist_ok=True)


# --- LangChain init ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)


# --- FastAPI app ---
app = FastAPI(title="NoteMate Backend", version="1.0")


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---
def get_file_type(path: str) -> str:
    return magic.from_file(path, mime=True)


def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail="PDF extraction failed")
    return text


def preprocess_syllabus_text(text: str) -> str:
    lines = text.splitlines()
    merged = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        if buffer and not re.match(r'^(Unit|Chapter|Module|\d+\.\d+)', stripped, re.IGNORECASE):
            buffer += " " + stripped
        else:
            if buffer:
                merged.append(buffer)
            buffer = stripped
    if buffer:
        merged.append(buffer)
    return "\n".join(merged)


def extract_syllabus_structure_with_local_model(syllabus_text: str) -> SyllabusStructure:
    """
    Final fixed version:
    - Supports numbered sub-units (1.1, 2.2)
    - Supports plain numbers (1., 2., etc.)
    - Supports bullets (-, •)
    - Removes trailing lab/tutorials/evaluation from sub-units
    """
    logger.info("Extracting syllabus structure...")

    cut_keywords = [
        "Laboratory Works", "Practical Works", "List of Tutorials",
        "Evaluation System", "Students’ Responsibilities",
        "Prescribed Books and References", "Text Books", "References"
    ]
    for kw in cut_keywords:
        idx = syllabus_text.lower().find(kw.lower())
        if idx != -1:
            syllabus_text = syllabus_text[:idx]
            break

    syllabus_text = re.sub(r'\n+', '\n', syllabus_text.strip())

    unit_pattern = re.compile(r'^\s*(Unit|Chapter|Module)\s*-?\s*([0-9]+|[IVXLCDM]+)\b.*$', re.MULTILINE | re.IGNORECASE)
    unit_matches = list(unit_pattern.finditer(syllabus_text))

    units = []

    for idx, match in enumerate(unit_matches):
        unit_title = match.group(0).strip()
        start = match.end()
        end = unit_matches[idx + 1].start() if idx + 1 < len(unit_matches) else len(syllabus_text)
        unit_block = syllabus_text[start:end].strip()

        sub_units = []

        # 1️⃣ Numbered sub-units: 1.1, 2.2
        subunit_numbered = re.findall(r'^\s*(\d+\.\d+)\.?\s*(.+)', unit_block, re.MULTILINE)
        for number, title in subunit_numbered:
            clean_title = re.split(r'●|•|- Understand|- Review', title)[0].strip()
            sub_units.append(SubUnit(title=f"{number} {clean_title}"))

        # 2️⃣ Plain numbers: 1., 2., 3.
        subunit_plain_numbers = re.findall(r'^\s*\d+\.\s+(.+)', unit_block, re.MULTILINE)
        for title in subunit_plain_numbers:
            clean_title = re.split(r'●|•|- Understand|- Review', title)[0].strip()
            if not any(clean_title in sub.title for sub in sub_units):
                sub_units.append(SubUnit(title=clean_title))

        # 3️⃣ Bullets: - or •
        subunit_bullets = re.findall(r'^\s*[•-]\s*(.+)', unit_block, re.MULTILINE)
        for bullet in subunit_bullets:
            clean_bullet = re.split(r'●|•|- Understand|- Review', bullet)[0].strip()
            if not any(clean_bullet in sub.title for sub in sub_units):
                sub_units.append(SubUnit(title=clean_bullet))

        # 4️⃣ Filter out anything like "Laboratory Work" in sub_units (cleanup safeguard)
        cleaned_sub_units = [
            sub for sub in sub_units
            if not any(bad in sub.title.lower() for bad in ["laboratory", "tutorial", "evaluation", "assignment"])
        ]

        units.append(Unit(title=unit_title, sub_units=cleaned_sub_units))

    return SyllabusStructure(units_data=units)



# --- Process endpoint ---
@app.post("/process-syllabus/")
async def process_syllabus(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Upload a PDF file only")

    syllabus_id = str(uuid.uuid4())
    filename = f"{syllabus_id}_{file.filename}"
    file_path = Path(UPLOAD_DIR) / filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        raw = extract_text_from_pdf(str(file_path))
        clean = preprocess_syllabus_text(raw)

        # Add to vector store
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(raw)
        vector_store.add_texts(chunks, metadatas=[{"syllabus_id": syllabus_id}] * len(chunks))
        vector_store.persist()

        # Extract structure
        structure = extract_syllabus_structure_with_local_model(clean)
        logger.info(f"Extracted {len(structure.units_data)} units")

        return {
            "message": "File processed, structure extracted.",
            "filename": file.filename,
            "syllabus_id": syllabus_id,
            "units_data": [u.model_dump() for u in structure.units_data]
        }
    finally:
        if file_path.exists():
            os.remove(file_path)


@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "NoteMate Backend is running. Access /docs for API documentation."}
