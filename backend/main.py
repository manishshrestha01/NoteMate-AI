import os
import shutil
import logging
import re
import uuid
from typing import Optional, List, Literal
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Document Loaders for text extraction
from PyPDF2 import PdfReader

# LangChain components for text splitting, embeddings, vector store, LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel, Field as LCField

# Google Search Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Schemas ---
class GeneratedNote(LCBaseModel):
    topicTitle: str = LCField(description="The main topic/chapter title this note relates to (e.g., 'Introduction to React').")
    title: str = LCField(description="A concise title for the generated resource/note (e.g., 'Official React Docs').")
    snippet: str = LCField(description="A brief summary or excerpt of the resource content.")
    url: str = LCField(description="A placeholder or dummy URL for the resource. In a real app, this would come from a web search API.")
    source: str = LCField(description="The type of source (e.g., 'educational', 'github', 'article', 'video', 'web_search').")

class GeneratedNotesList(LCBaseModel):
    notes: List[GeneratedNote] = LCField(description="A list of generated notes/resources for the given topics.")

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
    gemini_notes_generation_model_name: str = "gemini-1.5-flash"
    
    gcp_project_id: Optional[str] = None
    
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
embeddings = None
vector_store = None
notes_generation_llm = None
Google_Search = None

if settings.gcp_project_id:
    try:
        embeddings = VertexAIEmbeddings(
            model_name="text-embedding-gecko@001",
            project=settings.gcp_project_id
        )
        vector_store = Chroma(
            persist_directory=settings.chroma_db_path,
            embedding_function=embeddings
        )
        logger.info("VertexAIEmbeddings and ChromaDB initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize VertexAIEmbeddings/ChromaDB: {e}", exc_info=True)

if settings.gemini_api_key:
    notes_generation_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_notes_generation_model_name,
        temperature=0.7,
        google_api_key=settings.gemini_api_key,
        convert_system_message_to_human=True
    )
    logger.info("Gemini LLM for notes generation initialized.")

if settings.google_api_key and settings.google_cse_id:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["GOOGLE_CSE_ID"] = settings.google_cse_id
    try:
        Google_Search = GoogleSearchAPIWrapper()
        logger.info("Google Search API Wrapper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Google Search API: {e}")
        Google_Search = None
else:
    logger.warning("Google API Key or CSE ID missing. Internet search will be unavailable.")

app = FastAPI(
    title="NoteMate Backend - Syllabus & Notes Processor",
    description="API for uploading and processing PDF syllabuses, and generating study notes.",
    version="0.1.0"
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions for Document Processing ---
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

def process_document_and_add_to_chroma(file_path: Path, filename: str, syllabus_id: str):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized. Check server logs.")

    raw_text = extract_text_from_pdf(str(file_path))

    if not raw_text.strip():
        logger.warning(f"Extracted no meaningful text from {filename}. Skipping.")
        return {"message": "File processed, but no significant text was extracted.", "syllabus_id": syllabus_id}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    docs = [
        Document(page_content=text, metadata={"syllabus_id": syllabus_id, "source_filename": filename})
        for text in texts
    ]
    
    try:
        vector_store.add_documents(docs)
        vector_store.persist()
        logger.info(f"Added {len(docs)} chunks from {filename} to ChromaDB.")
        return {
            "message": "File processed and added to knowledge base.",
            "filename": filename,
            "chunks_added": len(docs),
            "syllabus_id": syllabus_id
        }
    except Exception as e:
        logger.error(f"Error adding chunks to ChromaDB for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {e}")

# --- API Endpoints ---
@app.post("/process-syllabus/")
async def process_syllabus(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF file.")

    syllabus_id = str(uuid.uuid4())
    file_location = Path(settings.upload_dir) / f"{syllabus_id}_{file.filename}"

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = process_document_and_add_to_chroma(file_location, file.filename, syllabus_id)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Error processing syllabus: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process syllabus: {e}")
    finally:
        if file_location.exists():
            os.remove(file_location)

@app.post("/generate-notes/", response_model=GeneratedNotesList)
async def generate_notes(request: GenerateNotesRequest):
    if not notes_generation_llm:
        raise HTTPException(status_code=500, detail="AI generation model not initialized. Check logs.")
    
    all_generated_notes = []
    notes_prompt_template = """
    You are an AI assistant specialized in generating concise study notes and relevant resource suggestions for academic topics.
    Given a specific topic and its context, your task is to:
    1. Create a compelling, brief title for a study resource related to the topic.
    2. Write a short, informative snippet (1-2 sentences) summarizing the key takeaway or purpose of the resource.
    3. Provide a placeholder URL. Use a relevant placeholder like "https://example.com/resource/{{topic_slug}}" or if search results provide a good URL, use that.
    4. Categorize the source. Use 'educational', 'github', 'article', 'video', or 'web_search' for internet results, 'syllabus_excerpt' for syllabus results.
    
    Generate ONE highly relevant resource for the following topic based on the context provided.
    Ensure the output strictly adheres to the JSON schema for 'GeneratedNote'.
    
    Topic: {topic_title}
    
    Context:
    ---
    {context}
    ---
    
    Now, generate the resource for the given Topic and Context.
    """
    notes_prompt = ChatPromptTemplate.from_template(notes_prompt_template)
    
    for topic in request.topics:
        context_text = ""
        primary_url = ""
        source_category = ""
        
        try:
            if request.source_type == "syllabus":
                if not request.syllabus_id:
                    raise HTTPException(status_code=400, detail="Syllabus ID is required for 'syllabus' source type.")
                if not vector_store:
                    raise HTTPException(status_code=500, detail="Vector store not initialized. Check server logs.")

                retrieved_docs = vector_store.similarity_search(query=topic, k=3, filter={"syllabus_id": request.syllabus_id})
                context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                source_category = "syllabus_excerpt"

            elif request.source_type == "internet":
                if not Google_Search:
                    raise HTTPException(status_code=500, detail="Google Search API is not initialized. Check your API keys.")
                
                search_query = f"{topic} study notes"
                search_results_str = Google_Search.run(search_query)
                context_text = "Internet Search Results:\n" + search_results_str
                source_category = "web_search"
                
                match = re.search(r'\[(https?://\S+)\]', search_results_str)
                if match:
                    primary_url = match.group(1)
                else:
                    primary_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"

            # If no context, provide a default note
            if not context_text.strip():
                generated_note = GeneratedNote(
                    topicTitle=topic,
                    title=f"No Context Found for {topic}",
                    snippet=f"Unable to find specific content for '{topic}' from the selected source.",
                    url=f"https://www.google.com/search?q={topic.replace(' ', '+')}",
                    source="general"
                )
            else:
                note_generation_chain = notes_prompt | notes_generation_llm.with_structured_output(GeneratedNote)
                generated_note = note_generation_chain.invoke({"topic_title": topic, "context": context_text})
                
                if primary_url:
                    generated_note.url = primary_url
                generated_note.source = source_category

            all_generated_notes.append(generated_note)

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error during note generation for topic '{topic}': {e}", exc_info=True)
            all_generated_notes.append(GeneratedNote(
                topicTitle=topic,
                title=f"Error generating note for {topic}",
                snippet=f"An unexpected error occurred: {e}",
                url="#error",
                source="error"
            ))

    return GeneratedNotesList(notes=all_generated_notes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
