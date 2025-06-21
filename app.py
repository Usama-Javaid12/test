# streamlined_rag_chatbot.py

import os
import hashlib
import sqlite3
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import shutil

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment variables")

DOCS_DIR = Path("./documents")
VECTOR_DIR = Path("./chroma_store")
DB_FILE = Path("./chatlogs.db")

# Create directories if they don't exist
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# Processing parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.pptx', '.xlsx'}

# --- Data Models ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: str
    processing_time: float

class UploadResponse(BaseModel):
    uploaded_files: List[str]
    failed_files: List[Dict[str, str]]
    ingestion_result: Dict[str, Any]

# --- Database Operations ---

def init_database():
    """Initialize SQLite database"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    conversation_id TEXT,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_id ON chat_logs(conversation_id);"
            )
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def log_chat(question: str, answer: str, conversation_id: str, processing_time: float):
    """Log chat interaction"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                """
                INSERT INTO chat_logs (question, answer, conversation_id, processing_time) 
                VALUES (?, ?, ?, ?)
                """,
                (question, answer, conversation_id, processing_time),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log chat: {str(e)}")

# --- Document Processing ---

def load_document(file_path: Path) -> List:
    """Load document with the appropriate loader"""
    try:
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(str(file_path))
        elif ext == ".xlsx":
            loader = UnstructuredExcelLoader(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []

        documents = loader.load()
        logger.info(f"Loaded {file_path.name} ({len(documents)} chunks)")
        return documents
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {str(e)}")
        return []

def ingest_documents():
    """Ingest all documents and create/update vector store"""
    start_time = datetime.now()
    
    try:
        # Find all supported files
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(DOCS_DIR.glob(f"**/*{ext}"))
        
        if not all_files:
            return {
                "status": "no_documents",
                "files_processed": 0,
                "chunks_created": 0,
                "processing_time": 0.0,
            }
        
        # Load all documents
        all_documents = []
        processed_files = 0
        
        for file_path in all_files:
            loaded_docs = load_document(file_path)
            if loaded_docs:
                all_documents.extend(loaded_docs)
                processed_files += 1
        
        if not all_documents:
            return {
                "status": "no_valid_documents",
                "files_processed": 0,
                "chunks_created": 0,
                "processing_time": (datetime.now() - start_time).total_seconds(),
            }
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks from {processed_files} files")
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Check if vector store exists
        if VECTOR_DIR.exists():
            # Load existing vector store and add new documents
            logger.info("Adding documents to existing vector store")
            vectorstore = Chroma(
                persist_directory=str(VECTOR_DIR), 
                embedding_function=embeddings
            )
            # Add new documents to existing store
            vectorstore.add_documents(chunks)
        else:
            # Create new vector store
            logger.info("Creating new vector store")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(VECTOR_DIR),
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            "status": "success",
            "files_processed": processed_files,
            "chunks_created": len(chunks),
            "processing_time": processing_time,
        }
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

# --- RAG System ---

def get_vectorstore():
    """Get vector store"""
    if not VECTOR_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Please upload documents first.",
        )
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)

def get_qa_chain():
    """Get QA chain"""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0.2, 
        openai_api_key=OPENAI_API_KEY
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

def process_chat_request(request: ChatRequest) -> ChatResponse:
    """Process chat request"""
    start_time = datetime.now()
    
    try:
        # Generate conversation ID if not provided
        if not request.conversation_id:
            request.conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        # Get QA chain and process question
        qa_chain = get_qa_chain()
        result = qa_chain({"query": request.question})
        
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        
        # Extract unique sources
        sources = list({
            doc.metadata.get("source", "Unknown").split("/")[-1]  # Get filename only
            for doc in source_docs
            if doc.metadata.get("source")
        })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log the chat
        log_chat(
            question=request.question,
            answer=answer,
            conversation_id=request.conversation_id,
            processing_time=processing_time,
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=request.conversation_id,
            processing_time=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FastAPI Application ---

app = FastAPI(title="Streamlined RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
init_database()

@app.get("/")
def root():
    return {"message": "Streamlined RAG Chatbot API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store_exists": VECTOR_DIR.exists(),
        "supported_formats": list(SUPPORTED_EXTENSIONS),
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents and automatically create vector store"""
    uploaded_files = []
    failed_files = []
    
    for file in files:
        try:
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_EXTENSIONS:
                failed_files.append({
                    "filename": file.filename, 
                    "error": f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
                })
                continue
            
            # Check file size
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                failed_files.append({
                    "filename": file.filename, 
                    "error": "File too large (max 50MB)"
                })
                continue
            
            # Save file
            file_path = DOCS_DIR / file.filename
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append(file.filename)
            logger.info(f"Uploaded: {file.filename}")
            
        except Exception as e:
            failed_files.append({
                "filename": file.filename, 
                "error": str(e)
            })
        finally:
            await file.seek(0)  # Reset file pointer
    
    # Automatically ingest documents after upload
    try:
        ingestion_result = ingest_documents()
    except Exception as e:
        ingestion_result = {"status": "failed", "error": str(e)}
    
    return UploadResponse(
        uploaded_files=uploaded_files,
        failed_files=failed_files,
        ingestion_result=ingestion_result,
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat with the documents"""
    return process_chat_request(request)

@app.delete("/documents")
def clear_documents():
    """Clear all documents and vector store"""
    try:
        # Remove all uploaded documents
        if DOCS_DIR.exists():
            for file_path in DOCS_DIR.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        
        # Remove vector store
        if VECTOR_DIR.exists():
            shutil.rmtree(VECTOR_DIR)
        
        return {"message": "All documents and vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.get("/conversations")
def list_conversations():
    """List all conversations"""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT 
                    conversation_id,
                    COUNT(*) as message_count,
                    MIN(timestamp) as started_at,
                    MAX(timestamp) as last_message_at
                FROM chat_logs 
                WHERE conversation_id IS NOT NULL
                GROUP BY conversation_id 
                ORDER BY last_message_at DESC
                """
            )
            conversations = [dict(row) for row in cursor.fetchall()]
            return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
