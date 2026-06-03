import os
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import (
    load_files_from_folder,
    chunk_files,
    build_vector_store,
    load_vector_store,
    HybridRetriever,
    answer_question,
)

# Global variables to cache retriever and stats in memory
retriever: Optional[HybridRetriever] = None
indexed_folder: Optional[str] = None
chunks_cache = []

CHROMA_DIR = "./chroma_db"
META_FILE = "./chroma_db/meta.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, indexed_folder, chunks_cache

    if Path(CHROMA_DIR).exists() and Path(META_FILE).exists():
        try:
            with open(META_FILE) as f:
                meta = json.load(f)
            indexed_folder = meta.get("folder")

            print(f"Loading cached vector database for: {indexed_folder}")
            vector_store = load_vector_store(CHROMA_DIR)

            files = load_files_from_folder(indexed_folder)
            chunks_cache = chunk_files(files)
            retriever = HybridRetriever(vector_store, chunks_cache)
            print("Gitty index loading complete.")
        except Exception as e:
            print(f"Could not load cached index: {e}")

    yield


app = FastAPI(
    title="Gitty API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    folder_path: str


class QuestionRequest(BaseModel):
    query: str
    use_reranker: bool = True


class Source(BaseModel):
    file: str
    snippet: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]
    query: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "indexed": indexed_folder is not None,
        "indexed_folder": indexed_folder,
        "chunks": len(chunks_cache),
    }


@app.post("/index")
async def index_codebase(request: IndexRequest):
    """Builds vector database and BM25 index from a local folder path."""
    global retriever, indexed_folder, chunks_cache

    folder = os.path.expanduser(request.folder_path)

    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {folder}")

    try:
        files = load_files_from_folder(folder)
        if not files:
            raise HTTPException(status_code=400, detail="No source files found in the folder.")

        chunks = chunk_files(files)
        chunks_cache = chunks

        # Wipe old db contents
        import shutil
        if Path(CHROMA_DIR).exists():
            try:
                shutil.rmtree(CHROMA_DIR)
            except Exception as e:
                print(f"Warning: Could not clear previous db folder: {e}")

        vector_store = build_vector_store(chunks, persist_dir=CHROMA_DIR)

        retriever = HybridRetriever(vector_store, chunks)
        indexed_folder = folder

        Path(CHROMA_DIR).mkdir(exist_ok=True)
        with open(META_FILE, "w") as f:
            json.dump({"folder": folder, "files": len(files), "chunks": len(chunks)}, f)

        return {
            "status": "indexed",
            "folder": folder,
            "files_indexed": len(files),
            "chunks_created": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    """Query the codebase retriever and return the LLM answer with sources."""
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No project has been indexed yet. Post to /index first."
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer, sources = answer_question(
            query=request.query,
            retriever=retriever,
            use_reranker=request.use_reranker,
        )

        return QuestionResponse(
            answer=answer,
            query=request.query,
            sources=[
                Source(
                    file=doc.metadata.get("source", "unknown"),
                    snippet=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                )
                for doc in sources
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    return {
        "indexed_folder": indexed_folder,
        "total_chunks": len(chunks_cache),
        "ready": retriever is not None,
    }
