import os
from pathlib import Path
from typing import List, Dict, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import google.generativeai as genai


def load_files_from_folder(folder_path: str) -> List[Dict]:
    """Scan folder and read content of supported text/code files."""
    supported_extensions = [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".html", ".css", ".java", ".go", ".rs", ".cpp", ".c", ".h"
    ]

    skip_dirs = {
        ".git", "node_modules", "__pycache__", ".venv",
        "venv", "dist", "build", ".next", ".idea", ".vscode"
    }

    files = []

    for root, dirs, filenames in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for filename in filenames:
            if not any(filename.endswith(ext) for ext in supported_extensions):
                continue

            full_path = os.path.join(root, filename)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip() or len(content) > 50_000:
                    continue

                files.append({
                    "path": full_path,
                    "filename": filename,
                    "extension": Path(filename).suffix,
                    "content": content,
                    "size": len(content)
                })

            except Exception as e:
                print(f"Could not read {full_path}: {e}")

    return files


def chunk_files(files: List[Dict]) -> List:
    """Split file contents into text chunks with overlap using LangChain splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", "def ", "class ", " ", ""]
    )

    all_chunks = []

    for file in files:
        try:
            chunks = splitter.create_documents(
                texts=[file["content"]],
                metadatas=[{
                    "source": file["path"],
                    "filename": file["filename"],
                    "extension": file["extension"],
                }]
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Could not chunk {file['filename']}: {e}")

    return all_chunks


def build_vector_store(chunks: List, persist_dir: str = "./chroma_db") -> Chroma:
    """Generate embeddings using sentence-transformers and store in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="codebase"
    )
    return vector_store


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """Load existing Chroma DB from disk using the MiniLM embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="codebase"
    )


class HybridRetriever:
    """Merges results of Chroma semantic search and BM25 keyword search."""

    def __init__(self, vector_store: Chroma, chunks: List):
        self.vector_store = vector_store
        self.chunks = chunks

        # Tokenize chunks for BM25 search index
        tokenized_corpus = [c.page_content.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10) -> List:
        # Semantic vector search
        try:
            semantic_hits = self.vector_store.similarity_search(query, k=k)
        except Exception:
            semantic_hits = []

        # BM25 keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]
        keyword_hits = [self.chunks[i] for i in top_bm25_indices]

        # Merge and deduplicate
        seen_content = set()
        combined = []
        for doc in semantic_hits + keyword_hits:
            key = doc.page_content[:100]
            if key not in seen_content:
                seen_content.add(key)
                combined.append(doc)

        return combined[:k]


def rerank(query: str, candidates: List, top_n: int = 4) -> List:
    """Re-rank candidate documents using a CrossEncoder model."""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def ask_gemini(query: str, top_docs: List) -> str:
    """Generate response from Gemini model using retrieved source code chunks as context."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY env variable not set."

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Chunk {i}] File: {source}\n{doc.page_content}")

    context = ("\n" + "─" * 60 + "\n").join(context_parts)

    prompt = f"""You are an expert code assistant. You answer questions about a codebase
using the code snippets provided below.

Rules:
- Always reference the file name when explaining something.
- If the answer is not in the provided chunks, say so honestly.
- Be concise but complete.
- Use markdown formatting for code snippets.

Here are the most relevant code snippets from the codebase:

{context}

Question: {query}"""

    response = model.generate_content(prompt)
    return response.text


def answer_question(
    query: str,
    retriever: HybridRetriever,
    use_reranker: bool = True
) -> Tuple[str, List]:
    """Retrieve relevant documents, rerank if enabled, and request answer from Gemini."""
    candidates = retriever.retrieve(query, k=10)

    if not candidates:
        return "No relevant code found.", []

    if use_reranker and len(candidates) > 1:
        top_docs = rerank(query, candidates, top_n=4)
    else:
        top_docs = candidates[:4]

    answer = ask_gemini(query, top_docs)
    return answer, top_docs
