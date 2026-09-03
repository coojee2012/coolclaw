import os
import logging
from pathlib import Path
from typing import Optional, list
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    name: str
    content: str
    chunks: list[str]
    embeddings: Optional[list] = None
    metadata: dict = None


@dataclass
class SearchResult:
    content: str
    score: float
    source: str
    chunk_index: int


class KnowledgeBase:
    def __init__(self, storage_path: Optional[str] = None):
        self._storage_path = Path(
            storage_path or os.path.expanduser("~/.opencode_helper/knowledge")
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._storage_path / "index"
        self._documents_path = self._storage_path / "documents.json"
        self._documents: dict[str, Document] = {}
        self._chromadb = None
        self._embedding_function = None
        self._load_index()

    def _load_index(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._chromadb = chromadb.PersistentClient(path=str(self._index_path))
            self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

            if os.path.exists(self._documents_path):
                with open(self._documents_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_id, doc_data in data.items():
                        self._documents[doc_id] = Document(**doc_data)

            logger.info(f"Knowledge base loaded: {len(self._documents)} documents")
        except ImportError:
            logger.warning("ChromaDB not installed. Run: pip install chromadb")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    def _save_documents(self):
        with open(self._documents_path, "w", encoding="utf-8") as f:
            data = {
                k: {
                    "id": v.id,
                    "name": v.name,
                    "content": v.content,
                    "chunks": v.chunks,
                    "metadata": v.metadata or {},
                }
                for k, v in self._documents.items()
            }
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]

            if len(chunk) < 50 and start + chunk_size < text_len:
                continue

            chunks.append(chunk.strip())
            start += chunk_size - overlap

        return chunks

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def add_document(self, name: str, content: str, metadata: dict = None) -> str:
        if not self._chromadb:
            raise RuntimeError(
                "ChromaDB not initialized. Please install: pip install chromadb"
            )

        doc_id = self._generate_id(name + content[:100])

        chunks = self._chunk_text(content)
        if not chunks:
            raise ValueError("Document content is empty")

        collection = self._chromadb.get_or_create_collection(
            name="knowledge_base", metadata={"description": "Knowledge base for RAG"}
        )

        if doc_id in self._documents:
            collection.delete(ids=[doc_id])

        embeddings = self._embedding_function(chunks)

        collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {
                    "document_id": doc_id,
                    "document_name": name,
                    "chunk_index": i,
                    **(metadata or {}),
                }
                for i in range(len(chunks))
            ],
        )

        self._documents[doc_id] = Document(
            id=doc_id,
            name=name,
            content=content,
            chunks=chunks,
            metadata=metadata or {},
        )
        self._save_documents()

        logger.info(f"Added document: {name} ({len(chunks)} chunks)")
        return doc_id

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._chromadb:
            return []

        collection = self._chromadb.get_or_create_collection(
            name="knowledge_base", metadata={"description": "Knowledge base for RAG"}
        )

        query_embedding = self._embedding_function([query])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 / (1.0 + distance)

                search_results.append(
                    SearchResult(
                        content=doc,
                        score=score,
                        source=metadata.get("document_name", "Unknown"),
                        chunk_index=metadata.get("chunk_index", 0),
                    )
                )

        return search_results

    def delete_document(self, doc_id: str) -> bool:
        if not self._chromadb:
            return False

        if doc_id not in self._documents:
            return False

        collection = self._chromadb.get_or_create_collection(name="knowledge_base")
        collection.delete(
            ids=[f"{doc_id}_{i}" for i in range(len(self._documents[doc_id].chunks))]
        )

        del self._documents[doc_id]
        self._save_documents()

        logger.info(f"Deleted document: {doc_id}")
        return True

    def list_documents(self) -> list[dict]:
        return [
            {
                "id": doc.id,
                "name": doc.name,
                "chunks": len(doc.chunks),
                "size": len(doc.content),
                "metadata": doc.metadata,
            }
            for doc in self._documents.values()
        ]

    def get_context(self, query: str, max_length: int = 2000) -> str:
        results = self.search(query, top_k=5)

        context_parts = []
        total_length = 0

        for result in results:
            if total_length + len(result.content) > max_length:
                break
            context_parts.append(f"[来源: {result.source}]\n{result.content}")
            total_length += len(result.content) + 50

        return "\n\n".join(context_parts)

    def clear(self):
        if self._chromadb:
            try:
                self._chromadb.delete_collection(name="knowledge_base")
            except:
                pass
        self._documents.clear()
        if os.path.exists(self._documents_path):
            os.remove(self._documents_path)
        logger.info("Knowledge base cleared")


knowledge_base = KnowledgeBase()
