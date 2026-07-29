import re
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi


# Keep this in sync with the identical helper in rag_service.py — both
# need to tokenize text the same way for BM25 scores to be meaningful,
# since the index is built from these tokens here and queried with them
# there.
def tokenize_for_bm25(text: str):
    return re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())


def build_bm25_index(chunks):
    """
    Builds a BM25 (lexical/keyword) index over the same chunks used for
    the dense/embedding index. Unlike cosine similarity, BM25 naturally
    down-weights words that appear in most/all chunks (via inverse
    document frequency computed from this specific repository's
    vocabulary) — e.g. "python" or "def" contribute almost nothing to a
    match score if they're in nearly every chunk, without needing a
    hand-maintained stopword list. It's also much better at exact
    identifier/keyword lookups ("what does call_llm do") that embedding
    similarity can sometimes miss.

    Returns None for an empty chunk list rather than raising, since an
    empty repo is already handled as an error earlier in build_index().
    """
    if not chunks:
        return None
    tokenized_corpus = [tokenize_for_bm25(chunk.page_content) for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


class IngestionService:
    def __init__(self, settings):
        self.settings = settings

    def load_python_files(self, repo_dir: Path):
        documents = []

        for file_path in repo_dir.rglob("*.py"):
            # Skip any files in .git directories
            if ".git" in file_path.parts:
                continue
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                if docs:
                    # Add metadata with file path
                    for doc in docs:
                        doc.metadata["source"] = str(file_path.relative_to(repo_dir))
                    documents.extend(docs)
            except Exception:
                # Skip files we can't read
                continue

        if len(documents) > self.settings.max_python_files:
            raise ValueError(
                f"Repository has {len(documents)} Python files, which exceeds "
                f"the maximum allowed of {self.settings.max_python_files}."
            )

        return documents

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        if len(chunks) > self.settings.max_chunks:
            raise ValueError(
                f"Repository produced {len(chunks)} chunks, which exceeds "
                f"the maximum allowed of {self.settings.max_chunks}. Try a "
                "smaller repository, or increase CHUNK_SIZE to produce fewer chunks."
            )

        return chunks

    def create_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.settings.hf_embedding_model,
            model_kwargs={"token": self.settings.huggingfacehub_api_token},
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_index(self, repo_dir: Path):
        documents = self.load_python_files(repo_dir)
        if not documents:
            raise ValueError("No Python files were found in the repository.")
        chunks = self.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.create_embeddings(),
            persist_directory=str(self.settings.chroma_dir),
            collection_name="github_code",
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Built here, while the chunk list is already in memory, rather
        # than having RAGService round-trip through Chroma's .get() to
        # reconstruct the corpus on first use. RAGService still has a
        # lazy fallback (RAGService._ensure_bm25_index) that rebuilds
        # this from the persisted vectorstore if bm25_index/bm25_documents
        # aren't passed through — e.g. after a backend restart, where
        # only the on-disk vectorstore is available.
        bm25_index = build_bm25_index(chunks)

        return {
            "files": len(documents),
            "chunks": len(chunks),
            "vectorstore": vectorstore,
            "bm25_index": bm25_index,
            "bm25_documents": chunks,
        }