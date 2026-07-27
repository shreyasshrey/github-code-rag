from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class IngestionService:
    def __init__(self, settings):
        self.settings = settings

    def load_python_files(self, repo_dir: Path):
        documents = []
        parser = LanguageParser(language="python", parser_threshold=500)
        
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
            except Exception as e:
                # Skip files we can't read
                continue
                
        return documents

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        return splitter.split_documents(documents)

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
        return {"files": len(documents), "chunks": len(chunks), "vectorstore": vectorstore}
