import re
from typing import List, Optional, Tuple

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi


REFUSAL_MESSAGE = (
    "I couldn't find enough information about that "
    "in the indexed repository."
)

# Words that are too generic to count as "meaningful overlap" between
# a question and retrieved repo chunks (e.g. "write a simple python
# program" shares "python"/"write"/"simple" with almost any codebase).
# Kept as a cheap final safety net even with hybrid retrieval below —
# BM25's IDF weighting already discounts these automatically within a
# given repository's vocabulary, but this check is nearly free and adds
# defense in depth.
GENERIC_TERMS = {
    "python", "java", "javascript", "file", "files", "write", "writing",
    "simple", "program", "code", "coding", "function", "functions",
    "class", "classes", "script", "app", "application", "create",
    "make", "build", "example", "how", "what", "why", "the", "and",
    "for", "with", "this", "that",
}

# Standard Reciprocal Rank Fusion constant. Larger values flatten out the
# difference between high and low ranks; 60 is the commonly-used default
# from the original RRF paper and works fine without tuning.
RRF_K = 60


# Keep this in sync with the identical helper in ingestion_service.py —
# both need to tokenize text the same way for BM25 scores to be
# meaningful, since the index is built from these tokens there and
# queried with them here.
def tokenize_for_bm25(text: str):
    return re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())


class RAGService:

    def __init__(self, settings):
        self.settings = settings
        self.history = []
        self.vectorstore = None
        self.llm = None

        # Cosine distance from Chroma: LOWER is more similar.
        # Tune this based on your embedding model using real
        # relevant-query vs irrelevant-query score baselines.
        self.similarity_threshold = 0.8

        # BM25 (lexical) index, built lazily from whatever vectorstore is
        # currently set — see _ensure_bm25_index(). Can also be injected
        # directly via set_index() right after ingestion, which is
        # cheaper since IngestionService.build_index() already has the
        # chunk list in memory and doesn't need to re-fetch it from Chroma.
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_documents: Optional[List[Document]] = None
        self._bm25_source_vectorstore = None

    def _embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.settings.hf_embedding_model,
            model_kwargs={
                "token": self.settings.huggingfacehub_api_token
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    def load_vectorstore(self):

        if not self.settings.chroma_dir.exists():
            raise ValueError(
                "No repository has been indexed yet."
            )

        self.vectorstore = Chroma(
            persist_directory=str(
                self.settings.chroma_dir
            ),
            embedding_function=self._embeddings(),
            collection_name="github_code",
        )

    def set_index(self, vectorstore, bm25_index=None, bm25_documents=None):
        """
        Preferred way to wire up a freshly built index (e.g. right after
        ingestion): pass the vectorstore plus the bm25_index/bm25_documents
        returned by IngestionService.build_index(), so the BM25 side
        doesn't need to be rebuilt from scratch on the next question.

        Just assigning `.vectorstore = ...` directly still works — the
        BM25 index will be rebuilt lazily from the vectorstore's stored
        documents on the next call to answer() instead.
        """
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.bm25_documents = bm25_documents
        self._bm25_source_vectorstore = vectorstore if bm25_index is not None else None

    def _get_llm(self):

        if self.llm is None:

            endpoint = HuggingFaceEndpoint(
                repo_id=self.settings.hf_chat_model,
                huggingfacehub_api_token=(
                    self.settings.huggingfacehub_api_token
                ),
                temperature=0.1,
                max_new_tokens=700,
            )

            self.llm = ChatHuggingFace(
                llm=endpoint
            )

        return self.llm

    def clear_memory(self):
        self.history = []

    def _ensure_bm25_index(self):
        """
        Makes sure self.bm25_index/self.bm25_documents are built and
        correspond to the *current* self.vectorstore. Rebuilds are cheap
        to detect (an identity check) but not cheap to do (re-fetches and
        re-tokenizes the whole corpus), so this only actually rebuilds
        when self.vectorstore has changed since the index was last built
        — e.g. after a new ingest, or after load_vectorstore() reloads
        from disk following a restart.
        """
        if self.vectorstore is None:
            return

        if self.bm25_index is not None and self._bm25_source_vectorstore is self.vectorstore:
            return  # already built for this exact vectorstore instance

        raw = self.vectorstore.get(include=["documents", "metadatas"])
        contents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        documents = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(contents, metadatas)
        ]

        self.bm25_documents = documents
        self.bm25_index = (
            BM25Okapi([tokenize_for_bm25(doc.page_content) for doc in documents])
            if documents
            else None
        )
        self._bm25_source_vectorstore = self.vectorstore

    def _bm25_search(self, question: str, k: int) -> List[Tuple[Document, float]]:
        if self.bm25_index is None or not self.bm25_documents:
            return []

        tokenized_question = tokenize_for_bm25(question)
        if not tokenized_question:
            return []

        scores = self.bm25_index.get_scores(tokenized_question)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        # A score of 0 means none of the question's tokens appear in that
        # chunk at all — not a real match, just an artifact of taking the
        # top-k regardless. Drop those rather than treating "closest to
        # nothing" as if it were relevant.
        return [
            (self.bm25_documents[i], float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _doc_key(doc: Document) -> Tuple[str, str]:
        return (doc.metadata.get("source", "unknown"), doc.page_content)

    @classmethod
    def _reciprocal_rank_fusion(
        cls,
        dense_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int = RRF_K,
    ) -> List[Tuple[Document, float]]:
        """
        Merges the dense (embedding/cosine-distance) and BM25 (lexical)
        result lists into a single ranking using Reciprocal Rank Fusion:
        each document's fused score is the sum of 1/(k + rank) across
        every list it appears in, using rank position rather than the
        raw scores. Cosine distance and BM25 score live on incomparable
        scales, so combining by rank sidesteps having to normalize or
        weight one against the other.
        """
        fused_scores = {}
        doc_by_key = {}

        for result_list in (dense_results, bm25_results):
            for rank, (doc, _score) in enumerate(result_list, start=1):
                key = cls._doc_key(doc)
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
                doc_by_key.setdefault(key, doc)

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
        return [(doc_by_key[key], fused_scores[key]) for key in ranked_keys]

    @staticmethod
    def _has_meaningful_overlap(question: str, documents) -> bool:
        """
        Reject matches that are only *topically* similar (e.g. both
        are "Python code") but share no actual repo-specific
        vocabulary with the question. Kept as a cheap final safety net
        even with hybrid retrieval — see the GENERIC_TERMS comment above.
        """
        q_tokens = set(re.findall(r"[a-z_][a-z0-9_]{2,}", question.lower()))
        q_tokens -= GENERIC_TERMS

        if not q_tokens:
            # Nothing specific enough in the question to check against —
            # don't block on overlap in this case, let the retrieval
            # filters above be the sole gate.
            return True

        repo_tokens = set()
        for doc in documents:
            repo_tokens |= set(
                re.findall(r"[a-z_][a-z0-9_]{2,}", doc.page_content.lower())
            )
            source = doc.metadata.get("source", "")
            repo_tokens |= set(re.findall(r"[a-z_][a-z0-9_]{2,}", source.lower()))

        return len(q_tokens & repo_tokens) > 0

    def answer(self, question: str):

        if self.vectorstore is None:
            self.load_vectorstore()

        self._ensure_bm25_index()

        # ------------------------------------------------
        # 1. Dense (embedding) retrieval with cosine distance
        # ------------------------------------------------

        dense_raw = (
            self.vectorstore
            .similarity_search_with_score(
                question,
                k=self.settings.retrieval_k,
            )
        )

        # ------------------------------------------------
        # 2. No documents in the index at all
        # ------------------------------------------------

        if not dense_raw and not self.bm25_documents:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
            }

        # ------------------------------------------------
        # 3. Filter dense results by distance threshold
        #    (lower cosine distance = more similar)
        # ------------------------------------------------

        dense_relevant = [
            (document, score)
            for document, score in dense_raw
            if score <= self.similarity_threshold
        ]

        # ------------------------------------------------
        # 3b. Lexical (BM25) retrieval — catches exact
        #     identifier/keyword matches that embedding
        #     similarity sometimes misses, and requires at
        #     least one real token match by construction.
        # ------------------------------------------------

        bm25_relevant = self._bm25_search(question, k=self.settings.retrieval_k)

        # ------------------------------------------------
        # 3c. Fuse both rankings, keep the top retrieval_k
        # ------------------------------------------------

        fused = self._reciprocal_rank_fusion(dense_relevant, bm25_relevant)
        fused = fused[: self.settings.retrieval_k]

        # ------------------------------------------------
        # 3d. Reject topically-similar-but-not-actually-related matches
        # ------------------------------------------------

        if fused:
            docs_only = [doc for doc, _ in fused]
            if not self._has_meaningful_overlap(question, docs_only):
                fused = []

        # ------------------------------------------------
        # 4. Reject unrelated questions
        # ------------------------------------------------

        if not fused:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
            }

        # ------------------------------------------------
        # 5. Build context
        # ------------------------------------------------

        documents = [document for document, _score in fused]

        context = "\n\n".join(
            (
                f"FILE: "
                f"{doc.metadata.get('source', 'unknown')}\n"
                f"{doc.page_content}"
            )
            for doc in documents
        )

        # ------------------------------------------------
        # 6. Conversation history
        # ------------------------------------------------

        history = "\n".join(
            f"User: {user_question}\n"
            f"Assistant: {assistant_answer}"
            for user_question, assistant_answer
            in self.history[-6:]
        )

        # ------------------------------------------------
        # 7. Strict grounding prompt
        # ------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a GitHub repository code analysis assistant.

                STRICT SCOPE:

                You can answer ONLY questions that can be answered
                from the indexed GitHub repository.

                The retrieved repository context is your ONLY source of truth.

                You MUST NOT:

                - Use general knowledge.
                - Generate new code that does not exist in the repository.
                - Answer generic programming questions.
                - Answer questions about Python, Java, JavaScript,
                or any other language unless the answer is supported
                by the indexed repository.
                - Answer weather, news, sports, politics, current events,
                or unrelated questions.
                - Invent files, classes, functions, variables, APIs,
                or application behavior.

                If the retrieved context does not contain enough information
                to answer the question, respond exactly:

                I couldn't find enough information about that
                in the indexed repository.

                Always mention relevant file paths when possible.

                Retrieved repository context:
                {context}

                Previous conversation:
                {history}
                """,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

        # ------------------------------------------------
        # 8. Generate answer
        # ------------------------------------------------

        response = (
            prompt
            | self._get_llm()
        ).invoke(
            {
                "history": (
                    history
                    if history
                    else "No previous conversation."
                ),
                "context": context,
                "question": question,
            }
        )

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        # ------------------------------------------------
        # 8b. Backstop: if the model hedges with the refusal
        # phrase and then keeps generating anyway, don't let
        # the extra text reach the user. This does not depend
        # on the model's instruction-following quality.
        # ------------------------------------------------

        if REFUSAL_MESSAGE.strip() in " ".join(answer.split()):
            answer = REFUSAL_MESSAGE

        # ------------------------------------------------
        # 9. Store history
        # ------------------------------------------------

        self.history.append(
            (
                question,
                answer,
            )
        )

        # ------------------------------------------------
        # 10. Return answer and sources
        # ------------------------------------------------

        return {
            "answer": answer,
            "sources": (
                []
                if answer == REFUSAL_MESSAGE
                else list(
                    {
                        doc.metadata.get(
                            "source",
                            "unknown",
                        )
                        for doc in documents
                    }
                )
            ),
        }