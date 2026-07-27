import re

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


REFUSAL_MESSAGE = (
    "I couldn't find enough information about that "
    "in the indexed repository."
)

# Words that are too generic to count as "meaningful overlap" between
# a question and retrieved repo chunks (e.g. "write a simple python
# program" shares "python"/"write"/"simple" with almost any codebase).
GENERIC_TERMS = {
    "python", "java", "javascript", "file", "files", "write", "writing",
    "simple", "program", "code", "coding", "function", "functions",
    "class", "classes", "script", "app", "application", "create",
    "make", "build", "example", "how", "what", "why", "the", "and",
    "for", "with", "this", "that",
}


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

    @staticmethod
    def _has_meaningful_overlap(question: str, documents) -> bool:
        """
        Reject matches that are only *topically* similar (e.g. both
        are "Python code") but share no actual repo-specific
        vocabulary with the question. Distance-based retrieval alone
        can't tell "asks about Python" apart from "asks about this
        specific repository" — this catches that gap.
        """
        q_tokens = set(re.findall(r"[a-z_][a-z0-9_]{2,}", question.lower()))
        q_tokens -= GENERIC_TERMS

        if not q_tokens:
            # Nothing specific enough in the question to check against —
            # don't block on overlap in this case, let the distance
            # filter be the sole gate.
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

        # ------------------------------------------------
        # 1. Retrieve relevant documents with scores
        # ------------------------------------------------

        results = (
            self.vectorstore
            .similarity_search_with_score(
                question,
                k=self.settings.retrieval_k,
            )
        )
        
        # ------------------------------------------------
        # 2. No documents found
        # ------------------------------------------------

        if not results:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
            }

        for document, score in results:
            print(
                f"SCORE: {score:.4f} | "
                f"SOURCE: {document.metadata.get('source')}"
            )

        # ------------------------------------------------
        # 3. Apply relevance threshold (distance: lower is better)
        # ------------------------------------------------

        relevant_results = [
            (document, score)
            for document, score in results
            if score <= self.similarity_threshold
        ]

        # ------------------------------------------------
        # 3b. Reject topically-similar-but-not-actually-related matches
        # ------------------------------------------------

        if relevant_results:
            docs_only = [doc for doc, _ in relevant_results]
            if not self._has_meaningful_overlap(question, docs_only):
                print(
                    {
                        "-------------------------rejected: no meaningful overlap------------------------------": question
                    }
                )
                relevant_results = []

        # ------------------------------------------------
        # 4. Reject unrelated questions
        # ------------------------------------------------

        if not relevant_results:
            return {
                "answer": REFUSAL_MESSAGE,
                "sources": [],
            }

        # ------------------------------------------------
        # 5. Build context
        # ------------------------------------------------

        documents = [
            document
            for document, score in relevant_results
        ]

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