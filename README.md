# GitHub Code RAG Assistant

A full-stack application that lets you submit a public GitHub repository, clone and index its Python code, and ask conversational questions about the codebase using Retrieval-Augmented Generation (RAG).

## Architecture

React Frontend -> FastAPI -> GitPython -> Python File Loader -> Code Chunking -> Hugging Face Embeddings API -> ChromaDB (cosine distance) -> Distance + Keyword-Overlap Filter -> Hugging Face Chat Model -> Refusal Backstop

## Requirements

- Python 3.10+
- Node.js 18+
- Git installed and available in PATH
- Hugging Face account and API token
- Internet connection

**Models are not downloaded by this application.** Embeddings and chat generation use the Hugging Face API. The API key is kept in the FastAPI backend.

## 1. Configure Hugging Face API Key

Create a Hugging Face access token with permission to call inference models.

From the project root, copy:

```text
backend/.env.example -> backend/.env
```

Then edit `backend/.env`:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_CHAT_MODEL=HuggingFaceH4/zephyr-7b-beta
REPO_PATH=./repo
CHROMA_PATH=./db
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
RETRIEVAL_K=8
CORS_ORIGINS=http://localhost:5173
```

Never put the Hugging Face token in the React frontend and never commit `.env` to Git.

## 2. Start the Backend

From the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Windows CMD

```bash
python -m venv .venv
.venv\Scripts\activate
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend URLs:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

## 3. Start the React Frontend

Open a second terminal from the project root:

```bash
cd frontend
npm install
npm start

To test the particular file
npm test -- --watchAll=false FileName.test.js

To test all the test files in the application
npm test
```

Open:

```text
http://localhost:5173
```

## 4. Use the Application

1. Enter a public GitHub repository URL, for example:

```text
https://github.com/psf/requests
```

2. Click **Ingest Repository**.
3. The backend clears the previous repository/index, clones the new repository, loads Python files, splits code, generates embeddings through Hugging Face, and stores vectors in ChromaDB.
4. Ask questions such as:

- Where is authentication implemented?
- Explain the main entry point.
- What classes are defined in this project?
- Where is the database connection created?
- Explain the request flow.
- Which file handles configuration?
- How does this function work?

Questions unrelated to the indexed repository — generic programming
requests ("write me a calculator program"), general knowledge, or anything
the repository has no code for — are rejected with a fixed message instead
of being answered from the model's general knowledge:

```text
I couldn't find enough information about that in the indexed repository.
```

This is enforced by the retrieval filters described below, not just by the
prompt, so it holds even if the underlying Hugging Face model doesn't
strictly follow instructions.

## Ingestion Pipeline

```text
GitHub URL
    ↓
GitPython clone
    ↓
./repo
    ↓
Recursively find *.py files, skip .git
    ↓
Load each file with TextLoader
    ↓
RecursiveCharacterTextSplitter (Python-aware)
    ↓
Chunk size: 2000
Overlap: 200
    ↓
Hugging Face Embeddings API
    ↓
ChromaDB (cosine distance metric)
    ↓
./db
```

Note: the codebase also imports and instantiates LangChain's `LanguageParser`,
but it is never actually passed to the loader — `TextLoader` is used
instead, so `LanguageParser` is currently dead code. See `flow.md` for
details on what it would add if wired in.

## Chat Pipeline

```text
Question
    ↓
ChromaDB similarity_search_with_score (top RETRIEVAL_K by cosine distance)
    ↓
Drop chunks with distance above the similarity threshold
    ↓
Drop chunks with no meaningful keyword overlap with the question
    ↓
If nothing survives → return refusal message, sources: [] (LLM is not called)
    ↓
Conversation history + surviving chunks
    ↓
Hugging Face chat model
    ↓
If the model's reply contains the refusal phrase, discard anything after it
    ↓
Answer + source file paths
```

The two-filter step is what stops the assistant from answering generic
programming questions (e.g. "write a simple Python calculator") just
because the indexed repository happens to also be Python — cosine
similarity alone can't tell "same language" apart from "same topic," so a
keyword-overlap check runs after the distance filter.

## Scope Enforcement (Why It Won't Answer Off-Topic Questions)

`RAGService.answer()` never lets the chat model see a question it can't
ground in the repository:

1. **Distance threshold** — `similarity_search_with_score` returns each
   candidate chunk's cosine distance (lower = more similar). Chunks with
   distance above `similarity_threshold` (default `0.8`) are dropped.
2. **Keyword-overlap check** — cosine distance alone can't separate "this
   question is topically similar" from "this question is actually about
   the repository." A generic request like "write a simple Python file"
   can score well under the threshold just because both the question and
   the repository share ordinary Python vocabulary. The overlap check
   strips common words from the question and requires whatever remains to
   appear somewhere in the surviving chunks' content or file paths.
3. **Refusal backstop** — even when context does reach the model, smaller
   open-weight Hugging Face models sometimes emit the refusal sentence and
   then continue generating an answer anyway. If the model's output
   contains the refusal phrase anywhere, the app discards everything else
   and returns just the refusal, with `sources: []`.

If you index a different kind of repository and see too many false
refusals or false positives, tune `similarity_threshold` and the
`GENERIC_TERMS` set in `rag_service.py` using real relevant/irrelevant
query score comparisons, printed to the console on every request.

## Important: One Active Repository

This application maintains one active repository index at a time. When a new repository is ingested, the existing `repo/` and `db/` directories are removed first, then the new repository is cloned and indexed. This prevents code from different repositories from being mixed.

## Clear Data

Use the **Clear Repository and Index** button or call:

```text
DELETE /api/repository
```

The special chat input `clear` also clears the repository, vector index, and conversation memory.

The application uses Python `shutil` instead of Unix-only `rm -rf`, so clearing works on Windows, macOS, and Linux.

## API Endpoints

### Health

```text
GET /api/health
```

### Ingest

```text
POST /api/repository/ingest
```

```json
{
  "repo_url": "https://github.com/user/repository"
}
```

### Chat

```text
POST /api/chat
```

```json
{
  "question": "Where is the main entry point?"
}
```

### Clear

```text
DELETE /api/repository
```

## Troubleshooting

### Hugging Face authentication error

Check `backend/.env`, verify the token starts with `hf_`, and restart the backend.

### Model unavailable

Hugging Face model availability can change. Update `HF_EMBEDDING_MODEL` or `HF_CHAT_MODEL` in `backend/.env` to another compatible model.

### Git is not recognized

Install Git and restart the terminal.

### Port 8000 is busy

Run the backend on another port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then set the frontend API URL:

```bash
# frontend/.env
REACT_URL=http://localhost:3000
```

### ChromaDB problems

Stop the backend, remove the `db/` directory, restart the backend, and ingest the repository again.

### Large repositories

Only Python files are indexed. Very large repositories can still take time and may consume Hugging Face API quota.

## Project Structure

```text
New folder/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── api/
│   │   │   └── routes.py
│   │   └── services/
│   │       ├── repository_service.py
│   │       ├── ingestion_service.py
│   │       └── rag_service.py
│   ├── repo/            # cloned target repository (created at ingest time)
│   ├── db/              # persistent ChromaDB data (created at ingest time)
│   ├── .env
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── .env
│   ├── .env.example
│   └── package.json
└── README.md
```