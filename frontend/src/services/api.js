const API_URL = process.env.REACT_APP_API_URL;

const API_KEY = process.env.REACT_APP_API_KEY;

const JSON_HEADERS = {
  "Content-Type": "application/json",
  "X-API-Key": API_KEY,
};

const API_HEADERS = {
  "X-API-Key": API_KEY,
};

export async function ingestRepositoryApi(repoUrl) {
  const response = await fetch(`${API_URL}/api/repository/ingest`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      repo_url: repoUrl,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Ingestion failed.");
  }

  return data;
}

export async function askQuestion(question) {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      question,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Chat request failed.");
  }

  return data;
}

export async function clearRepositoryApi() {
  const response = await fetch(`${API_URL}/api/repository`, {
    method: "DELETE",
    headers: API_HEADERS
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Clear failed.");
  }

  return data;
}