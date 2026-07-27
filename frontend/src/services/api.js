const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export async function ingestRepositoryApi(repoUrl) {
  const response = await fetch(`${API_URL}/api/repository/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
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
    headers: {
      "Content-Type": "application/json",
    },
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
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Clear failed.");
  }

  return data;
}