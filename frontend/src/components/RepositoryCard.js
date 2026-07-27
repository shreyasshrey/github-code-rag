import React from "react";

const RepositoryCard = ({
  repoUrl,
  setRepoUrl,
  ingestRepository,
  clearRepository,
  ingesting,
  chatLoading,
  status,
}) => {
  return (
    <section className="card">
      <h2>Repository URL</h2>

      <form onSubmit={ingestRepository} className="repo-form">
        <input
          type="url"
          placeholder="https://github.com/user/repository"
          value={repoUrl}
          onChange={(e) => setRepoUrl(e.target.value)}
        />

        <button className="ingest-button" disabled={ingesting || chatLoading}>
          {ingesting ? "Indexing Repository..." : "Ingest Repository"}
        </button>
      </form>

      <button
        className="secondary-button"
        onClick={clearRepository}
        disabled={ingesting || chatLoading}
      >
        Clear Repository and Index
      </button>

      {status && <p className="status">{status}</p>}
    </section>
  );
};

export default RepositoryCard;
