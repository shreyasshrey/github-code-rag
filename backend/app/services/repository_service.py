import re
import shutil
from pathlib import Path
from git import Repo
import stat
import os


def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Only accepts a plain GitHub repository URL:
#   https://github.com/<owner>/<repo>
#   https://github.com/<owner>/<repo>.git
#   https://github.com/<owner>/<repo>/
# Anything else (branch links, subfolder links, blob/file links, other
# hosts, etc.) is rejected.
_GITHUB_REPO_URL_RE = re.compile(
    r"^https://github\.com/[^/\s]+/[^/\s]+?(?:\.git)?/?$"
)

EXAMPLE_URL = "https://github.com/psf/requests"


class InvalidRepositoryUrlError(ValueError):
    def __init__(self, given_url: str):
        self.given_url = given_url
        super().__init__(
            f"Invalid repository URL: '{given_url}'. "
            f"Please provide the valid URL, for example: {EXAMPLE_URL} "
            "(not a link to a branch, folder, or file inside the repository)."
        )


def validate_github_url(url: str) -> str:
    """
    Validates that `url` is a plain GitHub repository URL and returns it
    with any trailing slash/.git removed. Raises InvalidRepositoryUrlError
    with the offending URL and an example of the expected format if it
    isn't (e.g. a /tree/<branch>/<path> or /blob/<branch>/<path> link
    copied from GitHub's web UI).
    """
    cleaned = (url or "").strip()
    if not cleaned or not _GITHUB_REPO_URL_RE.match(cleaned):
        raise InvalidRepositoryUrlError(cleaned)

    return re.sub(r"(\.git)?/?$", "", cleaned)


class RepositoryService:
    def __init__(self, repo_dir: Path, chroma_dir: Path):
        self.repo_dir = repo_dir
        self.chroma_dir = chroma_dir

    def clear(self):
        if self.repo_dir.exists():
            shutil.rmtree(self.repo_dir, onerror=handle_remove_readonly)
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir, onerror=handle_remove_readonly)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    def clone(self, repo_url: str):
        clone_url = validate_github_url(repo_url)

        self.clear()

        try:
            Repo.clone_from(clone_url, self.repo_dir)
        except Exception as exc:
            raise ValueError(f"Failed to clone {clone_url}: {exc}") from exc

        return self.repo_dir