import httpx
import os


class PaperNotFoundError(Exception):
    """Semantic Scholar has no record of this paper (404) -- soft/non-fatal."""


class SemanticScholarClient:
    """Wraps Semantic Scholar's Graph API. Transport only."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "paperId,title,references.paperId,references.title,citations.paperId,citations.title"

    def __init__(self, api_key: str | None = None):
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            raise Exception("API key for Semantic Scholar is not")
        self.api_key = api_key

    async def get_citation_graph_data(
        self, client: httpx.AsyncClient, arxiv_id: str
    ) -> dict:
        """arxiv_id must be the bare numeric id, no version suffix, e.g. '2307.08072'."""
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = await client.get(
            f"{self.BASE_URL}/ARXIV:{arxiv_id}",
            params={"fields": self.FIELDS},
            headers=headers,
            timeout=15,
        )
        if response.status_code == 404:
            raise PaperNotFoundError(arxiv_id)
        response.raise_for_status()
        return response.json()
