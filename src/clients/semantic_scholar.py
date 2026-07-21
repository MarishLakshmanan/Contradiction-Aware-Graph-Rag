import httpx

from settings import settings


class PaperNotFoundError(Exception):
    """Semantic Scholar has no record of this paper (404) -- soft/non-fatal."""


class SemanticScholarThrottleException(Exception):
    """We reached the Semantic scholars 1req/s limit"""


class SemanticScholarClient:
    """Wraps Semantic Scholar's Graph API. Transport only."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "paperId,title,references.paperId,references.title,citations.paperId,citations.title"

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key (str | None): Semantic Scholar API key, sent as the x-api-key
                header when present. Optional -- requests are sent unauthenticated
                (and more likely to be throttled) when None.
        """
        self.api_key = api_key

    async def get_citation_graph_data(
        self, client: httpx.AsyncClient, arxiv_id: str
    ) -> dict:
        """
        Fetches a paper's references and citations from the Graph API.

        Args:
            client (httpx.AsyncClient): shared async HTTP client to send the request on.
            arxiv_id (str): bare numeric arXiv id, no version suffix, e.g. '2307.08072'.

        Returns:
            dict: JSON response with paperId/title/references/citations fields.
        """
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        response = await client.get(
            f"{self.BASE_URL}/ARXIV:{arxiv_id}",
            params={"fields": self.FIELDS},
            headers=headers,
            timeout=15,
        )
        if response.status_code == 404:
            raise PaperNotFoundError(arxiv_id)
        if response.status_code == 429:
            raise SemanticScholarThrottleException(arxiv_id)
        response.raise_for_status()
        return response.json()
