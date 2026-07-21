import logging

import requests
import arxiv
from src.schema import MetaData
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class FetchPDFs:
    """Searches arXiv and filters results down to reachable PDFs."""

    regex = r"[^A-Za-z]"

    def _url_reachable(self, url: str) -> bool:
        """
        Args:
            url (str): PDF URL to check.

        Returns:
            bool: True if a GET request to url succeeds with status < 400.
        """
        try:
            response = requests.get(url, timeout=5)
            return response.status_code < 400
        except RequestException as exc:
            logger.debug("PDF url unreachable %s: %s", url, exc)
            return False

    def __init__(self, topic: str, num_of_pdfs: int, query: str):
        """
        Args:
            topic (str): original research topic (kept for reference, not used in the search itself).
            num_of_pdfs (int): number of reachable PDFs to collect.
            query (str): arXiv search query to run.
        """
        self.topic = topic
        self.num_of_pdfs = num_of_pdfs
        self.query = query

    def fetch(self) -> list[MetaData]:
        """
        Searches arXiv and returns metadata for the first num_of_pdfs reachable results.

        Returns:
            list[MetaData]: metadata for each reachable paper found.
        """
        logger.info("Searching arXiv: query=%r target=%d", self.query, self.num_of_pdfs)

        search = arxiv.Search(
            query=self.query,
            max_results=self.num_of_pdfs
            * 4,  # just to handle results with unreachable pdfs
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        pdf_iterator = client.results(search)
        data: list[MetaData] = []
        n_processed = 0
        result = next(pdf_iterator, None)
        while n_processed < self.num_of_pdfs and result is not None:

            if result.pdf_url is not None and self._url_reachable(result.pdf_url):
                metadata = MetaData(
                    id=result.entry_id,
                    title=result.title,
                    abstract=result.summary,
                    pdf_url=result.pdf_url,
                )
                data.append(metadata)
                n_processed += 1
            else:
                logger.debug("%s is not reachable, skipping", result.entry_id)

            result = next(pdf_iterator, None)

        logger.info(
            "Fetch complete: %d/%d reachable PDFs found", len(data), self.num_of_pdfs
        )

        # Check if we have at lease 10 pdfs on the selected topic
        if len(data) < 10:
            logger.error("Only found %d reachable PDFs (need at least 10)", len(data))
            raise RuntimeError("Fewer than 10 reachable PDFs found for this topic")
        return data
