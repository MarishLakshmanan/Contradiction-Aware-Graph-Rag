import requests
import arxiv
from vectordb.schema import MetaData
from requests.exceptions import ReadTimeout


class FetchPDFs:
    regex = r"[^A-Za-z]"

    def _url_reachable(self, url: str) -> bool:
        try:
            response = requests.get(url, timeout=5)
            return response.status_code < 400
        except ReadTimeout:
            return False

    def __init__(self, topic: str, num_of_pdfs: int, query: str):
        self.topic = topic
        self.num_of_pdfs = num_of_pdfs
        self.query = query

    def fetch(self) -> list[MetaData]:

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
                print(f"{result.entry_id} is not reachable")

            result = next(pdf_iterator)

        # Check if we have at lease 10 pdfs on the selected topic
        if len(data) < 10:
            raise RuntimeError("Fewer than 10 reachable PDFs found for this topic")
        return data
