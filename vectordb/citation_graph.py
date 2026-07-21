import json
import re

import httpx
import networkx as nx

from clients.semantic_scholar import PaperNotFoundError, SemanticScholarClient
from vectordb.schema import MetaData

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def clean_arxiv_id(raw_id: str) -> str:
    """Extract the bare numeric arXiv id (no version suffix) from an entry_id URL or pdf_url."""
    match = ARXIV_ID_RE.search(raw_id)
    if not match:
        raise ValueError(f"Could not extract arXiv id from {raw_id!r}")
    return match.group(1)


async def build_citation_graph(
    s2_client: SemanticScholarClient, http_client: httpx.AsyncClient, metadata: MetaData
) -> tuple[nx.DiGraph, str]:
    """
    One-hop citation graph centered on this article: edges point from a citing
    paper to the paper it cites. Returns an empty graph and an empty center id
    (soft failure) if the arXiv id can't be parsed or Semantic Scholar has no
    record of the paper; any other exception propagates to the caller.
    """
    try:
        arxiv_id = clean_arxiv_id(metadata["id"])
        data = await s2_client.get_citation_graph_data(http_client, arxiv_id)
    except (ValueError, PaperNotFoundError):
        return nx.DiGraph(), ""

    graph = nx.DiGraph()
    center = data["paperId"]
    graph.add_node(center, title=data.get("title"))
    for ref in data.get("references") or []:
        if ref.get("paperId"):
            graph.add_node(ref["paperId"], title=ref.get("title"))
            graph.add_edge(center, ref["paperId"])
    for cit in data.get("citations") or []:
        if cit.get("paperId"):
            graph.add_node(cit["paperId"], title=cit.get("title"))
            graph.add_edge(cit["paperId"], center)
    return graph, center


def serialize_graph(graph: nx.DiGraph) -> str:
    return json.dumps(nx.node_link_data(graph, edges="edges"))


def deserialize_graph(payload: str) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(payload), edges="edges")
