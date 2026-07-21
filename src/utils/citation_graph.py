import json
import logging
import re

import httpx
import networkx as nx

from src.clients.semantic_scholar import (
    PaperNotFoundError,
    SemanticScholarClient,
    SemanticScholarThrottleException,
)
from src.schema import MetaData

logger = logging.getLogger(__name__)

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def clean_arxiv_id(raw_id: str) -> str:
    """
    Extracts the bare numeric arXiv id (no version suffix) from an entry_id URL or pdf_url.

    Args:
        raw_id (str): entry_id URL or pdf_url containing an arXiv id.

    Returns:
        str: bare numeric arXiv id, e.g. '2307.08072'.
    """
    match = ARXIV_ID_RE.search(raw_id)
    if not match:
        raise ValueError(f"Could not extract arXiv id from {raw_id!r}")
    return match.group(1)


async def build_citation_graph(
    s2_client: SemanticScholarClient,
    http_client: httpx.AsyncClient,
    metadata: MetaData,
    max_references: int = 10,
    max_citations: int = 10,
) -> tuple[nx.DiGraph, str]:
    """
    One-hop citation graph centered on this article: edges point from a citing
    paper to the paper it cites. Returns an empty graph and an empty center id
    (soft failure) if the arXiv id can't be parsed or Semantic Scholar has no
    record of the paper; any other exception propagates to the caller.

    Semantic Scholar returns every reference/citation with no limit, and some
    papers have hundreds -- capped to max_references/max_citations so the
    serialized graph stored as chroma metadata can't blow past chroma's
    per-field size limit.

    Args:
        s2_client (SemanticScholarClient): client used to fetch citation data.
        http_client (httpx.AsyncClient): shared async HTTP client.
        metadata (MetaData): paper metadata; reads metadata["id"].
        max_references (int): max references to include as graph edges.
        max_citations (int): max citing papers to include as graph edges.

    Returns:
        tuple[nx.DiGraph, str]: the citation graph and the center paper's Semantic Scholar id.
    """
    try:
        arxiv_id = clean_arxiv_id(metadata["id"])
        data = await s2_client.get_citation_graph_data(http_client, arxiv_id)
    except SemanticScholarThrottleException:
        logger.warning(
            "Semantic Scholar rate limit reached, unable to build citation graph for %s",
            metadata["id"],
        )
        return nx.DiGraph(), ""
    except (ValueError, PaperNotFoundError):
        return nx.DiGraph(), ""

    graph = nx.DiGraph()
    center = data["paperId"]
    graph.add_node(center, title=data.get("title"))
    for ref in (data.get("references") or [])[:max_references]:
        if ref.get("paperId"):
            graph.add_node(ref["paperId"], title=ref.get("title"))
            graph.add_edge(center, ref["paperId"])
    for cit in (data.get("citations") or [])[:max_citations]:
        if cit.get("paperId"):
            graph.add_node(cit["paperId"], title=cit.get("title"))
            graph.add_edge(cit["paperId"], center)
    return graph, center


def serialize_graph(graph: nx.DiGraph) -> str:
    """
    Args:
        graph (nx.DiGraph): citation graph to serialize.

    Returns:
        str: JSON string suitable for storing as a chroma metadata field.
    """
    return json.dumps(nx.node_link_data(graph, edges="edges"))


def deserialize_graph(payload: str) -> nx.DiGraph:
    """
    Args:
        payload (str): JSON string as produced by serialize_graph.

    Returns:
        nx.DiGraph: the reconstructed citation graph.
    """
    return nx.node_link_graph(json.loads(payload), edges="edges")
