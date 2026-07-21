import os
from unittest import mock

import pymupdf

from src.utils.chunking import chunk_pdf_blocks

PDF_PATH = os.path.join(os.path.dirname(__file__), "asset", "pdf.pdf")

with open(PDF_PATH, mode="rb") as file:
    pdf_bytes = file.read()


"""
Tests
1. Check whether the text given back is within the max_chars limit
2. Check if there's true overlap
3. Check whether the images are handled correctly
"""


def test_max_chars_limit(max_chars: int = 500, overlap: int = 100):
    chunks = chunk_pdf_blocks(pdf_bytes, max_chars=max_chars, overlap=overlap)
    assert chunks, "no chunks produced"

    for i, chunk in enumerate(chunks):
        assert (
            len(chunk["text"]) <= max_chars
        ), f"chunk {i} is {len(chunk['text'])} chars, exceeds max_chars={max_chars}"

    print(f"[max_chars] OK - all {len(chunks)} chunks within {max_chars} chars")


def _longest_suffix_prefix_match(a: str, b: str, max_len: int) -> int:
    """Length of the longest suffix of a that's also a prefix of b, capped at max_len."""
    for length in range(min(max_len, len(a), len(b)), 0, -1):
        if a[-length:] == b[:length]:
            return length
    return 0


def test_overlap(max_chars: int = 500, overlap: int = 100):
    chunks = chunk_pdf_blocks(pdf_bytes, max_chars=max_chars, overlap=overlap)
    assert len(chunks) > 1, "need at least two chunks to check overlap"

    # .strip() on chunk text can shift the tail by a couple of chars relative
    # to the raw buffer the tail was actually cut from, so allow small slack.
    slack = 5
    for i in range(len(chunks) - 1):
        prev_text = chunks[i]["text"]
        next_text = chunks[i + 1]["text"]

        match_len = _longest_suffix_prefix_match(prev_text, next_text, overlap + slack)
        assert match_len >= overlap - slack, (
            f"no meaningful overlap found between chunk {i} and chunk {i + 1} "
            f"(longest match: {match_len} chars, expected ~{overlap})"
        )

    print(f"[overlap] OK - overlap present between all {len(chunks) - 1} chunk pairs")


def test_images_are_skipped():
    """Synthetic PDF stand-in: real asset/pdf.pdf has no images, so we fake a
    page with a text block and an image block to exercise the skip path."""

    class FakePage:
        def __init__(self, blocks):
            self._blocks = blocks

        def get_text(self, kind):
            assert kind == "blocks"
            return self._blocks

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def pages(self):
            return iter(self._pages)

    blocks = [
        (0, 0, 10, 10, "Some real text.\n", 0, 0),
        (0, 10, 10, 20, "<image: DeviceRGB, 100x100>", 1, 1),
        (0, 20, 10, 30, "More real text.\n", 2, 0),
    ]
    fake_doc = FakeDoc([FakePage(blocks)])

    with mock.patch.object(pymupdf, "open", return_value=fake_doc):
        chunks = chunk_pdf_blocks(b"fake-bytes", max_chars=1000, overlap=0)

    combined = " ".join(chunk["text"] for chunk in chunks)
    assert "<image:" not in combined, "image block text leaked into a chunk"
    assert "Some real text." in combined
    assert "More real text." in combined

    print("[images] OK - image blocks are filtered out of chunk text")


if __name__ == "__main__":
    test_max_chars_limit()
    test_overlap()
    test_images_are_skipped()
