import pymupdf
from src.schema import Chunk


def chunk_pdf_blocks(
    pdf_bytes: bytes, max_chars: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Chunks the pdf into blocks with an overlap

    Args:
        pdf_bytes (bytes): The pdf in bytes
        max_chars (int, optional): Max characters for one chunk. Defaults to 1000.
        overlap (int, optional): overlap between each chunk. Defaults to 150.

    Returns:
        list[Chunk]: A list of Chunk which is a dict with text pointing to actual text and page is a list which gives the pages that the chunk is from
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    # buffer_segments holds (page_num, text) pieces that make up the current buffer
    buffer_segments = []

    def buffer_text() -> str:
        return "".join(seg_text for _, seg_text in buffer_segments)

    def buffer_pages() -> list:
        return sorted({p for p, _ in buffer_segments})

    def flush():
        """Emit current buffer as a chunk, then seed next buffer with overlap."""
        text = buffer_text().strip()
        if not text:
            return []
        chunks.append({"text": text, "pages": buffer_pages()})

        # Build overlap tail from the end of buffer_segments
        tail = []
        tail_len = 0
        for page_num, seg_text in reversed(buffer_segments):
            if tail_len >= overlap:
                break
            tail.insert(0, (page_num, seg_text))
            tail_len += len(seg_text)

        # If the oldest kept segment overshoots the overlap window, trim it
        if tail_len > overlap and tail:
            page_num, seg_text = tail[0]
            excess = tail_len - overlap
            tail[0] = (page_num, seg_text[excess:])

        return tail

    def split_oversized(text):
        """Hard-split a block so no piece, plus a full overlap tail ahead of
        it, can push a chunk past max_chars."""
        piece_size = max(max_chars - overlap, 1)
        if len(text) <= piece_size:
            return [text]
        return [text[j : j + piece_size] for j in range(0, len(text), piece_size)]

    for i, page in enumerate(doc.pages()):
        for b in page.get_text("blocks"):
            if b[6] == 1:
                continue  # image block, no extractable text

            text = b[4].strip()
            if not text:
                continue

            for piece in split_oversized(text):
                if len(buffer_text()) + len(piece) > max_chars and buffer_segments:
                    buffer_segments = flush()

                buffer_segments.append((i + 1, piece + "\n"))

    if buffer_text().strip():
        chunks.append({"text": buffer_text().strip(), "pages": buffer_pages()})

    return chunks
