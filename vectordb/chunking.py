# So we are gonna do a markdown splitter then an character splitter

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


class ChunkMarkdown:

    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]

    def chunk(
        self, md_content: str, chunk_size: int = 700, overlap: int = 100
    ) -> list[str]:
        """
        This function will first chunk the markdown file and then again chunk it using a recursive splitter
        """

        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )

        md_split = md_splitter.split_text(md_content)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            is_separator_regex=False,
            separators=[
                " ",
                ".",
                "\n\n",
            ],  # having just \n chunks the heading separately and the default separator has that so overwrite the default separator
        )
        split_text = recursive_splitter.split_documents(md_split)
        return [text.page_content for text in split_text]
