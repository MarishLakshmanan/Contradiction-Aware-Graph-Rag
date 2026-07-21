import gradio as gr

from src.clients.store import VectorStore
from src.make import start_topic_ingestion
from src.utils.topics import TOPICS_COLLECTION_NAME, list_topics

_pipeline_registry = VectorStore(collection_name=TOPICS_COLLECTION_NAME)

TABLE_HEADERS = [
    "collection_name",
    "topic",
    "num_of_pdfs",
    "max_chars",
    "chunk_overlap",
    "created_at",
]

STATUS_MESSAGES = {
    "ingested": "✅ Ingested '{topic}' into '{collection_name}'.",
    "conflict": "⚠ '{collection_name}' is already registered to a different topic: '{conflicting_topic}'.",
    "failed": "❌ Ingestion failed: {error}",
}


def _validate_ingestion_params(
    num_of_pdfs: int, max_chars: int, chunk_overlap: int
) -> str | None:
    if not (10 <= num_of_pdfs <= 50):
        return "Number of PDFs must be between 10 and 50."
    if not (1000 <= max_chars <= 10000):
        return "Max characters must be between 1000 and 10000."
    if chunk_overlap < 0 or chunk_overlap > max_chars // 2:
        return f"Chunk overlap must be between 0 and {max_chars // 2} (half of max characters)."
    return None


def _topics_table_rows() -> list[list]:
    topics = list_topics(_pipeline_registry)
    return [
        [
            t["collection_name"],
            t["topic"],
            t["num_of_pdfs"],
            t["max_chars"],
            t["chunk_overlap"],
            t["created_at"],
        ]
        for t in topics
    ]


def on_max_chars_change(max_chars: float):
    new_max = int(max_chars) // 2
    return gr.update(maximum=new_max)


async def on_submit(
    topic: str,
    collection_name: str,
    num_of_pdfs: float,
    max_chars: float,
    chunk_overlap: float,
    force: bool,
):
    num_of_pdfs, max_chars, chunk_overlap = (
        int(num_of_pdfs),
        int(max_chars),
        int(chunk_overlap),
    )

    error = _validate_ingestion_params(num_of_pdfs, max_chars, chunk_overlap)
    if error:
        yield f"⚠ {error}", gr.update(), gr.update(interactive=True), gr.update(
            visible=False
        )
        return
    if not topic or not collection_name:
        yield "⚠ Topic and collection name are required.", gr.update(), gr.update(
            interactive=True
        ), gr.update(visible=False)
        return

    yield (
        "Ingestion started — this may take a few minutes…",
        gr.update(),
        gr.update(interactive=False),
        gr.update(visible=False),
    )

    outcome = await start_topic_ingestion(
        topic,
        collection_name,
        max_chars,
        chunk_overlap,
        num_of_pdfs,
        force=force,
    )

    if outcome["status"] == "skipped_similar":
        match = outcome["match"]
        status_text = (
            f"ℹ A similar topic already exists: '{match['topic']}' in "
            f"'{match['collection_name']}' (distance={match['distance']:.3f})."
        )
    else:
        status_text = STATUS_MESSAGES[outcome["status"]].format(**outcome)

    show_force_button = outcome["status"] == "skipped_similar"
    yield (
        status_text,
        _topics_table_rows(),
        gr.update(interactive=True),
        gr.update(visible=show_force_button),
    )


async def on_force_ingest(
    topic: str,
    collection_name: str,
    num_of_pdfs: float,
    max_chars: float,
    chunk_overlap: float,
):
    async for result in on_submit(
        topic, collection_name, num_of_pdfs, max_chars, chunk_overlap, True
    ):
        yield result


with gr.Blocks(title="RAG Pipeline") as demo:
    gr.Markdown("# RAG Ingestion Pipeline")

    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(label="Topic")
            collection_name_input = gr.Textbox(label="Collection name")
            num_of_pdfs_input = gr.Number(
                label="Number of PDFs", minimum=10, maximum=50, step=1, value=10
            )
            max_chars_input = gr.Number(
                label="Max chars per chunk",
                minimum=1000,
                maximum=10000,
                step=100,
                value=1000,
            )
            chunk_overlap_input = gr.Number(
                label="Chunk overlap", minimum=0, maximum=500, step=10, value=200
            )
            submit_btn = gr.Button("Start ingestion")
            force_btn = gr.Button("Ingest anyway", visible=False)
            status_md = gr.Markdown("")
        with gr.Column(scale=2):
            collections_table = gr.Dataframe(
                headers=TABLE_HEADERS, label="Existing collections", interactive=False
            )

    max_chars_input.change(
        on_max_chars_change, inputs=[max_chars_input], outputs=[chunk_overlap_input]
    )

    submit_btn.click(
        on_submit,
        inputs=[
            topic_input,
            collection_name_input,
            num_of_pdfs_input,
            max_chars_input,
            chunk_overlap_input,
            gr.State(False),
        ],
        outputs=[status_md, collections_table, submit_btn, force_btn],
    )

    force_btn.click(
        on_force_ingest,
        inputs=[
            topic_input,
            collection_name_input,
            num_of_pdfs_input,
            max_chars_input,
            chunk_overlap_input,
        ],
        outputs=[status_md, collections_table, submit_btn, force_btn],
    )

    demo.load(lambda: _topics_table_rows(), inputs=None, outputs=[collections_table])

if __name__ == "__main__":
    demo.launch(server_port=7861)
