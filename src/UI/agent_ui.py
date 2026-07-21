import uuid
from datetime import datetime, timezone

import gradio as gr
from langgraph_sdk import get_client

from src.clients.store import VectorStore
from src.utils.topics import TOPICS_COLLECTION_NAME, list_topics
from settings import settings

client_url = settings.langgraph_client_url
graph_client = (
    get_client(url=client_url, api_key=settings.langgraph_client_api)
    if settings.langgraph_client_api
    else get_client(url=client_url)
)

_topics_registry = VectorStore(collection_name=TOPICS_COLLECTION_NAME)

NODE_LABELS = {
    "rewrite_query": "Rewriting your question…",
    "retrieve_chunks": "Retrieving relevant passages…",
    "expand_related_papers": "Expanding citation graph…",
    "generate_answer": "Generating answer…",
}

_EMPTY_STATE = {"sessions": [], "active_session_id": None}


def _get_topic_choices() -> list[tuple[str, str]]:
    topics = list_topics(_topics_registry)
    return [
        (f"{t['topic']} ({t['collection_name']})", t["collection_name"]) for t in topics
    ]


def _find_session(sessions_state: dict, session_id: str | None) -> dict | None:
    if session_id is None:
        return None
    for session in sessions_state["sessions"]:
        if session["id"] == session_id:
            return session
    return None


def _session_choices(sessions_state: dict) -> list[tuple[str, str]]:
    return [(session["title"], session["id"]) for session in sessions_state["sessions"]]


def on_load(sessions_state: dict):
    if sessions_state is None:
        sessions_state = dict(_EMPTY_STATE)
    active_session = _find_session(
        sessions_state, sessions_state.get("active_session_id")
    )
    history = active_session["messages"] if active_session else []
    topic_label = (
        f"Topic: {active_session['topic']} ({active_session['collection_name']})"
        if active_session
        else "Topic: (start a new chat to pick one)"
    )
    return (
        sessions_state,
        gr.update(choices=_get_topic_choices()),
        gr.update(
            choices=_session_choices(sessions_state),
            value=sessions_state.get("active_session_id"),
        ),
        history,
        topic_label,
    )


def on_new_chat(topic_collection_name: str | None, sessions_state: dict):
    if not topic_collection_name:
        gr.Warning("Select a topic before starting a new chat.")
        return sessions_state, gr.update(), gr.update(), gr.update()

    topics = {t["collection_name"]: t["topic"] for t in list_topics(_topics_registry)}
    topic_text = topics.get(topic_collection_name, topic_collection_name)

    session = {
        "id": str(uuid.uuid4()),
        "title": f"{topic_text[:40]} chat {len(sessions_state['sessions']) + 1}",
        "collection_name": topic_collection_name,
        "topic": topic_text,
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    sessions_state = dict(sessions_state)
    sessions_state["sessions"] = sessions_state["sessions"] + [session]
    sessions_state["active_session_id"] = session["id"]

    return (
        sessions_state,
        gr.update(choices=_session_choices(sessions_state), value=session["id"]),
        [],
        f"Topic: {session['topic']} ({session['collection_name']})",
    )


def on_switch_session(session_id: str | None, sessions_state: dict):
    session = _find_session(sessions_state, session_id)
    sessions_state = dict(sessions_state)
    sessions_state["active_session_id"] = session_id
    if session is None:
        return sessions_state, [], "Topic: (start a new chat to pick one)"
    return (
        sessions_state,
        session["messages"],
        f"Topic: {session['topic']} ({session['collection_name']})",
    )


async def respond(user_message: str, sessions_state: dict, chatbot_history: list):
    active_id = sessions_state.get("active_session_id")
    session = _find_session(sessions_state, active_id)

    if not user_message:
        yield chatbot_history, "", sessions_state, gr.update()
        return

    if session is None:
        chatbot_history = chatbot_history + [
            {"role": "assistant", "content": "Start a new chat and pick a topic first."}
        ]
        yield chatbot_history, "", sessions_state, gr.update()
        return

    chatbot_history = chatbot_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    answer = ""
    label = ""

    try:
        async for part in graph_client.runs.stream(
            thread_id=None,
            assistant_id="agent",
            input={"query": user_message},
            context={"collection_name": session["collection_name"]},
            stream_mode=["updates", "messages-tuple"],
        ):
            if (
                part.event.startswith("updates")
                and isinstance(part.data, dict)
                and part.data
            ):
                node = next(iter(part.data))
                label = f"Processing: {NODE_LABELS.get(node, node)}"
            elif part.event.startswith("messages"):
                message_chunk, metadata = part.data
                if metadata.get("langgraph_node") != "generate_answer":
                    continue
                answer += message_chunk.get("content", "")
                chatbot_history[-1]["content"] = answer
            yield chatbot_history, label, gr.update(), ""
    except Exception as exc:
        chatbot_history[-1][
            "content"
        ] = f"⚠ Could not reach the agent server ({exc}). Is `uv run langgraph dev` running?"
        yield chatbot_history, "", sessions_state, ""
        return

    session["messages"] = chatbot_history
    sessions_state = dict(sessions_state)
    sessions_state["sessions"] = [
        session if s["id"] == active_id else s for s in sessions_state["sessions"]
    ]
    yield chatbot_history, "", sessions_state, ""


with gr.Blocks(title="RAG Chat") as demo:
    browser_state = gr.BrowserState(_EMPTY_STATE, storage_key="rag_chat_sessions_v1")

    gr.Markdown("# RAG Chat")

    with gr.Row():
        with gr.Column(scale=1):
            topic_dropdown = gr.Dropdown(label="Topic", choices=[])
            new_chat_btn = gr.Button("New chat")
            session_picker = gr.Dropdown(label="Your chats", choices=[])
        with gr.Column(scale=3):
            topic_label = gr.Markdown("Topic: (start a new chat to pick one)")
            chatbot = gr.Chatbot(height=450)
            status_label = gr.Markdown("")
            message_box = gr.Textbox(
                label="Your question", placeholder="Ask something…"
            )

    demo.load(
        on_load,
        inputs=[browser_state],
        outputs=[browser_state, topic_dropdown, session_picker, chatbot, topic_label],
    )

    new_chat_btn.click(
        on_new_chat,
        inputs=[topic_dropdown, browser_state],
        outputs=[browser_state, session_picker, chatbot, topic_label],
    )

    session_picker.change(
        on_switch_session,
        inputs=[session_picker, browser_state],
        outputs=[browser_state, chatbot, topic_label],
    )

    message_box.submit(
        respond,
        inputs=[message_box, browser_state, chatbot],
        outputs=[chatbot, status_label, browser_state, message_box],
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)
