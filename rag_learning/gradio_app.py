from __future__ import annotations

import os

import httpx
import gradio as gr

API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")


def _build_curl() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API_BASE, timeout=120)


async def ask_question(
    question: str,
    history: list[list[str | None]],
    top_k: int,
    llm: str,
    show_context: bool,
) -> tuple[list[list[str | None]], str | None]:
    if not question.strip():
        return history, None

    history.append([question, None])

    try:
        async with _build_curl() as client:
            resp = await client.post(
                "/api/ask",
                json={"question": question, "top_k": top_k, "llm": llm or None},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        history[-1][1] = f"Error: {detail}"
        return history, ""
    except httpx.RequestError as exc:
        history[-1][1] = f"Cannot reach API at {API_BASE}: {exc}"
        return history, ""

    answer = data["answer"]
    lines = [answer]

    if show_context:
        lines.append("\n--- Retrieved Context ---")
        for r in data.get("results", []):
            lines.append(f"\n[score={r['score']}] {r['source']}")
            lines.append(r["text"][:300])

    history[-1][1] = "\n".join(lines)
    return history, ""


async def rebuild_index(
    source_dir: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    try:
        async with _build_curl() as client:
            resp = await client.post(
                "/api/ingest",
                json={
                    "source_dir": source_dir or None,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        return f"Error: {detail}"
    except httpx.RequestError as exc:
        return f"Cannot reach API at {API_BASE}: {exc}"

    return (
        f"Index built successfully.\n"
        f"Documents: {data['document_count']}\n"
        f"Chunks: {data['chunk_count']}\n"
        f"Duration: {data['duration_seconds']}s\n"
        f"Index path: {data['index_path']}"
    )


async def load_status() -> dict:
    try:
        async with _build_curl() as client:
            resp = await client.get("/api/status")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"error": f"Cannot reach API at {API_BASE}"}


async def refresh_status() -> tuple[gr.DataFrame, gr.Textbox]:
    status = await load_status()
    if "error" in status:
        return gr.DataFrame(value=None), gr.Textbox(value=status["error"])

    rows = [
        ["Index exists", str(status["index_exists"])],
        ["Index path", status["index_path"]],
        ["Source directory", status["source_dir"]],
        ["Document count", str(status.get("document_count", "N/A"))],
        ["Chunk count", str(status.get("chunk_count", "N/A"))],
        ["Last modified", str(status.get("last_modified", "N/A"))],
    ]
    df = gr.DataFrame(
        value=rows,
        headers=["Key", "Value"],
        datatype=["str", "str"],
        column_count=2,
    )
    files = "\n".join(f"• {f}" for f in status.get("source_files", []))
    files_box = gr.Textbox(value=files or "(no source files found)")
    return df, files_box


with gr.Blocks(
    title="RAG Pipeline",
    fill_width=True,
) as demo:
    gr.Markdown("# RAG Pipeline")
    gr.Markdown(
        f"API backend: `{API_BASE}` — "
        "[Swagger docs]({}/docs)".format(API_BASE)
    )

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                render_markdown=True,
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask something about the indexed documents...",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", scale=1, variant="primary")
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=4, step=1,
                    label="Top-K retrieved chunks",
                )
                llm_dropdown = gr.Dropdown(
                    choices=["extractive", "ollama"],
                    value="extractive",
                    label="Generator",
                )
                show_context_cb = gr.Checkbox(
                    value=False,
                    label="Show retrieved context",
                )
            clear_btn = gr.ClearButton([chatbot, question_input], value="Clear")

            ask_event = question_input.submit(
                fn=ask_question,
                inputs=[question_input, chatbot, top_k_slider, llm_dropdown, show_context_cb],
                outputs=[chatbot, question_input],
            )
            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, chatbot, top_k_slider, llm_dropdown, show_context_cb],
                outputs=[chatbot, question_input],
            )

        with gr.Tab("Index Management"):
            with gr.Group():
                source_dir_input = gr.Textbox(
                    label="Source directory",
                    value="data/source_docs",
                    info="Path to folder with .md, .txt, .pdf files",
                )
                with gr.Row():
                    chunk_size_slider = gr.Slider(
                        minimum=50, maximum=500, value=180, step=10,
                        label="Chunk size (words)",
                    )
                    chunk_overlap_slider = gr.Slider(
                        minimum=0, maximum=100, value=40, step=5,
                        label="Chunk overlap (words)",
                    )
                rebuild_btn = gr.Button("Rebuild Index", variant="primary")
                rebuild_output = gr.Textbox(
                    label="Build result",
                    lines=5,
                    interactive=False,
                )

            rebuild_btn.click(
                fn=rebuild_index,
                inputs=[source_dir_input, chunk_size_slider, chunk_overlap_slider],
                outputs=[rebuild_output],
            )

        with gr.Tab("Status"):
            status_df = gr.DataFrame(
                label="Pipeline Configuration",
                headers=["Key", "Value"],
                datatype=["str", "str"],
                column_count=2,
            )
            source_files_box = gr.Textbox(
                label="Available source files",
                lines=8,
                interactive=False,
            )
            refresh_btn = gr.Button("Refresh Status", variant="secondary")
            refresh_btn.click(
                fn=refresh_status,
                inputs=[],
                outputs=[status_df, source_files_box],
            )

    demo.load(
        fn=refresh_status,
        inputs=[],
        outputs=[status_df, source_files_box],
    )
