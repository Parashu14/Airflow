from __future__ import annotations

import argparse
import threading

import uvicorn


def run_fastapi(host: str, port: int) -> None:
    uvicorn.run("rag_learning.api:app", host=host, port=port, log_level="info")


def run_gradio(host: str, port: int, api_url: str) -> None:
    import os

    os.environ["RAG_API_URL"] = api_url
    from rag_learning.gradio_app import demo

    demo.launch(server_name=host, server_port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the RAG pipeline web services.")
    parser.add_argument("--api-host", default="127.0.0.1", help="FastAPI host")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI port")
    parser.add_argument("--gradio-host", default="127.0.0.1", help="Gradio host")
    parser.add_argument("--gradio-port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--api-only", action="store_true", help="Start only the FastAPI server")
    parser.add_argument("--gradio-only", action="store_true", help="Start only the Gradio UI")
    args = parser.parse_args()

    api_url = f"http://{args.api_host}:{args.api_port}"

    if args.gradio_only:
        run_gradio(args.gradio_host, args.gradio_port, api_url)
        return

    if args.api_only:
        run_fastapi(args.api_host, args.api_port)
        return

    api_thread = threading.Thread(
        target=run_fastapi,
        args=(args.api_host, args.api_port),
        daemon=True,
    )
    api_thread.start()

    run_gradio(args.gradio_host, args.gradio_port, api_url)


if __name__ == "__main__":
    main()
