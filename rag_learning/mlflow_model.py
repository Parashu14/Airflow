from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_learning.pipeline import ask
from rag_learning.schema import RagAnswer
from rag_learning.vector_store import VectorStore

try:
    import mlflow
    from mlflow.pyfunc import PythonModel
except ImportError:
    PythonModel = object  # type: ignore[misc,assignment]


class RagPipelineModel(PythonModel):
    """MLflow pyfunc model for the RAG pipeline.

    Allows serving the pipeline via ``mlflow models serve``.
    """

    def __init__(self, index_path: str | Path, llm: str | None = None):
        self.index_path = str(index_path)
        self.llm = llm

    def load_context(self, context: Any) -> None:
        _ = context
        self._store = VectorStore.load(self.index_path)

    def predict(
        self,
        context: Any,
        model_input: list[list[str] | dict[str, list[str]]],
    ) -> list[dict[str, object]]:
        _ = context

        if isinstance(model_input, list):
            questions = model_input
        elif isinstance(model_input, dict):
            questions = list(model_input.values())[0]
        else:
            raise TypeError(f"Unsupported input type: {type(model_input)}")

        results: list[dict[str, object]] = []
        for question in questions:
            if not isinstance(question, str):
                continue
            answer: RagAnswer = ask(question, index_path=self.index_path, llm=self.llm)
            results.append({
                "question": question,
                "answer": answer.answer,
                "sources": answer.sources,
                "used_llm": answer.used_llm,
            })

        return results


def log_rag_pipeline(
    index_path: str | Path,
    llm: str | None = None,
    artifact_path: str = "rag_pipeline",
) -> None:
    """Log the RAG pipeline as an MLflow model to the current active run."""
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required to log the model") from exc

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=RagPipelineModel(index_path=index_path, llm=llm),
        artifacts={"index": str(index_path)},
    )
