"""
MLflow PyFunc wrapper around WorkflowBuilderPipeline.

This is the thin shell that makes the pipeline deployable as a Databricks
Model Serving endpoint; all the actual behaviour lives in serving.py. Kept
separate so the pipeline stays importable and testable without MLflow, and so
this file can be read as exactly what it is — glue.

WHY THE CONFIG IS BAKED IN AS AN ARTIFACT rather than read at request time:
the endpoint must produce the same workflows the benchmark measured, and the
benchmark's behaviour is a function of the prompt text, retrieval settings and
model choices in config.yaml. Reading that from somewhere mutable at request
time would mean the deployed system could silently drift away from the thing
that was evaluated. Logging it as an artifact pins prompt + settings to a
model VERSION, so "which prompt produced this workflow" has an answer.

The Databricks token is deliberately NOT baked in — it is read from the
environment at load time so it can be rotated without re-logging a model, and
so a secret never lands in an artifact.
"""
import os
from typing import Any, Dict, List

import mlflow
import pandas as pd


class WorkflowBuilderModel(mlflow.pyfunc.PythonModel):
    """One request in, one built workflow (or a clarifying question) out."""

    def load_context(self, context):
        # Imported inside load_context, not at module import: MLflow imports
        # this module during logging, when the repo's own dependencies may not
        # be importable in the same way they will be at serving time.
        from prompt_optimizer.config import load_config
        from prompt_optimizer.serving import WorkflowBuilderPipeline

        config_path = context.artifacts["config"]
        # load_config resolves ${DATABRICKS_HOST}/${DATABRICKS_TOKEN} from the
        # environment. On a serving endpoint these come from the endpoint's
        # environment variables (set to secret references at deploy time), so
        # no credential is ever written into the logged model.
        os.environ.setdefault("DATABRICKS_HOST", os.environ.get("DATABRICKS_HOST", ""))
        self._pipeline = WorkflowBuilderPipeline(load_config(config_path))

    def _one(self, payload: Any) -> Dict[str, Any]:
        from prompt_optimizer.serving import (
            _context_from_payload, _conversation_from_payload,
        )

        try:
            conversation = _conversation_from_payload(payload)
        except ValueError as e:
            return {"status": "error", "message": str(e), "data": str(e),
                    "workflow": None, "valid": False, "errors": [], "warnings": [],
                    "repair_rounds": 0, "kept_sources": []}
        # Credentials/user/minutes are resolved by n8n per request rather than
        # baked into the prompt — see RequestContext. Absent ones degrade the
        # answer (no credentials wired) instead of failing the call.
        ctx = _context_from_payload(payload)

        # One endpoint, two jobs. "scope" is the conversational step that
        # decides whether a request is buildable; "build" produces the
        # workflow. They share the index and the retrieval path but not the
        # prompt. Default is build, so existing callers that send no mode keep
        # working unchanged.
        mode = ""
        if isinstance(payload, dict):
            mode = str(payload.get("mode") or "").strip().lower()

        if mode in ("scope", "scoping"):
            return self._pipeline.scope(conversation, ctx)
        if mode in ("", "build", "builder"):
            return self._pipeline.build(conversation, ctx)
        msg = f'Unknown mode "{mode}" — expected "scope" or "build".'
        return {"status": "error", "message": msg, "data": msg, "workflow": None,
                "valid": False, "errors": [], "warnings": [], "repair_rounds": 0,
                "kept_sources": []}

    def predict(self, context, model_input) -> List[Dict[str, Any]]:
        """Accepts a DataFrame (the split/records shapes Model Serving sends)
        or a plain dict/list, and always returns a list of result dicts — one
        per input row — so a caller sending one request gets a one-element
        list rather than a shape that changes with batch size."""
        if isinstance(model_input, pd.DataFrame):
            rows: List[Any] = model_input.to_dict(orient="records")
        elif isinstance(model_input, list):
            rows = model_input
        else:
            rows = [model_input]
        return [self._one(row) for row in rows]
