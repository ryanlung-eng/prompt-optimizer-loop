"""
Pulls the Databricks-native judges' scores (retrieval_groundedness, safety,
Valid Workflow, etc. — whatever's configured on the KA endpoint) for a given
trace, so they show up alongside our own custom judge + structural validator
instead of requiring a separate look in the MLflow Traces UI.

Best-effort: those judges run as production-monitoring scorers, which can
execute asynchronously. There's no documented guarantee they're attached the
instant a call returns, so a missing assessment here doesn't necessarily mean
a bad score — it may just not have been computed yet.
"""
import os

import mlflow

from .config import DatabricksConfig

_configured = False


def _ensure_mlflow_configured(config: DatabricksConfig) -> None:
    global _configured
    if _configured:
        return
    os.environ["DATABRICKS_HOST"] = config.workspace_url
    os.environ["DATABRICKS_TOKEN"] = config.token
    mlflow.set_tracking_uri("databricks")
    _configured = True


def fetch_assessments(config: DatabricksConfig, trace_id: str) -> dict:
    """Returns {assessment_name: value} for a trace, or {} if unavailable
    (network error, trace not found, or assessments not yet computed)."""
    if not trace_id:
        return {}
    _ensure_mlflow_configured(config)
    try:
        trace = mlflow.get_trace(trace_id)
        return {a.name: a.value for a in trace.info.assessments}
    except Exception as e:
        print(f"  Warning: could not fetch native assessments for trace {trace_id}: {e}")
        return {}
