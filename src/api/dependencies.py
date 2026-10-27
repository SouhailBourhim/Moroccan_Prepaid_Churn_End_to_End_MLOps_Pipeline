"""Model artifact loading for the churn prediction API.

The loader lives in src.models.artifacts so the API and the batch predict CLI
serve the same artifact; it is re-exported here for the API layer.
"""
from __future__ import annotations

from src.models.artifacts import ModelArtifacts, load_artifacts

__all__ = ["ModelArtifacts", "load_artifacts"]
