"""Jobs Dagster du template — point d'entrée : `make dagster-dev`."""

from pipelines.training_pipeline import defs

__all__ = ["defs"]
