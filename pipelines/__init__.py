"""Jobs Dagster du template — point d'entrée : `make dagster-dev`."""

from pipelines.monitoring_pipeline import defs as monitoring_defs
from pipelines.training_pipeline import defs

__all__ = ["defs", "monitoring_defs"]
