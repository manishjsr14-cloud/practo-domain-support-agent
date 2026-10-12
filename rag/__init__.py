"""Offline RAG package: set privacy controls before importing dependencies."""

import os


# The graded runtime must never fetch models or emit framework telemetry.
RUNTIME_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
}
for _name, _value in RUNTIME_ENVIRONMENT.items():
    os.environ[_name] = _value
