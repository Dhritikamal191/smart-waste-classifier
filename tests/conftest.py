from fastapi.testclient import TestClient

from src.api import app


# ============================================================
# FASTAPI TEST CLIENT
# ============================================================

import pytest


@pytest.fixture
def client():
    """
    Create a FastAPI test client.

    Using TestClient as a context manager ensures that
    FastAPI startup and shutdown events are executed.
    """

    with TestClient(app) as test_client:
        yield test_client