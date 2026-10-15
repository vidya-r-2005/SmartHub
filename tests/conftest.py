"""
Shared pytest fixtures for SmartHub tests.

Each test module gets a fresh temporary SQLite database so that:
  - Tests are isolated from each other.
  - The production database (data/smarthub.db) is never touched.
  - Test order does not matter.
"""

import tempfile
from pathlib import Path

import pytest

from app.database import db as database
from app.main import create_app
from app.config import TestingConfig


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Create an initialised temporary database and return its path."""
    db_file = tmp_path / "test_smarthub.db"
    database.initialize_database(db_path=db_file)
    return db_file


@pytest.fixture()
def app(tmp_db: Path):
    """Return a Flask test application backed by a temp database."""
    TestingConfig.DATABASE_PATH = tmp_db
    flask_app = create_app(TestingConfig)
    flask_app.config["DB_PATH"] = tmp_db
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    """Return a Flask test client."""
    return app.test_client()

