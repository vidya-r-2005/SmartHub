"""
SmartHub application configuration.

Exposes three configuration classes:
  - DevelopmentConfig  (default when running locally)
  - TestingConfig      (used by automated tests — temp DB, no hardware)
  - ProductionConfig   (Raspberry Pi deployment)

Select via the SMARTHUB_ENV environment variable or by passing the class
directly to create_app().
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # SmartHub/

_DEFAULT_SECRET_KEY = "dev-secret-change-in-production"


class Config:
    """Shared base configuration."""

    # Flask
    SECRET_KEY: str = os.environ.get("SECRET_KEY", _DEFAULT_SECRET_KEY)
    DEBUG: bool = False
    TESTING: bool = False

    # Database
    DATABASE_PATH: Path = BASE_DIR / "data" / "smarthub.db"

    # Hardware
    SIMULATION_MODE: bool = True  # Always simulate unless overridden
    GPIO_AVAILABLE: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    # Upload safety
    MAX_CONTENT_LENGTH: int = 4 * 1024 * 1024  # 4 MB cap for image uploads

    # Stale-session recovery: sessions active longer than this many hours will
    # be marked 'failed' on startup.  Override via SESSION_TIMEOUT_HOURS env var.
    SESSION_TIMEOUT_HOURS: int = int(os.environ.get("SESSION_TIMEOUT_HOURS", "8"))


class DevelopmentConfig(Config):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    SIMULATION_MODE: bool = True


class TestingConfig(Config):
    """Isolated testing configuration — uses a temporary database."""

    TESTING: bool = True
    DEBUG: bool = True
    SIMULATION_MODE: bool = True

    # Tests override DATABASE_PATH via the db_path fixture; this is a safe
    # default that ensures the production database is never touched.
    DATABASE_PATH: Path = BASE_DIR / "data" / "test_smarthub.db"

    # Disable CSRF-like protections that would complicate test clients
    WTF_CSRF_ENABLED: bool = False


class ProductionConfig(Config):
    DEBUG: bool = False
    SIMULATION_MODE: bool = bool(os.environ.get("SIMULATION_MODE", ""))
    GPIO_AVAILABLE: bool = not SIMULATION_MODE
    LOG_LEVEL: str = "WARNING"


_CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(env: str | None = None) -> type[Config]:
    """Return the configuration class for the given environment string."""
    if env is None:
        env = os.environ.get("SMARTHUB_ENV", "development")
    cfg = _CONFIGS.get(env.lower())
    if cfg is None:
        raise ValueError(
            f"Unknown SMARTHUB_ENV '{env}'. "
            f"Valid values: {list(_CONFIGS.keys())}"
        )
    return cfg


def warn_if_insecure(cfg: type[Config]) -> None:
    """Log a warning if SECRET_KEY is still the insecure development default.

    Call this inside create_app() after loading config so that any deployment
    that forgets to set SECRET_KEY is clearly flagged in the application log.
    This is a no-op during automated tests (TESTING=True).
    """
    import logging

    _log = logging.getLogger(__name__)
    if (
        getattr(cfg, "SECRET_KEY", _DEFAULT_SECRET_KEY) == _DEFAULT_SECRET_KEY
        and not getattr(cfg, "TESTING", False)
    ):
        _log.warning(
            "SECURITY WARNING: SECRET_KEY is the default development value. "
            "Set the SECRET_KEY environment variable before deploying to production."
        )
