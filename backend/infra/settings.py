#!/usr/bin/env python3
"""
🔧 SETTINGS CONFIGURATION
Pydantic Settings for FastAPI application configuration
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow",  # Allow extra environment variables
    }

    # Application
    app_name: str = Field(default="Intraday Trading Platform", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8002, env="PORT")
    workers: int = Field(default=1, env="WORKERS")

    # Security
    secret_key: str = Field(env="SECRET_KEY", default="your-secret-key-here")
    allowed_hosts: list[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:3003"], env="CORS_ORIGINS"
    )

    # Database (if needed)
    database_url: str | None = Field(default=None, env="DATABASE_URL")

    # Alpaca Trading
    alpaca_api_key: str = Field(default="test_key", env="APCA_API_KEY_ID")
    alpaca_secret_key: str = Field(default="test_secret", env="APCA_API_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", env="APCA_API_BASE_URL"
    )

    # Risk Management
    max_portfolio_heat: float = Field(default=0.25, env="MAX_PORTFOLIO_HEAT")
    max_position_size: float = Field(default=0.10, env="MAX_POSITION_SIZE")
    daily_loss_limit: float = Field(default=0.05, env="DAILY_LOSS_LIMIT")
    max_drawdown_limit: float = Field(default=0.15, env="MAX_DRAWDOWN_LIMIT")

    # ML Model Configuration
    model_update_interval: int = Field(
        default=3600, env="MODEL_UPDATE_INTERVAL"
    )  # seconds
    prediction_confidence_threshold: float = Field(
        default=0.65, env="PREDICTION_CONFIDENCE_THRESHOLD"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text

    # External APIs
    news_api_key: str | None = Field(default=None, env="NEWS_API_KEY")
    alpha_vantage_api_key: str | None = Field(default=None, env="ALPHA_VANTAGE_API_KEY")

    # Trading Configuration
    enable_paper_trading: bool = Field(default=True, env="ENABLE_PAPER_TRADING")
    enable_risk_management: bool = Field(default=True, env="ENABLE_RISK_MANAGEMENT")
    enable_auto_trading: bool = Field(default=False, env="ENABLE_AUTO_TRADING")

    # Performance
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")  # seconds
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")


def get_settings() -> Settings:
    """Get application settings (cached)"""
    return Settings()


# Global settings instance
settings = get_settings()
