"""FastAPI application setup and configuration."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

VERSION = "agent-shim-v2-2025-08-20"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(title="Continue Agent Shim", version=VERSION)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


def get_server_config() -> dict:
    """Get server configuration from environment variables.
    
    Returns:
        Dictionary with server configuration
    """
    return {
        "host": os.getenv("SERVER_HOST", "0.0.0.0"),
        "port": int(os.getenv("SERVER_PORT", "8000")),
        "reload": os.getenv("SERVER_RELOAD", "false").lower() == "true",
        "cors_origins": os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
        "workers": int(os.getenv("SERVER_WORKERS", "1"))
    }
