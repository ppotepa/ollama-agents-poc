#!/usr/bin/env python3
"""Continue Agent Shim Server - Modular facade for OpenAI-compatible API."""

# Backwards compatibility facade that delegates to specialized server components
from .server import create_app, get_server_config, APIRouteHandler

# Create the FastAPI app
app = create_app()

# Initialize route handler
route_handler = APIRouteHandler(app)

# For direct execution
if __name__ == "__main__":
    import uvicorn
    
    config = get_server_config()
    
    uvicorn.run(
        "src.core.server:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"],
        log_level=config["log_level"],
        workers=config["workers"]
    )
