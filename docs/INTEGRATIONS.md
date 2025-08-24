# Integrations System Documentation

## Overview

The integrations system follows the **Single Responsibility Principle** by separating external service communication into dedicated, focused modules. Each integration handles communication with a specific external service.

## Architecture

```
src/integrations/
├── __init__.py                 # Package exports
├── base_integration.py         # Abstract base class
├── ollama_integration.py       # Ollama service integration
└── integration_manager.py      # Coordination and management
```

## Components

### 1. BaseIntegration (Abstract Interface)

**Purpose**: Defines the contract for all external service integrations.

**Key Methods**:
- `get_models()` - Fetch available models
- `is_available()` - Check service availability
- `get_version()` - Get service version
- `health_check()` - Perform health check

### 2. OllamaIntegration

**Purpose**: Handles all communication with Ollama service.

**Features**:
- Multiple discovery methods (HTTP API + CLI fallback)
- OpenAI-compatible model formatting
- Automatic error handling and fallbacks
- Human-readable size formatting
- Smart model family detection

**Configuration**:
```python
# Default (container network)
ollama = OllamaIntegration()

# Custom host
ollama = OllamaIntegration(host="http://localhost:11434")

# Custom timeout
ollama = OllamaIntegration(timeout=10)
```

### 3. IntegrationManager

**Purpose**: Coordinates multiple integrations and provides unified access.

**Features**:
- Dynamic integration registration
- Health monitoring across all services
- Aggregated model discovery
- Service availability tracking

**Usage**:
```python
from integrations import IntegrationManager

manager = IntegrationManager()

# Get models from all integrations
all_models = manager.get_all_models()

# Get models from specific integration
ollama_models = manager.get_models_from("ollama")

# Health check all integrations
health = manager.health_check()
```

## API Endpoints

The server now provides several integration-related endpoints:

### `/v1/models`
Lists all available models from Ollama integration.

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "deepcoder:14b",
      "object": "model",
      "created": 1755671816,
      "owned_by": "ollama",
      "supports_agent": true,
      "supports_tools": true,
      "capabilities": { ... },
      "details": {
        "size": "8.4GB",
        "family": "unknown",
        "parameter_size": "14.8B"
      }
    }
  ]
}
```

### `/v1/integrations`
Lists all registered integrations and their availability.

**Response**:
```json
{
  "integrations": ["ollama"],
  "available": ["ollama"]
}
```

### `/v1/integrations/health`
Provides health status for all integrations.

**Response**:
```json
{
  "ollama": {
    "available": true,
    "version": "0.11.5",
    "status": "healthy"
  }
}
```

## Benefits of This Architecture

### 1. Single Responsibility Principle
- Each class has one clear purpose
- OllamaIntegration only handles Ollama communication
- IntegrationManager only handles coordination
- BaseIntegration only defines contracts

### 2. Extensibility
Easy to add new integrations:

```python
class NewServiceIntegration(BaseIntegration):
    def get_models(self):
        # Implementation for new service
        pass
    
    def is_available(self):
        # Check new service availability
        pass

# Add to manager
manager.add_integration("newservice", NewServiceIntegration())
```

### 3. Testability
- Each component can be tested independently
- Mock integrations can be easily injected
- Clear interfaces make testing straightforward

### 4. Maintainability
- Changes to one service don't affect others
- Clear separation of concerns
- Easy to debug and troubleshoot

## Testing

Run the integration tests:

```bash
python test_integrations.py
```

This verifies:
- Individual integration functionality
- Integration manager coordination
- Service availability and health
- Model discovery and formatting

## Configuration

### Environment Variables

- `OLLAMA_HOST` - Ollama server URL (default: `http://ollama:11434` in containers)

### Docker Configuration

The system automatically adapts to different environments:
- **Container**: Uses `http://ollama:11434` for inter-container communication
- **Host**: Uses `http://localhost:11434` for local development

## Error Handling

The system provides robust error handling:

1. **Service Unavailable**: Falls back to default model
2. **Network Timeout**: Graceful degradation with error logging
3. **Invalid Response**: Format validation and recovery
4. **Multiple Methods**: HTTP API primary, CLI fallback

## Future Extensions

The architecture easily supports adding:

- **OpenAI Integration**: For hosted models
- **Anthropic Integration**: For Claude models
- **Local File Integration**: For static model definitions
- **Custom Endpoints**: For proprietary services

Each new integration just needs to implement the `BaseIntegration` interface and register with the manager.
