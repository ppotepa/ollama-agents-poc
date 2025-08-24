# Model-Agent Matching System

## Overview

The enhanced model-agent matching system provides intelligent discovery and comparison between available models from integrations (like Ollama) and agent implementations in the codebase.

## Command Line Interface

### `--list-all`
Shows **ALL** available models from all configured integrations.

```bash
python main.py --list-all
```

**Output Example:**
```
🔍 All Available Models (from integrations):
  📦 deepcoder:14b (8.4GB) [unknown] via ollama
  📦 llama3.3:70b-instruct-q3_K_M (31.9GB) [llama] via ollama
  📦 qwen2.5:7b-instruct-q4_K_M (4.4GB) [qwen] via ollama
  📦 gemma:7b-instruct-q4_K_M (5.1GB) [gemma] via ollama
  ...
📊 Total: 17 models available
```

### `--list-models`
Shows models with their **agent implementation status** and compatibility analysis.

```bash
python main.py --list-models
```

**Output Example:**
```
🤖 Models with Agent Implementation Status:

✅ Models WITH Agent Implementation (3):
  🎯 deepcoder:14b → deepcoder (8.4GB) - Exact name match: deepcoder
  🎯 deepseek-coder:6.7b → deepcoder (3.6GB) - Family match: deepseek-coder

❌ Models WITHOUT Agent Implementation (14):
  ⚠️  llama3.3:70b-instruct-q3_K_M (31.9GB) [llama] - No compatible agent found
  ⚠️  qwen2.5:7b-instruct-q4_K_M (4.4GB) [qwen] - No compatible agent found
  ...

📊 Summary: 3 ready, 14 need agents
```

## API Endpoints

### `/v1/models/all`
Returns all available models from integrations.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "deepcoder:14b",
      "object": "model",
      "owned_by": "ollama",
      "details": {
        "size": "8.4GB",
        "family": "unknown"
      },
      "source": "ollama"
    }
  ],
  "total": 17
}
```

### `/v1/models/with-agents`
Returns models with agent implementation analysis.

**Response:**
```json
{
  "with_agents": [
    {
      "model_id": "deepcoder:14b",
      "has_agent": true,
      "match_confidence": 1.0,
      "match_reason": "Exact name match: deepcoder",
      "agent_info": {
        "name": "deepcoder",
        "description": "DeepCoderAgent - concrete implementation using Ollama backend",
        "capabilities": ["code", "chat", "tools"],
        "family": "deepseek-coder"
      }
    }
  ],
  "without_agents": [
    {
      "model_id": "llama3.3:70b-instruct-q3_K_M",
      "has_agent": false,
      "match_confidence": 0.0,
      "match_reason": "No compatible agent found"
    }
  ],
  "summary": {
    "total_models": 17,
    "with_agents": 3,
    "without_agents": 14,
    "coverage_percentage": 17.6
  }
}
```

### `/v1/agents`
Returns all available agent implementations.

**Response:**
```json
{
  "agents": [
    {
      "name": "deepcoder",
      "description": "DeepCoderAgent - concrete implementation using Ollama backend",
      "family": "deepseek-coder",
      "capabilities": ["code", "chat", "tools"],
      "model_patterns": ["deepcoder:*", "deepseek-coder:*", "*coder*"],
      "module_path": "/app/src/agents/deepcoder/agent.py"
    }
  ],
  "total": 1
}
```

## Matching Algorithm

The system uses a sophisticated matching algorithm with confidence scoring:

### 1. **Exact Name Match** (Confidence: 1.0)
- Model: `deepcoder:14b`
- Agent: `deepcoder`
- ✅ Perfect match

### 2. **Family Match** (Confidence: 0.9)
- Model: `deepseek-coder:6.7b` (family: "deepseek-coder")
- Agent: `deepcoder` (family: "deepseek-coder")
- ✅ High confidence match

### 3. **Pattern Match** (Confidence: 0.4-0.8)
- Model: `any-coder-model`
- Agent Pattern: `*coder*`
- ✅ Pattern-based match

### 4. **Partial Name Match** (Confidence: 0.6)
- Model: `deepseek-v2:latest`
- Agent: `deepseek`
- ✅ Partial name overlap

### 5. **No Match** (Confidence: 0.0)
- Model: `llama3.3:70b`
- Available Agents: `deepcoder`
- ❌ No compatibility found

## Architecture Components

### `AgentRegistry`
- **Purpose**: Discovers and catalogs available agent implementations
- **Responsibilities**:
  - Scan the `src/agents/` directory
  - Extract agent metadata and capabilities  
  - Calculate model-agent compatibility scores
  - Provide agent lookup and matching services

### `IntegrationManager` (Enhanced)
- **Purpose**: Manages external service integrations
- **New Features**:
  - Smart host detection (localhost vs container)
  - Model source aggregation with metadata
  - Health monitoring across integrations

### Enhanced CLI
- **`--list-all`**: Pure integration view (what's available)
- **`--list-models`**: Development view (what's ready to use)

## Use Cases

### 1. **Development Planning**
Use `--list-models` to see which models need agent implementations:
```bash
python main.py --list-models
# Shows 14 models without agents - development opportunities!
```

### 2. **Integration Overview**  
Use `--list-all` to see all available models across services:
```bash
python main.py --list-all
# Shows 17 total models from Ollama integration
```

### 3. **API Integration**
Query `/v1/models/with-agents` to build model selection UIs:
```javascript
fetch('/v1/models/with-agents')
  .then(response => response.json())
  .then(data => {
    // Show only models with agents in UI
    const readyModels = data.with_agents;
  });
```

### 4. **Agent Development**
Use `/v1/agents` to understand existing agent patterns:
```bash
curl http://localhost:8000/v1/agents
# See what capabilities and patterns are already implemented
```

## Benefits

### **🎯 Clear Separation of Concerns**
- `--list-all`: Integration/infrastructure view
- `--list-models`: Application/development view

### **🚀 Development Guidance**  
- Instantly see which models need agent implementations
- Understand agent coverage percentage (currently 17.6%)
- Prioritize agent development based on model popularity

### **🔧 Smart Matching**
- Automatic compatibility detection
- Confidence scoring for match quality
- Multiple matching strategies (exact, family, pattern, partial)

### **📊 Comprehensive Reporting**
- Visual indicators (✅❌🎯⚠️) for quick scanning
- Detailed match reasoning
- Coverage statistics and summaries

### **🌐 API-First Design**
- All functionality available via REST API
- Suitable for building management UIs
- Programmatic access for automation

## Future Enhancements

1. **Agent Scaffolding**: Auto-generate agent templates for unmatched models
2. **Capability Analysis**: Match models to agents based on required capabilities  
3. **Performance Metrics**: Track agent success rates per model
4. **Configuration Management**: Allow custom model-agent mappings
5. **Multi-Source Integration**: Support for OpenAI, Anthropic, etc.
