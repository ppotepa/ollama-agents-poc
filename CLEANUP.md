## Repository Cleanup (2025-08-20)

To simplify the project and remove redundant platform-specific launcher scripts, a consolidation pass was performed.

### Removed Scripts

Windows BAT / PowerShell:
- run_deepcoder.bat
- run_deepcoder_model.bat
- run_interactive.bat
- start_agent_server.bat
- start_simple_server.bat (empty placeholder)
- create_file_simulator.bat (empty placeholder)
- run_deepcoder.ps1
- start_agent_server.ps1

Shell (Linux / WSL / macOS):
- run_deepcoder.sh
- run_model.sh
- start_agent_server.sh
- stop_agent_server.sh
- restart.sh

### Rationale
1. Single cross‑platform entrypoint (main.py) reduces drift.
2. Scripts duplicated logic (venv activation, server start, model selection) now handled by CLI flags.
3. Encourages consistent invocation & simplifies documentation/testing.

### Unified Usage
List agents:
```bash
python main.py --list-agents
```
Interactive selection:
```bash
python main.py
```
Direct agent selection (old --model still works but deprecated):
```bash
python main.py --agent deepcoder
```
Run API server (replaces all start_* / restart / stop scripts):
```bash
python main.py --server --host 0.0.0.0 --port 8000
```

### Virtual Environment (Optional)
```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows
source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

### Docker / Ollama Model Execution
If you previously relied on run_model.sh to run a raw Ollama model inside a container:
```bash
docker exec -it ollama ollama run deepcoder:14b "Hello"
```

### Next Steps
- Extract remaining tools into src/tools/*
- Add tests for agent initialization
- Optional: Provide Makefile for common tasks

### Migration Notes
--model flag prints a deprecation warning; prefer --agent.

Cleanup performed on 2025-08-20.
