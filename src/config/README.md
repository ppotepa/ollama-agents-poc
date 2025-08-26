# Configuration Authority

## Single Source of Truth

The authoritative configuration for models is: **`src/config/models.yaml`**

All other model files have been moved to `docs/legacy/` and should be considered:
- Historical artifacts
- Generated outputs 
- Deprecated alternatives

## Guidelines

1. **Add new models** → Update `src/config/models.yaml`
2. **Change model capabilities** → Update `src/config/models.yaml` 
3. **Generate model lists** → Create scripts that read from `models.yaml`

## Legacy Files (Deprecated)

- `models.txt` - Plain text list
- `models_cache.json` - Cached model data
- `models_json_example.txt` - Example JSON format
- `models-full-list.json` - Full model registry
- `models-ollama-library.json` - Ollama library models

These files should NOT be edited directly. They are kept for reference only.
