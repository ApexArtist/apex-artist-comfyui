# AI Agent Instructions for ComfyUI Apex Artist Nodes

## Project Overview
Professional VFX and post-production nodes for ComfyUI, focused on advanced image manipulation and film-grade effects. Nodes are designed for modularity, performance, and compatibility with AI workflows.

## Architecture & Node Patterns
- Each node is a standalone Python class, registered in `__init__.py` via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- Node interface:
  - `@classmethod INPUT_TYPES(cls)`: declares input types and ranges
  - `RETURN_TYPES`, `RETURN_NAMES`: output types and names
  - `FUNCTION`: main processing method
  - `CATEGORY`: UI grouping
- All image data is processed as PyTorch tensors (NHWC format).
- Key nodes: `ApexSmartResize`, `ApexDepthToNormal`, `ApexStableNormal`, `ApexColorReference`, `ApexLayerBlend`, `ApexBlur`, `ApexSharpen`.

## Conventions & Constants
- Resolution presets for AI models in `ApexSmartResize.resolution_sets`.
- Blend modes, color matching, and blur algorithms are implemented as explicit methods (see `apex_layer_blend.py`, `apex_color_reference.py`, `apex_blur.py`).
- Input validation uses ComfyUI's type/range system (see any node's `INPUT_TYPES`).
- Model cache for StableNormal is managed in `models/` (see `MODELS_CACHE.md`).

## Developer Workflows
- **Versioning:** Use `update_version.py <new_version>` to update all manifest/version files. Follow with `git add .; git commit; git push` to trigger registry publishing.
- **Node Addition:** Copy an existing node class, update `INPUT_TYPES`, `RETURN_TYPES`, and register in `__init__.py`.
- **Testing:** Manual via ComfyUI interface. Key tests: resolution snapping, depth/normal conversion, color matching, blending, blur/sharpen algorithms.
- **Model Management:** StableNormal models auto-download to `models/` on first use. To clear cache, delete the `models/` directory.

## Integration & Dependencies
- Core: torch, numpy, PIL, scipy (provided by ComfyUI)
- Extra: scikit-image>=0.19.0 (see `requirements.txt`)
- All nodes require ComfyUI runtime; not standalone scripts.
- StableNormal integration uses torch.hub and local cache (see `apex_stable_normal.py`).

## Project-Specific Patterns
- All tensor operations are batch-compatible and GPU-accelerated where possible.
- Error handling: nodes return original image and error info string on failure.
- Console/debug info is returned as JSON string in some nodes (see `ApexSmartResize`).
- Masking and blending use explicit tensor logic for compositing.

## Key Files & Directories
- `__init__.py`: Node registration
- `apex_*.py`: Node implementations
- `models/`: Model cache (StableNormal)
- `update_version.py`: Version update script
- `.github/workflows/publish_action.yml`: GitHub Actions for registry publishing
- `MODELS_CACHE.md`: Model cache details
- `README.md`: Node features and install instructions

---
For unclear conventions or missing details, ask the user for clarification or examples from their workflow.