# AI Agent Instructions for ComfyUI Apex Artist Nodes

## Project Overview
Professional VFX and post-production nodes for ComfyUI, focused on advanced image manipulation and film-grade effects. Nodes are designed for modularity, performance, and compatibility with AI workflows.

## Integration & Dependencies
- Core: torch, numpy, PIL, scipy (provided by ComfyUI)
- Extra: scikit-image>=0.19.0 (see `requirements.txt`)
- All nodes require ComfyUI runtime; not standalone scripts.
- StableNormal integration uses torch.hub and local cache (see `apex_stable_normal.py`).

## Key Files & Directories
- `__init__.py`: Node registration
- `apex_*.py`: Node implementations
- `update_version.py`: Version update script
- `.github/workflows/publish_action.yml`: GitHub Actions for registry publishing
- `README.md`: Node features and install instructions

## Copilot Guidance for Node Generation
# Copilot Instruction:
# Generate a ComfyUI custom node class.
# Follow ComfyUI conventions strictly:
# - INPUT_TYPES is a @classmethod returning dict with 'required' inputs
# - RETURN_TYPES is a tuple with one or more return types (e.g., "IMAGE", "AUDIO", "MASK")
# - FUNCTION is the method name to call
# - CATEGORY is a string for the node category
# - Do not add unrelated imports or functions
# - Use PyTorch tensor operations only
# - Always follow ComfyUI discipline; do not hallucinate Python code
# - Avoid blocking operations that prevent the node from returning
# - Use async or fast processing for heavy operations (e.g., TTS, audio)
# - Keep class and method names consistent with FUNCTION and file name
# - Avoid CPU-intensive loops in the main thread
