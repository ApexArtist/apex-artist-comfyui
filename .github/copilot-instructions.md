# AI Agent Instructions for ComfyUI Apex Artist Nodes

## Project Overview
This project provides professional image processing nodes for ComfyUI, focusing on high-quality image manipulation with advanced algorithms. The codebase consists of specialized nodes for:
- Smart image resizing with AI-compatible resolutions (`apex_smart_resize.py`)
- Depth to normal map conversion (`apex_depth_to_normal.py`)
- Color reference and matching tools (`apex_color_reference.py`)
- Layer blending with Photoshop-style blend modes (`apex_layer_blend.py`)

## Core Architecture
- Each node is implemented as a standalone Python class
- Nodes follow ComfyUI's node structure pattern:
  ```python
  class ApexNode:
      @classmethod
      def INPUT_TYPES(cls):  # Defines node inputs
      RETURN_TYPES = ("IMAGE",)  # Defines output types
      FUNCTION = "process"  # Main processing function
      CATEGORY = "ApexArtist"  # Node category in UI
  ```

## Key Conventions
1. **Image Processing**:
   - Images are processed as PyTorch tensors (NHWC format)
   - Resolution constraints follow AI model requirements (see `ApexSmartResize.resolution_sets`)

2. **Input Validation**:
   - Parameters use ComfyUI's input type system with ranges
   - Example from `apex_rgb_curve.py`:
     ```python
     "master_shadows": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1})
     ```
   - JSON string inputs for complex data (curve points, color profiles)

3. **Configuration Constants**:
   - Presets and constants are defined as class attributes
   - Examples: `curve_presets` in `ApexRGBCurve`, `resolution_sets` in `ApexSmartResize`
   - Film profiles stored as dictionaries with RGB transformations

4. **Node Registration**:
   - All nodes registered in `__init__.py` with `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
   - Four total nodes: ApexSmartResize, ApexDepthToNormal, ApexColorReference, ApexLayerBlend

## Dependencies
- Core dependencies: torch, numpy, PIL, scipy (provided by ComfyUI)
- Additional: scikit-image>=0.19.0 for advanced image processing
- Version requirements in `requirements.txt`
- All nodes require ComfyUI environment

## Development Guidelines
1. **Adding New Nodes**:
   - Follow existing node class structure
   - Include INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY
   - Document parameters with ranges and defaults

2. **Image Processing Best Practices**:
   - Handle tensor dimensions consistently (NHWC format)
   - Use torch.nn.functional for optimized operations
   - Batch processing support for multiple images
   - Include error handling for invalid inputs

3. **Performance Considerations**:
   - Implement GPU acceleration where possible
   - Use vectorized operations with torch/numpy
   - Cache computed values for reuse

## Testing
- Manual testing through ComfyUI interface
- Key test cases:
  - Resolution snapping functionality (ApexSmartResize)
  - Depth map conversions (ApexDepthToNormal)
  - Color reference matching (ApexColorReference)
  - Layer blending modes (ApexLayerBlend)