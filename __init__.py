"""
Apex Artist Nodes for ComfyUI - Professional image processing nodes
"""
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_resize import ApexSmartResize
from .apex_rgb_curve import ApexRGBCurve

# Curve preset definitions
CURVE_PRESETS = {
    # Basic Adjustments
    "linear": [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1, 1)],
    "brighten": [(0, 0), (0.25, 0.35), (0.5, 0.65), (0.75, 0.85), (1, 1)],
    "darken": [(0, 0), (0.25, 0.15), (0.5, 0.35), (0.75, 0.65), (1, 1)],
    
    # Contrast Curves
    "contrast": [(0, 0), (0.25, 0.15), (0.5, 0.5), (0.75, 0.85), (1, 1)],
    "strong_contrast": [(0, 0), (0.25, 0.1), (0.5, 0.5), (0.75, 0.9), (1, 1)],
    "low_contrast": [(0, 0.1), (0.25, 0.3), (0.5, 0.5), (0.75, 0.7), (1, 0.9)],
    
    # S-Curves
    "slight_s": [(0, 0), (0.2, 0.15), (0.5, 0.5), (0.8, 0.85), (1, 1)],
    "medium_s": [(0, 0), (0.2, 0.1), (0.5, 0.5), (0.8, 0.9), (1, 1)],
    "strong_s": [(0, 0), (0.2, 0.05), (0.5, 0.5), (0.8, 0.95), (1, 1)],
    
    # Creative Looks
    "film_look": [(0.06, 0), (0.25, 0.2), (0.5, 0.55), (0.75, 0.8), (0.94, 1)],
    "cinematic": [(0.05, 0), (0.3, 0.25), (0.5, 0.55), (0.7, 0.75), (0.95, 0.95)],
    "vintage": [(0.08, 0.1), (0.3, 0.3), (0.5, 0.47), (0.7, 0.7), (0.92, 0.9)],
    "bleach_bypass": [(0.05, 0.05), (0.25, 0.3), (0.5, 0.6), (0.75, 0.8), (0.95, 0.95)],
    "technicolor": [(0, 0), (0.2, 0.15), (0.5, 0.6), (0.8, 0.85), (1, 0.95)]
}

# Blend mode definitions
BLEND_MODES = [
    "normal",
    "multiply", 
    "screen",
    "overlay",
    "soft_light",
    "hard_light",
    "color_dodge",
    "color_burn",
    "darken",
    "lighten"
]

# Default curve settings
DEFAULT_CURVE_POINTS = [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1, 1)]

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexRGBCurve": ApexRGBCurve,
    "ApexDepthToNormal": ApexDepthToNormal,
}

# Display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Artist - Smart Resize",
    "ApexRGBCurve": "Apex Artist - RGB Curves",
    "ApexDepthToNormal": "Apex Artist - Depth to Normal",
}
