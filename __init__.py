"""
ComfyUI Apex Artist Nodes
Professional image processing nodes for ComfyUI designed for AI artists and creators.
"""

__version__ = "1.2.0"
__author__ = "Apex Artist"
__description__ = "Professional image processing nodes for ComfyUI"

# Import all nodes
from .apex_resize import ApexSmartResize
from .apex_rgb_curve import ApexRGBCurve
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_color_reference import ApexColorReference
from .apex_color_science import ApexColorScience
from .apex_film_profiles import ApexFilmProfiles
from .apex_widgets import ApexWidgets

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexRGBCurve": ApexRGBCurve,
    "ApexDepthToNormal": ApexDepthToNormal,
    "ApexColorReference": ApexColorReference,
    "ApexColorScience": ApexColorScience,
    "ApexFilmProfiles": ApexFilmProfiles,
    "ApexWidgets": ApexWidgets,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "🎯 Apex Smart Resize",
    "ApexRGBCurve": "🎨 Apex RGB Curve",
    "ApexDepthToNormal": "🔄 Apex Depth to Normal",
    "ApexColorReference": "🎨 Apex Color Reference",
    "ApexColorScience": "🔬 Apex Color Science",
    "ApexFilmProfiles": "🎞️ Apex Film Profiles",
    "ApexWidgets": "🔧 Apex Widgets",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

print(f"[Apex Artist] Loading v{__version__} - {len(NODE_CLASS_MAPPINGS)} nodes available")