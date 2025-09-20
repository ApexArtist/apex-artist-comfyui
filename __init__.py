"""
ComfyUI Apex Artist Nodes
Professional image processing nodes for ComfyUI designed for AI artists and creators.
"""

# Version information
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

# Node mappings with version info in display names
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
    "ApexSmartResize": f"🎯 Apex Smart Resize v{__version__}",
    "ApexRGBCurve": f"🎨 Apex RGB Curve v{__version__}",
    "ApexDepthToNormal": f"🔄 Apex Depth to Normal v{__version__}",
    "ApexColorReference": f"🎨 Apex Color Reference v{__version__}",
    "ApexColorScience": f"🔬 Apex Color Science v{__version__}",
    "ApexFilmProfiles": f"🎞️ Apex Film Profiles v{__version__}",
    "ApexWidgets": f"🔧 Apex Widgets v{__version__}",
}

# Export version info
WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']

# Print version info when loading
print(f"[Apex Artist] Loading nodes v{__version__}")
print(f"[Apex Artist] {len(NODE_CLASS_MAPPINGS)} nodes available")