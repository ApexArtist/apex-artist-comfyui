from .apex_smart_resize import ApexSmartResize
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_color_reference import ApexColorReference

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexDepthToNormal": ApexDepthToNormal,
    "ApexColorReference": ApexColorReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize",
    "ApexDepthToNormal": "Apex Depth to Normal",
    "ApexColorReference": "Apex Color Reference",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']