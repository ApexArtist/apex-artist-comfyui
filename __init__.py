from .apex_smart_resize import ApexSmartResize
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_color_reference import ApexColorReference
from .apex_layer_blend import ApexLayerBlend

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexDepthToNormal": ApexDepthToNormal,
    "ApexColorReference": ApexColorReference,
    "ApexLayerBlend": ApexLayerBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize",
    "ApexDepthToNormal": "Apex Depth to Normal",
    "ApexColorReference": "Apex Color Reference",
    "ApexLayerBlend": "Apex Layer Blend",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']