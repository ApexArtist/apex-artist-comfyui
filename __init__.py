from .apex_smart_resize import ApexSmartResize
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_stable_normal import ApexStableNormal
from .apex_color_reference import ApexColorReference
from .apex_layer_blend import ApexLayerBlend
from .apex_blur import ApexBlur
from .apex_sharpen import ApexSharpen

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexDepthToNormal": ApexDepthToNormal,
    "ApexStableNormal": ApexStableNormal,
    "ApexColorReference": ApexColorReference,
    "ApexLayerBlend": ApexLayerBlend,
    "ApexBlur": ApexBlur,
    "ApexSharpen": ApexSharpen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "🚀 Apex Smart Resize",
    "ApexDepthToNormal": "🎯 Apex Depth to Normal",
    "ApexStableNormal": "🌟 Apex Stable Normal",
    "ApexColorReference": "🎨 Apex Color Reference",
    "ApexLayerBlend": "✨ Apex Layer Blend",
    "ApexBlur": "🌀 Apex Blur",
    "ApexSharpen": "🔍 Apex Sharpen",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']