
from .apex_smart_resize import ApexSmartResize
from .apex_depth_to_normal import ApexDepthToNormal
from .apex_layer_blend import ApexLayerBlend
from .apex_blur import ApexBlur
from .apex_sharpen import ApexSharpen
from .apex_rgb_curve import ApexRGBCurve
from .apex_upscale import ApexUpscaleBy
NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexDepthToNormal": ApexDepthToNormal,
    "ApexLayerBlend": ApexLayerBlend,
    "ApexBlur": ApexBlur,
    "ApexSharpen": ApexSharpen,
    "ApexRGBCurve": ApexRGBCurve,
    "ApexUpscaleBy": ApexUpscaleBy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize",
    "ApexDepthToNormal": "Apex Depth to Normal",
    "ApexLayerBlend": "Apex Layer Blend",
    "ApexBlur": "Apex Blur",
    "ApexSharpen": "Apex Sharpen",
    "ApexRGBCurve": "Apex RGB Curve",
    "ApexUpscaleBy": "Apex Upscale By Ratio",
}

WEB_DIRECTORY = "./web"
