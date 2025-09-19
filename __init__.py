from .apex_resize import ApexSmartResize  # Changed from ApexArtistImageResize
from .apex_depth_to_normal import ApexDepthToNormal

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,  # Changed class name
    "ApexDepthToNormal": ApexDepthToNormal
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize",  # Updated display name
    "ApexDepthToNormal": "Apex Depth To Normal"
}
