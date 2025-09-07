from .ApexArtist import ApexArtist
from .apex_depth_to_normal import ApexDepthToNormal

NODE_CLASS_MAPPINGS = {
    "ApexArtist": ApexArtist,
    "ApexArtist - Depth to Normal": ApexDepthToNormal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexArtist": "Apex Artist - Image Resize",
    "ApexArtist - Depth to Normal": "Apex Artist - Depth to Normal",
}
