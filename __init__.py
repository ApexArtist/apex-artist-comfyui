from apex_resize import ApexSmartResize
from apex_rgb_curve import ApexRGBCurve
from apex_color_reference import ApexColorReference
from apex_color_science import ColorScience, FilmGrain
from apex_depth_to_normal import ApexDepthToNormal
from apex_film_profiles import FilmStockProfiles

NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexRGBCurve": ApexRGBCurve,
    "ApexColorReference": ApexColorReference,
    "ColorScience": ColorScience,
    "FilmGrain": FilmGrain,
    "ApexDepthToNormal": ApexDepthToNormal,
    "FilmStockProfiles": FilmStockProfiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize",
    "ApexRGBCurve": "Apex RGB Curve", 
    "ApexColorReference": "Apex Color Reference",
    "ColorScience": "Apex Color Science",
    "FilmGrain": "Apex Film Grain",
    "ApexDepthToNormal": "Apex Depth to Normal",
    "FilmStockProfiles": "Apex Film Stock Profiles",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']