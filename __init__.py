"""
ComfyUI Apex Artist Nodes
Easy Convenient tools for AI artists and creators.
"""

from .apex_resize import ApexSmartResize
from .apex_depth_to_normal import ApexDepthToNormal


# Node Class Mappings
NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize,
    "ApexDepthToNormal": ApexDepthToNormal,
    
}

# Display Name Mappings  
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "🎯 Apex Smart Resize",
    "ApexDepthToNormal": "🗺️ Apex Depth to Normal",
   
}

# Package metadata
__version__ = "1.1.0"
__author__ = "ApexArtist"
__description__ = "Professional image processing nodes for ComfyUI"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"🎨 Apex Artist Nodes v{__version__} loaded successfully!")
print("   📦 Available nodes:")
print("      🎯 Apex Smart Resize - Intelligent image resizing")
print("      🗺️ Apex Depth to Normal - Depth map conversion") 



