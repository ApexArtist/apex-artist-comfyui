"""
Apex Artist - Image Resize Node
Advanced image resizing with multiple algorithms and aspect ratio handling
"""

import torch
import numpy as np
from PIL import Image

class ApexArtist:
    """
    Apex Artist - Professional image resizing with advanced options
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types and parameters for this node
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Input image tensor
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "resize_method": ([
                    "lanczos",
                    "bicubic", 
                    "bilinear",
                    "nearest",
                    "box",
                    "hamming"
                ], {
                    "default": "lanczos"
                }),
                "keep_proportion": ("BOOLEAN", {
                    "default": True
                }),
                "upscale_method": ([
                    "crop_center",
                    "crop_top",
                    "crop_bottom", 
                    "pad_center",
                    "pad_top",
                    "pad_bottom",
                    "stretch"
                ], {
                    "default": "crop_center"
                }),
            },
            "optional": {
                "pad_color": ([
                    "black",
                    "white", 
                    "gray",
                    "transparent"
                ], {
                    "default": "black"
                }),
                "sharpen_after_resize": ("BOOLEAN", {
                    "default": False
                }),
                "resize_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("resized_image", "final_width", "final_height", "resize_info")
    FUNCTION = "resize_image"
    CATEGORY = "Apex Artist"
    DESCRIPTION = "Professional image resizing with multiple algorithms and aspect ratio handling"

    def resize_image(self, image, width, height, resize_method="lanczos", keep_proportion=True, 
                    upscale_method="crop_center", pad_color="black", sharpen_after_resize=False, resize_factor=1.0):
        """
        Main resize function with advanced options
        """
        # Apply resize factor if not 1.0
        if resize_factor != 1.0:
            width = int(width * resize_factor)
            height = int(height * resize_factor)
        
        # Ensure dimensions are multiples of 8 for compatibility
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Convert tensor to PIL Image for processing
        batch_size = image.shape[0]
        resized_images = []
        
        # Map resize methods to PIL constants
        resize_methods = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR, 
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        
        # Map pad colors
        pad_colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "transparent": (0, 0, 0, 0)
        }
        
        for i in range(batch_size):
            # Convert tensor to PIL Image
            img_tensor = image[i]
            img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            original_width, original_height = pil_image.size
            
            if keep_proportion:
                resized_image = self._resize_with_aspect_ratio(
                    pil_image, width, height, resize_methods[resize_method],
                    upscale_method, pad_colors[pad_color]
                )
            else:
                # Direct resize without keeping proportion
                resized_image = pil_image.resize(
                    (width, height), 
                    resize_methods[resize_method]
                )
            
            # Apply post-processing
            if sharpen_after_resize:
                from PIL import ImageFilter
                resized_image = resized_image.filter(ImageFilter.SHARPEN)
            
            # Convert back to tensor
            resized_array = np.array(resized_image).astype(np.float32) / 255.0
            resized_images.append(torch.from_numpy(resized_array))
        
        # Stack processed images back into batch tensor
        result_tensor = torch.stack(resized_images)
        
        # Create resize info string
        resize_info = (f"Resized from {original_width}x{original_height} to {width}x{height} "
                      f"using {resize_method}, proportion={'kept' if keep_proportion else 'ignored'}, "
                      f"method={upscale_method}")
        
        return (result_tensor, width, height, resize_info)
    
    def _resize_with_aspect_ratio(self, image, target_width, target_height, resample_method, upscale_method, pad_color):
        """
        Resize image while maintaining aspect ratio with various handling methods
        """
        original_width, original_height = image.size
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if upscale_method == "stretch":
            # Simple stretch without maintaining aspect ratio
            return image.resize((target_width, target_height), resample_method)
        
        # Calculate dimensions for aspect ratio preservation
        if original_ratio > target_ratio:
            # Image is wider than target ratio
            if "crop" in upscale_method:
                # Crop width to fit height
                new_height = target_height
                new_width = int(new_height * original_ratio)
                resized = image.resize((new_width, new_height), resample_method)
                return self._crop_image(resized, target_width, target_height, upscale_method)
            else:
                # Pad height to fit width
                new_width = target_width
                new_height = int(new_width / original_ratio)
                resized = image.resize((new_width, new_height), resample_method)
                return self._pad_image(resized, target_width, target_height, upscale_method, pad_color)
        else:
            # Image is taller than target ratio
            if "crop" in upscale_method:
                # Crop height to fit width
                new_width = target_width
                new_height = int(new_width / original_ratio)
                resized = image.resize((new_width, new_height), resample_method)
                return self._crop_image(resized, target_width, target_height, upscale_method)
            else:
                # Pad width to fit height
                new_height = target_height
                new_width = int(new_height * original_ratio)
                resized = image.resize((new_width, new_height), resample_method)
                return self._pad_image(resized, target_width, target_height, upscale_method, pad_color)
    
    def _crop_image(self, image, target_width, target_height, crop_method):
        """
        Crop image to target dimensions with specified alignment
        """
        img_width, img_height = image.size
        
        if crop_method == "crop_center":
            left = (img_width - target_width) // 2
            top = (img_height - target_height) // 2
        elif crop_method == "crop_top":
            left = (img_width - target_width) // 2
            top = 0
        elif crop_method == "crop_bottom":
            left = (img_width - target_width) // 2
            top = img_height - target_height
        else:  # default to center
            left = (img_width - target_width) // 2
            top = (img_height - target_height) // 2
        
        right = left + target_width
        bottom = top + target_height
        
        # Ensure crop box is within image bounds
        left = max(0, min(left, img_width - target_width))
        top = max(0, min(top, img_height - target_height))
        right = left + target_width
        bottom = top + target_height
        
        return image.crop((left, top, right, bottom))
    
    def _pad_image(self, image, target_width, target_height, pad_method, pad_color):
        """
        Pad image to target dimensions with specified alignment
        """
        img_width, img_height = image.size
        
        # Create new image with target dimensions
        if len(pad_color) == 4:  # RGBA
            new_image = Image.new("RGBA", (target_width, target_height), pad_color)
        else:  # RGB
            new_image = Image.new("RGB", (target_width, target_height), pad_color)
        
        # Calculate paste position
        if pad_method == "pad_center":
            x = (target_width - img_width) // 2
            y = (target_height - img_height) // 2
        elif pad_method == "pad_top":
            x = (target_width - img_width) // 2
            y = 0
        elif pad_method == "pad_bottom":
            x = (target_width - img_width) // 2
            y = target_height - img_height
        else:  # default to center
            x = (target_width - img_width) // 2
            y = (target_height - img_height) // 2
        
        # Paste the resized image onto the new canvas
        if image.mode == "RGBA" or "transparency" in image.info:
            new_image.paste(image, (x, y), image)
        else:
            new_image.paste(image, (x, y))
        
        return new_image


# Node class mappings - this is required for ComfyUI to recognize your nodes
NODE_CLASS_MAPPINGS = {
    "ApexArtist": ApexArtist,
}

# Display names for the nodes in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexArtist": "Apex Artist - Image Resize",
}

# Optional: Version info
__version__ = "1.0.0"