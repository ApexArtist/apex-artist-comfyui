import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ImageResize:
    """
    Apex Image Resize node - Enhanced version 
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 16384, 
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 1, 
                    "max": 16384, 
                    "step": 1
                }),
                "interpolation": ([
                    "nearest",
                    "bilinear", 
                    "bicubic",
                    "area",
                    "nearest-exact",
                    "lanczos"
                ], {"default": "nearest"}),
                "method": ([
                    "stretch",
                    "keep proportion",
                    "fill / crop",
                    "pad"
                ], {"default": "stretch"}),
                "condition": ([
                    "always",
                    "if bigger",
                    "if smaller"
                ], {"default": "always"}),
                "multiple_of": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "ApexArtist"

    def resize(self, image, width, height, interpolation, method, condition, multiple_of):
        # Convert image tensor to proper format
        if len(image.shape) == 4:
            batch_size, h, w, c = image.shape
        else:
            # Add batch dimension if missing
            image = image.unsqueeze(0)
            batch_size, h, w, c = image.shape
        
        original_width, original_height = w, h
        target_width, target_height = width, height
        
        # Apply multiple_of constraint
        if multiple_of > 0:
            target_width = (target_width // multiple_of) * multiple_of
            target_height = (target_height // multiple_of) * multiple_of
        
        # Check condition
        if condition == "if bigger" and (original_width <= target_width and original_height <= target_height):
            return (image,)
        elif condition == "if smaller" and (original_width >= target_width and original_height >= target_height):
            return (image,)
        
        # Handle different methods
        if method == "keep proportion":
            # Calculate aspect ratios
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if original_ratio > target_ratio:
                # Width is the limiting factor
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                # Height is the limiting factor
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            target_width, target_height = new_width, new_height
            
        elif method == "fill / crop":
            # Resize to fill the target size, then crop
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if original_ratio > target_ratio:
                # Scale based on height, then crop width
                scale_factor = target_height / original_height
                new_width = int(original_width * scale_factor)
                new_height = target_height
            else:
                # Scale based on width, then crop height
                scale_factor = target_width / original_width
                new_width = target_width
                new_height = int(original_height * scale_factor)
            
            # First resize to the calculated dimensions
            resized_image = self._resize_tensor(image, new_width, new_height, interpolation)
            
            # Then crop to target size
            if new_width > target_width:
                # Crop width
                start_x = (new_width - target_width) // 2
                resized_image = resized_image[:, :, start_x:start_x + target_width, :]
            elif new_height > target_height:
                # Crop height
                start_y = (new_height - target_height) // 2
                resized_image = resized_image[:, start_y:start_y + target_height, :, :]
            
            return (resized_image,)
            
        elif method == "pad":
            # Resize keeping aspect ratio, then pad
            original_ratio = original_width / original_height
            target_ratio = target_width / target_height
            
            if original_ratio > target_ratio:
                # Width is the limiting factor
                new_width = target_width
                new_height = int(target_width / original_ratio)
            else:
                # Height is the limiting factor
                new_height = target_height
                new_width = int(target_height * original_ratio)
            
            # Resize to calculated dimensions
            resized_image = self._resize_tensor(image, new_width, new_height, interpolation)
            
            # Pad to target size
            pad_x = (target_width - new_width) // 2
            pad_y = (target_height - new_height) // 2
            pad_x_right = target_width - new_width - pad_x
            pad_y_bottom = target_height - new_height - pad_y
            
            # Padding format: (pad_left, pad_right, pad_top, pad_bottom)
            padded_image = F.pad(resized_image.permute(0, 3, 1, 2), 
                               (pad_x, pad_x_right, pad_y, pad_y_bottom), 
                               mode='constant', value=0)
            resized_image = padded_image.permute(0, 2, 3, 1)
            
            return (resized_image,)
        
        # Default: stretch method or direct resize
        resized_image = self._resize_tensor(image, target_width, target_height, interpolation)
        
        return (resized_image,)
    
    def _resize_tensor(self, image, width, height, interpolation):
        """
        Resize image tensor using specified interpolation method
        """
        # Convert to format expected by F.interpolate: (batch, channels, height, width)
        image_transposed = image.permute(0, 3, 1, 2)
        
        # Map interpolation methods
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "area": "area",
            "nearest-exact": "nearest-exact",
            "lanczos": "bicubic"  # PyTorch doesn't have lanczos, use bicubic as fallback
        }
        
        mode = mode_map.get(interpolation, "nearest")
        
        # Special handling for lanczos using PIL
        if interpolation == "lanczos":
            return self._resize_with_pil(image, width, height, Image.LANCZOS)
        
        # Use torch interpolation
        if mode == "nearest-exact":
            # Handle nearest-exact specially if available in your PyTorch version
            try:
                resized = F.interpolate(image_transposed, size=(height, width), 
                                      mode="nearest-exact", antialias=False)
            except:
                # Fallback to regular nearest
                resized = F.interpolate(image_transposed, size=(height, width), 
                                      mode="nearest", antialias=False)
        else:
            # For bicubic and bilinear, use antialias for better quality
            antialias = mode in ["bilinear", "bicubic"]
            resized = F.interpolate(image_transposed, size=(height, width), 
                                  mode=mode, antialias=antialias)
        
        # Convert back to (batch, height, width, channels)
        return resized.permute(0, 2, 3, 1)
    
    def _resize_with_pil(self, image, width, height, pil_method):
        """
        Resize using PIL for methods not available in PyTorch
        """
        batch_size = image.shape[0]
        resized_batch = []
        
        for i in range(batch_size):
            # Convert tensor to PIL Image
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # Resize using PIL
            resized_pil = pil_img.resize((width, height), pil_method)
            
            # Convert back to tensor
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_np).to(image.device)
            
            resized_batch.append(resized_tensor)
        
        return torch.stack(resized_batch, dim=0)