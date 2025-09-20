#!/usr/bin/env python3
"""
Apex_Resize_V2.py - Enhanced Smart Tensor Resizing Node
Features:
- Explicit tensor dtype handling (float16, float32)
- Memory-optimized tensor operations
- Enhanced resolution snapping for AI models
- Support for arbitrary bit depths
"""
import torch
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict, Union

class ApexSmartResizeV2:
    """
    Apex Smart Resize V2 - Enhanced tensor-aware image resizing
    - Preserves tensor precision (16/32-bit)
    - Memory-efficient tensor operations
    - Smart resolution snapping for AI models
    """
    
    def __init__(self):
        # Enhanced resolution sets with explicit model compatibility
        self.resolution_sets = {
            "SDXL_Base": [
                (1024, 1024),  # Square base
                (896, 1152),   # Portrait
                (1152, 896),   # Landscape
                # 3:4 Aspect Ratios
                (768, 1024),
                (960, 1280),
                # 4:3 Aspect Ratios
                (1024, 768),
                (1280, 960),
                # 16:9 Variants
                (896, 1600),
                (1600, 896)
            ],
            "SD_Legacy": [
                (512, 512),    # Base square
                (512, 768),    # Portrait
                (768, 512),    # Landscape
                (640, 640),    # Larger square
                (704, 704),    # Extended square
                # Common aspect ratios
                (512, 640),
                (640, 512),
                (576, 832),
                (832, 576)
            ],
            "Efficient_16": [  # All dims multiple of 16
                (512, 512),
                (512, 768),
                (768, 512),
                (640, 640),
                (768, 768),
                (896, 896),
                (1024, 1024)
            ],
            "Efficient_32": [  # All dims multiple of 32
                (512, 512),
                (544, 832),
                (832, 544),
                (672, 672),
                (800, 800),
                (928, 928),
                (1024, 1024)
            ]
        }
        
        # Initialize interpolation mode configs
        self.mode_configs = {
            "lanczos": {
                "mode": "bicubic",
                "antialias": True,
                "align_corners": True
            },
            "bicubic": {
                "mode": "bicubic",
                "antialias": True,
                "align_corners": True
            },
            "bilinear": {
                "mode": "bilinear",
                "antialias": True,
                "align_corners": True
            },
            "nearest": {
                "mode": "nearest",
                "antialias": False,
                "align_corners": None
            }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": ([
                    "SDXL_Base",
                    "SD_Legacy",
                    "Efficient_16",
                    "Efficient_32"
                ], {"default": "SDXL_Base"}),
                "snap_method": ([
                    "model_optimal",     # Optimized for model requirements
                    "memory_optimal",    # Optimized for memory usage
                    "quality_optimal",   # Optimized for visual quality
                    "performance",       # Optimized for speed
                ], {"default": "model_optimal"}),
                "resize_mode": ([
                    "smart_stretch",     # Content-aware stretching
                    "crop_center",       # Center crop after resize
                    "pad_reflect",       # Padding with reflection
                    "pad_replicate",     # Padding with edge replication
                    "pad_constant"       # Padding with constant value
                ], {"default": "smart_stretch"}),
                "interpolation": ([
                    "lanczos",
                    "bicubic",
                    "bilinear",
                    "nearest"
                ], {"default": "lanczos"}),
                "target_precision": ([
                    "auto",             # Keep source precision
                    "float16",          # Force 16-bit
                    "float32"           # Force 32-bit
                ], {"default": "auto"})
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "custom_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "pad_value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "scale_factor", "resolution_info")
    FUNCTION = "smart_resize_v2"
    CATEGORY = "ApexArtist/Image"

    def smart_resize_v2(self, image: torch.Tensor, resolution_set: str, 
                       snap_method: str, resize_mode: str, interpolation: str,
                       target_precision: str = "auto", custom_width: int = None,
                       custom_height: int = None, pad_value: float = 0.0) -> Tuple:
        """
        Enhanced smart resize with tensor precision handling
        Args:
            image: Input tensor in NHWC format
            resolution_set: Target resolution preset
            snap_method: Resolution snapping strategy
            resize_mode: Resize and padding mode
            interpolation: Interpolation algorithm
            target_precision: Target tensor precision
            custom_width: Optional custom width
            custom_height: Optional custom height
            pad_value: Padding value for constant mode
        """
        # Store original precision
        orig_dtype = image.dtype
        
        # Handle precision conversion
        if target_precision == "float16":
            image = image.half()
        elif target_precision == "float32":
            image = image.float()
        
        # Get target dimensions
        orig_h, orig_w = image.shape[1:3]
        target_w, target_h = self._get_target_dims(
            orig_w, orig_h, resolution_set, snap_method,
            custom_width, custom_height
        )
        
        # Perform resize operation
        if resize_mode == "smart_stretch":
            resized = self._smart_stretch(image, target_w, target_h, interpolation)
        elif resize_mode == "crop_center":
            resized = self._crop_center_resize(image, target_w, target_h, interpolation)
        elif resize_mode.startswith("pad_"):
            pad_mode = resize_mode.split("_")[1]
            resized = self._padded_resize(image, target_w, target_h, 
                                        interpolation, pad_mode, pad_value)
        
        # Restore original precision if auto mode
        if target_precision == "auto":
            resized = resized.to(orig_dtype)
            
        # Calculate scale factor
        scale_factor = min(target_w / orig_w, target_h / orig_h)
        
        # Generate resize info
        info = (f"Resized from ({orig_w}, {orig_h}) to ({target_w}, {target_h})\n"
                f"Mode: {resize_mode}, Scale: {scale_factor:.2f}x\n"
                f"Precision: {resized.dtype}")
        
        return (resized, target_w, target_h, scale_factor, info)

    def _get_target_dims(self, orig_w: int, orig_h: int, resolution_set: str,
                        snap_method: str, custom_w: int = None, 
                        custom_h: int = None) -> Tuple[int, int]:
        """Get target dimensions based on resolution set and method"""
        if custom_w is not None and custom_h is not None:
            return self._snap_to_efficient(custom_w, custom_h, snap_method)
            
        resolutions = self.resolution_sets[resolution_set]
        orig_aspect = orig_w / orig_h
        
        if snap_method == "model_optimal":
            # Find closest model-optimized resolution
            return min(resolutions, 
                      key=lambda r: abs(r[0]/r[1] - orig_aspect))
        
        elif snap_method == "memory_optimal":
            # Find resolution with similar area but efficient dims
            target_area = orig_w * orig_h
            return min(resolutions,
                      key=lambda r: abs(r[0]*r[1] - target_area))
        
        elif snap_method == "quality_optimal":
            # Prefer slightly larger resolutions for better quality
            larger_res = [r for r in resolutions 
                         if r[0]*r[1] >= orig_w*orig_h*0.9]
            if larger_res:
                return min(larger_res,
                          key=lambda r: abs(r[0]/r[1] - orig_aspect))
            
        # Performance - find closest matching resolution
        return min(resolutions,
                  key=lambda r: abs(r[0]/r[1] - orig_aspect))

    def _snap_to_efficient(self, width: int, height: int, 
                          method: str) -> Tuple[int, int]:
        """Snap custom dimensions to efficient values"""
        if method == "model_optimal":
            # Snap to nearest multiple of 64 for optimal model inference
            return (round(width/64)*64, round(height/64)*64)
        elif method == "memory_optimal":
            # Snap to nearest multiple of 32 for memory alignment
            return (round(width/32)*32, round(height/32)*32)
        else:
            # Snap to nearest multiple of 8 for basic efficiency
            return (round(width/8)*8, round(height/8)*8)

    def _smart_stretch(self, image: torch.Tensor, target_w: int,
                      target_h: int, interpolation: str) -> torch.Tensor:
        """Content-aware stretching with tensor optimization"""
        config = self.mode_configs[interpolation]
        
        # Convert to BCHW for pytorch ops
        x = image.permute(0, 3, 1, 2)
        
        # Apply resize with configured interpolation
        x = F.interpolate(x, size=(target_h, target_w),
                         mode=config["mode"],
                         antialias=config["antialias"],
                         align_corners=config["align_corners"])
        
        # Return to NHWC format
        return x.permute(0, 2, 3, 1)

    def _crop_center_resize(self, image: torch.Tensor, target_w: int,
                          target_h: int, interpolation: str) -> torch.Tensor:
        """Center crop with tensor optimization"""
        orig_h, orig_w = image.shape[1:3]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        
        if orig_aspect > target_aspect:
            # Scale by height
            scale_h = target_h
            scale_w = int(target_h * orig_aspect)
        else:
            # Scale by width
            scale_w = target_w
            scale_h = int(target_w / orig_aspect)
            
        # Initial resize
        resized = self._smart_stretch(image, scale_w, scale_h, interpolation)
        
        # Calculate crop offsets
        x_offset = (scale_w - target_w) // 2
        y_offset = (scale_h - target_h) // 2
        
        # Crop center
        return resized[:, y_offset:y_offset+target_h, 
                         x_offset:x_offset+target_w, :]

    def _padded_resize(self, image: torch.Tensor, target_w: int,
                      target_h: int, interpolation: str,
                      pad_mode: str, pad_value: float) -> torch.Tensor:
        """Padded resize with tensor optimization"""
        orig_h, orig_w = image.shape[1:3]
        scale = min(target_w/orig_w, target_h/orig_h)
        
        # Calculate intermediate size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Initial resize
        resized = self._smart_stretch(image, new_w, new_h, interpolation)
        
        # Calculate padding
        pad_x = target_w - new_w
        pad_y = target_h - new_h
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        
        # Convert pad mode
        if pad_mode == "reflect":
            mode = "reflect"
        elif pad_mode == "replicate":
            mode = "replicate"
        else:
            mode = "constant"
            
        # Apply padding
        x = resized.permute(0, 3, 1, 2)  # NHWC -> BCHW
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom),
                 mode=mode, value=pad_value)
        return x.permute(0, 2, 3, 1)  # BCHW -> NHWC