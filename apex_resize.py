#!/usr/bin/env python3
"""
Apex_Resize.py - Smart resolution snapping for AI compatibility
YouTube Channel: Apex Artist
Auto-snap to optimal resolutions with intelligent scaling
"""
import torch
import torch.nn.functional as F
import math

class ApexSmartResize:
    """
    Apex Smart Resize - Automatically snaps to closest compatible resolution
    Intelligent resolution detection and scaling with proportion preservation
    """
    
    def __init__(self):
        # Define compatible resolutions for different AI models
        self.resolution_sets = {
            "Standard": [
                (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
                (1344, 768), (768, 1344), (1536, 640), (640, 1536), 
                (832, 1280), (1280, 832), (704, 1504), (1504, 704),
                (896, 1344), (1344, 896), (960, 1280), (1280, 960),
                (512, 512), (768, 768), (640, 640)
            ],
            "Extended": [
                (1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
                (1344, 768), (768, 1344), (1536, 640), (640, 1536), (1728, 576),
                (576, 1728), (1920, 512), (512, 1920), (2048, 512), (512, 2048),
                (832, 1280), (1280, 832), (704, 1504), (1504, 704),
                (960, 1536), (1536, 960), (1088, 1472), (1472, 1088)
            ],
            "Flux": [
                (1024, 1024), (768, 1344), (832, 1216), (896, 1152), (1152, 896),
                (1216, 832), (1344, 768), (512, 512), (640, 1536), (1536, 640),
                (704, 1504), (1504, 704), (832, 1280), (1280, 832)
            ],
            "Portrait": [
                (832, 1216), (768, 1344), (640, 1536), (896, 1152), 
                (832, 1280), (704, 1504), (512, 768), (576, 1024),
                (640, 960), (720, 1280), (768, 1024), (896, 1344)
            ],
            "Landscape": [
                (1216, 832), (1344, 768), (1536, 640), (1152, 896),
                (1280, 832), (1504, 704), (768, 512), (1024, 576),
                (960, 640), (1280, 720), (1024, 768), (1344, 896)
            ],
            "Square": [
                (512, 512), (640, 640), (768, 768), (832, 832), (896, 896),
                (1024, 1024), (1152, 1152), (1216, 1216), (1280, 1280), (1344, 1344)
            ]
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_set": ([
                    "Standard",      # Core SDXL/Flux resolutions
                    "Extended",      # Extra experimental sizes  
                    "Flux",          # Flux-optimized
                    "Portrait",      # Tall formats
                    "Landscape",     # Wide formats
                    "Square"         # Square only
                ], {"default": "Standard"}),
                "snap_method": ([
                    "keep_proportion",   # Scale largest side first, maintain aspect ratio
                    "closest_area",      # Snap to closest total pixel count
                    "closest_ratio",     # Snap to closest aspect ratio
                    "prefer_larger",     # Prefer larger resolutions
                    "prefer_smaller",    # Prefer smaller resolutions
                ], {"default": "keep_proportion"}),
                "resize_mode": ([
                    "stretch",           # Stretch to exact dimensions
                    "crop_center",       # Crop from center
                    "fit_pad_black",     # Fit with black padding
                    "fit_pad_white",     # Fit with white padding  
                    "fit_pad_edge"       # Fit with edge extension
                ], {"default": "stretch"}),
                "interpolation": ([
                    "lanczos",
                    "bicubic",
                    "bilinear", 
                    "nearest"
                ], {"default": "lanczos"}),
                "show_candidates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show resolution candidates in console"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "scale_factor", "resolution_info")
    FUNCTION = "smart_resize"
    CATEGORY = "ApexArtist/Image"

    def smart_resize(self, image, resolution_set, snap_method, resize_mode, interpolation, show_candidates):
        
        try:
            # Get input dimensions
            if len(image.shape) == 4:
                batch_size, orig_h, orig_w, channels = image.shape
            else:
                image = image.unsqueeze(0)
                batch_size, orig_h, orig_w, channels = image.shape
            
            orig_area = orig_w * orig_h
            orig_aspect = orig_w / orig_h
            
            print(f"🖼️  Original: {orig_w}x{orig_h} (aspect: {orig_aspect:.3f})")
            
            # Find best target resolution
            target_w, target_h, info = self._find_best_resolution(
                orig_w, orig_h, resolution_set, snap_method, show_candidates
            )
            
            target_area = target_w * target_h
            scale_factor = math.sqrt(target_area / orig_area)
            
            print(f"🎯 Target: {target_w}x{target_h} (scale: {scale_factor:.2f}x)")
            print(f"📝 {info}")
            
            # Resize the image
            resized_image = self._apply_resize(image, target_w, target_h, resize_mode, interpolation)
            
            return (resized_image, target_w, target_h, scale_factor, info)
            
        except Exception as e:
            print(f"❌ Apex Smart Resize Error: {str(e)}")
            return (image, orig_w, orig_h, 1.0, f"Error: {str(e)}")
    
    def _find_best_resolution(self, orig_w, orig_h, resolution_set, snap_method, show_candidates):
        """Find the best target resolution based on method"""
        
        resolutions = self.resolution_sets[resolution_set]
        orig_area = orig_w * orig_h
        orig_aspect = orig_w / orig_h
        
        if snap_method == "keep_proportion":
            return self._keep_proportion_snap(orig_w, orig_h, resolutions, show_candidates)
        
        # Other methods
        candidates = []
        
        for w, h in resolutions:
            area = w * h
            aspect = w / h
            scale_factor = math.sqrt(area / orig_area)
            aspect_diff = abs(aspect - orig_aspect)
            area_diff = abs(area - orig_area)
            
            candidates.append({
                'resolution': (w, h),
                'area': area,
                'aspect': aspect,
                'scale_factor': scale_factor,
                'aspect_diff': aspect_diff,
                'area_diff': area_diff
            })
        
        # Sort candidates based on method
        if snap_method == "closest_area":
            candidates.sort(key=lambda x: x['area_diff'])
            best = candidates[0]
            info = f"Closest area match from {resolution_set}"
            
        elif snap_method == "closest_ratio":
            candidates.sort(key=lambda x: x['aspect_diff'])
            best = candidates[0]
            info = f"Closest aspect ratio from {resolution_set}"
            
        elif snap_method == "prefer_larger":
            larger_candidates = [c for c in candidates if c['area'] >= orig_area]
            if larger_candidates:
                larger_candidates.sort(key=lambda x: x['area'])
                best = larger_candidates[0]
            else:
                candidates.sort(key=lambda x: x['area'], reverse=True)
                best = candidates[0]
            info = f"Prefer larger from {resolution_set}"
            
        else:  # prefer_smaller
            smaller_candidates = [c for c in candidates if c['area'] <= orig_area]
            if smaller_candidates:
                smaller_candidates.sort(key=lambda x: x['area'], reverse=True)
                best = smaller_candidates[0]
            else:
                candidates.sort(key=lambda x: x['area'])
                best = candidates[0]
            info = f"Prefer smaller from {resolution_set}"
        
        # Show candidates if requested
        if show_candidates:
            print("🏆 Top 5 resolution candidates:")
            sorted_candidates = sorted(candidates, key=lambda x: x['area_diff'])[:5]
            for i, c in enumerate(sorted_candidates):
                w, h = c['resolution']
                marker = "👑" if c == best else f"  {i+1}."
                print(f"{marker} {w}x{h} (scale: {c['scale_factor']:.2f}x, aspect: {c['aspect']:.3f})")
        
        target_w, target_h = best['resolution']
        return target_w, target_h, info
    
    def _keep_proportion_snap(self, orig_w, orig_h, resolutions, show_candidates):
        """Scale by largest dimension while maintaining aspect ratio"""
        
        orig_aspect = orig_w / orig_h
        is_portrait = orig_h > orig_w
        
        best_match = None
        best_score = float('inf')
        candidates = []
        
        for target_w, target_h in resolutions:
            target_is_portrait = target_h > target_w
            
            # Only consider resolutions with same orientation
            if is_portrait == target_is_portrait:
                
                if is_portrait:
                    # Scale by height (largest dimension)
                    scale_factor = target_h / orig_h
                    calculated_w = orig_w * scale_factor
                    
                    # Round to nearest multiple of 64 for better compatibility
                    snapped_w = round(calculated_w / 64) * 64
                    
                    # Check if this creates a valid resolution
                    if abs(snapped_w - target_w) <= 64:  # Allow some tolerance
                        aspect_diff = abs((target_w / target_h) - orig_aspect)
                        scale_diff = abs(scale_factor - 1.0)
                        
                        # Scoring: prefer similar aspect ratio and reasonable scaling
                        score = aspect_diff * 10 + scale_diff * 2
                        
                        candidates.append({
                            'resolution': (target_w, target_h),
                            'scale_factor': scale_factor,
                            'aspect_diff': aspect_diff,
                            'score': score
                        })
                        
                        if score < best_score:
                            best_score = score
                            best_match = (target_w, target_h)
                
                else:  # Landscape
                    # Scale by width (largest dimension)
                    scale_factor = target_w / orig_w
                    calculated_h = orig_h * scale_factor
                    
                    snapped_h = round(calculated_h / 64) * 64
                    
                    if abs(snapped_h - target_h) <= 64:
                        aspect_diff = abs((target_w / target_h) - orig_aspect)
                        scale_diff = abs(scale_factor - 1.0)
                        
                        score = aspect_diff * 10 + scale_diff * 2
                        
                        candidates.append({
                            'resolution': (target_w, target_h),
                            'scale_factor': scale_factor,
                            'aspect_diff': aspect_diff,
                            'score': score
                        })
                        
                        if score < best_score:
                            best_score = score
                            best_match = (target_w, target_h)
        
        # Fallback to closest aspect ratio if no good match
        if best_match is None:
            best_aspect_diff = float('inf')
            for w, h in resolutions:
                aspect_diff = abs((w/h) - orig_aspect)
                if aspect_diff < best_aspect_diff:
                    best_aspect_diff = aspect_diff
                    best_match = (w, h)
        
        # Show candidates if requested
        if show_candidates and candidates:
            print("🏆 Keep Proportion candidates:")
            sorted_candidates = sorted(candidates, key=lambda x: x['score'])[:5]
            for i, c in enumerate(sorted_candidates):
                w, h = c['resolution']
                marker = "👑" if (w, h) == best_match else f"  {i+1}."
                print(f"{marker} {w}x{h} (scale: {c['scale_factor']:.2f}x, aspect_diff: {c['aspect_diff']:.3f})")
        
        target_w, target_h = best_match
        info = f"Keep proportion snap from {len(resolutions)} resolutions"
        
        return target_w, target_h, info
    
    def _apply_resize(self, image, target_w, target_h, resize_mode, interpolation):
        """Apply the actual resizing with specified method"""
        
        if resize_mode == "stretch":
            return self._resize_tensor(image, target_w, target_h, interpolation)
        
        elif resize_mode == "crop_center":
            return self._crop_center_resize(image, target_w, target_h, interpolation)
        
        elif resize_mode == "fit_pad_black":
            return self._fit_pad_resize(image, target_w, target_h, interpolation, pad_color=0.0)
        
        elif resize_mode == "fit_pad_white":
            return self._fit_pad_resize(image, target_w, target_h, interpolation, pad_color=1.0)
        
        elif resize_mode == "fit_pad_edge":
            return self._fit_pad_edge_resize(image, target_w, target_h, interpolation)
        
        else:
            return self._resize_tensor(image, target_w, target_h, interpolation)
    
    def _resize_tensor(self, image, width, height, interpolation):
        """Core tensor resize function"""
        
        image_bchw = image.permute(0, 3, 1, 2)
        
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear", 
            "bicubic": "bicubic",
            "lanczos": "bicubic"  # PyTorch fallback
        }
        
        mode = mode_map.get(interpolation, "bicubic")
        antialias = mode in ["bilinear", "bicubic"]
        
        resized = F.interpolate(image_bchw, size=(height, width), 
                              mode=mode, antialias=antialias)
        
        return resized.permute(0, 2, 3, 1)
    
    def _crop_center_resize(self, image, target_w, target_h, interpolation):
        """Resize to cover target, then center crop"""
        
        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        
        if orig_aspect > target_aspect:
            # Scale by height, crop width
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        else:
            # Scale by width, crop height
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        
        # Resize to cover
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        
        # Center crop
        crop_x = max(0, (new_w - target_w) // 2)
        crop_y = max(0, (new_h - target_h) // 2)
        
        cropped = resized[:, crop_y:crop_y+target_h, crop_x:crop_x+target_w, :]
        
        return cropped
    
    def _fit_pad_resize(self, image, target_w, target_h, interpolation, pad_color):
        """Fit image with solid color padding"""
        
        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        
        if orig_aspect > target_aspect:
            # Fit to width
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            # Fit to height
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        
        # Resize to fit
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(image_bchw, (pad_left, pad_right, pad_top, pad_bottom), 
                          mode='constant', value=pad_color)
            result = padded.permute(0, 2, 3, 1)
        else:
            result = resized
        
        return result
    
    def _fit_pad_edge_resize(self, image, target_w, target_h, interpolation):
        """Fit image with edge replication padding"""
        
        orig_h, orig_w = image.shape[1], image.shape[2]
        orig_aspect = orig_w / orig_h
        target_aspect = target_w / target_h
        
        if orig_aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        
        # Resize to fit
        resized = self._resize_tensor(image, new_w, new_h, interpolation)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        if pad_w > 0 or pad_h > 0:
            image_bchw = resized.permute(0, 3, 1, 2)
            padded = F.pad(image_bchw, (pad_left, pad_right, pad_top, pad_bottom), 
                          mode='replicate')
            result = padded.permute(0, 2, 3, 1)
        else:
            result = resized
        
        return result

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ApexSmartResize": ApexSmartResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexSmartResize": "Apex Smart Resize"
}
