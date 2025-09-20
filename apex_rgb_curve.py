#!/usr/bin/env python3
"""
Apex_RGB_Curve.py - Professional RGB curve color adjustment with curve points
YouTube Channel: Apex Artist
Uses curve points instead of individual sliders for better UX
"""
import torch
import json

class ApexRGBCurve:
    """
    Apex RGB Curve - Professional color grading with curve points
    Uses JSON strings to define curve points for each channel
    """
    
    def __init__(self):
        # Curve presets as point arrays
        self.curve_presets = {
            "linear": [[0, 0], [255, 255]],
            "slight_s": [[0, 0], [48, 32], [128, 128], [208, 224], [255, 255]],
            "strong_s": [[0, 0], [32, 16], [128, 128], [224, 240], [255, 255]],
            "brighten": [[0, 0], [80, 96], [160, 192], [224, 240], [255, 255]],
            "darken": [[0, 0], [32, 16], [96, 64], [176, 144], [255, 255]],
            "contrast": [[0, 0], [40, 16], [128, 128], [216, 240], [255, 255]],
            "low_contrast": [[0, 0], [76, 96], [128, 128], [180, 160], [255, 255]],
            "film_look": [[16, 16], [60, 48], [140, 160], [200, 220], [240, 240]],
            "vintage": [[20, 32], [80, 64], [120, 128], [180, 192], [235, 220]],
            "crushed_blacks": [[0, 32], [80, 80], [128, 128], [192, 192], [255, 255]],
            "lifted_blacks": [[0, 0], [96, 128], [144, 160], [200, 216], [255, 255]]
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_options = list(cls().curve_presets.keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                
                # Curve definitions as JSON strings or presets
                "master_curve": ("STRING", {
                    "default": "[[0,0],[255,255]]",
                    "multiline": True,
                    "tooltip": "JSON array of [x,y] points, e.g. [[0,0],[64,48],[128,128],[192,208],[255,255]]"
                }),
                "red_curve": ("STRING", {
                    "default": "[[0,0],[255,255]]", 
                    "multiline": True,
                    "tooltip": "JSON array of [x,y] points for red channel"
                }),
                "green_curve": ("STRING", {
                    "default": "[[0,0],[255,255]]",
                    "multiline": True, 
                    "tooltip": "JSON array of [x,y] points for green channel"
                }),
                "blue_curve": ("STRING", {
                    "default": "[[0,0],[255,255]]",
                    "multiline": True,
                    "tooltip": "JSON array of [x,y] points for blue channel"
                }),
                
                # Quick presets
                "master_preset": (["custom"] + preset_options, {"default": "custom"}),
                "red_preset": (["custom"] + preset_options, {"default": "custom"}),
                "green_preset": (["custom"] + preset_options, {"default": "custom"}),
                "blue_preset": (["custom"] + preset_options, {"default": "custom"}),
                
                # Blend settings
                "blend_mode": ([
                    "normal", "multiply", "screen", "overlay", "soft_light",
                    "hard_light", "color_dodge", "color_burn", "darken", "lighten"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Advanced options
                "preserve_luminance": ("BOOLEAN", {"default": False}),
                "curve_smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "auto_sort_points": ("BOOLEAN", {"default": True, "tooltip": "Automatically sort points by X coordinate"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curve_info")
    FUNCTION = "apply_rgb_curves"
    CATEGORY = "Apex Artist/Color"

    def apply_rgb_curves(self, image, master_curve, red_curve, green_curve, blue_curve,
                        master_preset, red_preset, green_preset, blue_preset,
                        blend_mode, opacity, preserve_luminance, curve_smoothing, auto_sort_points):
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            device = image.device
            
            # Parse curve points
            master_points = self._get_curve_points(master_curve, master_preset, auto_sort_points)
            red_points = self._get_curve_points(red_curve, red_preset, auto_sort_points)
            green_points = self._get_curve_points(green_curve, green_preset, auto_sort_points)
            blue_points = self._get_curve_points(blue_curve, blue_preset, auto_sort_points)
            
            # Create lookup tables for each channel
            master_lut = self._create_curve_lut(master_points, curve_smoothing, device)
            red_lut = self._create_curve_lut(red_points, curve_smoothing, device)
            green_lut = self._create_curve_lut(green_points, curve_smoothing, device)
            blue_lut = self._create_curve_lut(blue_points, curve_smoothing, device)
            
            # Apply curves
            if preserve_luminance:
                result_image = self._apply_luminance_preserving_curves(
                    image, master_lut, red_lut, green_lut, blue_lut
                )
            else:
                result_image = self._apply_direct_curves(
                    image, master_lut, red_lut, green_lut, blue_lut
                )
            
            # Apply blend mode
            if blend_mode != "normal" and opacity > 0:
                result_image = self._apply_blend_mode(image, result_image, blend_mode, opacity)
            elif opacity < 1.0:
                result_image = image * (1.0 - opacity) + result_image * opacity
            
            result_image = torch.clamp(result_image, 0.0, 1.0)
            
            # Generate info
            curve_info = self._generate_curve_info(
                master_points, red_points, green_points, blue_points,
                blend_mode, opacity, preserve_luminance
            )
            
            return (result_image, curve_info)
            
        except Exception as e:
            error_info = f"RGB Curve Error: {str(e)}"
            return (image, error_info)
    
    def _get_curve_points(self, curve_string, preset, auto_sort):
        """Parse curve points from JSON string or use preset"""
        
        if preset != "custom" and preset in self.curve_presets:
            return self.curve_presets[preset]
        
        try:
            # Try to parse JSON
            points = json.loads(curve_string)
            
            # Validate and normalize points
            validated_points = []
            for point in points:
                if len(point) >= 2:
                    x = max(0, min(255, int(point[0])))
                    y = max(0, min(255, int(point[1])))
                    validated_points.append([x, y])
            
            if len(validated_points) < 2:
                # Default to linear if not enough points
                return [[0, 0], [255, 255]]
            
            # Auto-sort by X coordinate if requested
            if auto_sort:
                validated_points.sort(key=lambda p: p[0])
            
            # Ensure endpoints
            if validated_points[0][0] != 0:
                validated_points.insert(0, [0, validated_points[0][1]])
            if validated_points[-1][0] != 255:
                validated_points.append([255, validated_points[-1][1]])
            
            return validated_points
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Fallback to linear curve on parse error
            print(f"Curve parse error: {e}, using linear curve")
            return [[0, 0], [255, 255]]
    
    def _create_curve_lut(self, points, smoothing, device):
        """Create 256-value lookup table from curve points with smooth interpolation"""
        
        lut = torch.zeros(256, dtype=torch.float32, device=device)
        
        # Convert points to tensors
        x_points = torch.tensor([p[0] for p in points], dtype=torch.float32, device=device)
        y_points = torch.tensor([p[1] / 255.0 for p in points], dtype=torch.float32, device=device)
        
        # Create smooth interpolation
        for i in range(256):
            x = float(i)
            
            # Find surrounding points
            if x <= x_points[0]:
                lut[i] = y_points[0]
            elif x >= x_points[-1]:
                lut[i] = y_points[-1]
            else:
                # Find interpolation segment
                for j in range(len(x_points) - 1):
                    if x_points[j] <= x <= x_points[j + 1]:
                        # Linear interpolation
                        if x_points[j + 1] - x_points[j] > 0:
                            t = (x - x_points[j]) / (x_points[j + 1] - x_points[j])
                            
                            # Apply smoothing (cubic interpolation)
                            if smoothing > 0:
                                t = self._smooth_interpolation(t, smoothing)
                            
                            lut[i] = y_points[j] * (1 - t) + y_points[j + 1] * t
                        else:
                            lut[i] = y_points[j]
                        break
        
        return lut
    
    def _smooth_interpolation(self, t, smoothing):
        """Apply smooth interpolation (cubic hermite)"""
        if smoothing <= 0:
            return t
        
        # Cubic hermite smoothing
        smooth_t = t * t * (3.0 - 2.0 * t)
        return t * (1.0 - smoothing) + smooth_t * smoothing
    
    def _apply_direct_curves(self, image, master_lut, red_lut, green_lut, blue_lut):
        """Apply curves directly to RGB channels"""
        
        result = image.clone()
        device = image.device
        
        # Convert to 0-255 range for LUT lookup
        image_255 = (image * 255).round().long()
        image_255 = torch.clamp(image_255, 0, 255)
        
        # Apply master curve if not linear
        linear_lut = torch.linspace(0, 1, 256, device=device)
        if not torch.allclose(master_lut, linear_lut, atol=1e-6):
            for c in range(min(3, image.shape[-1])):
                result[:, :, :, c] = master_lut[image_255[:, :, :, c]]
        
        # Convert back to 0-255 for individual channel curves
        result_255 = (result * 255).round().long()
        result_255 = torch.clamp(result_255, 0, 255)
        
        # Apply individual channel curves
        if not torch.allclose(red_lut, linear_lut, atol=1e-6):
            result[:, :, :, 0] = red_lut[result_255[:, :, :, 0]]
        
        if not torch.allclose(green_lut, linear_lut, atol=1e-6):
            result[:, :, :, 1] = green_lut[result_255[:, :, :, 1]]
        
        if not torch.allclose(blue_lut, linear_lut, atol=1e-6):
            result[:, :, :, 2] = blue_lut[result_255[:, :, :, 2]]
        
        return result
    
    def _apply_luminance_preserving_curves(self, image, master_lut, red_lut, green_lut, blue_lut):
        """Apply curves while preserving luminance"""
        
        device = image.device
        
        # Calculate original luminance (Rec. 709)
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
        original_luma = torch.sum(image * luma_weights, dim=-1, keepdim=True)
        
        # Apply curves normally
        curved = self._apply_direct_curves(image, master_lut, red_lut, green_lut, blue_lut)
        
        # Calculate new luminance
        new_luma = torch.sum(curved * luma_weights, dim=-1, keepdim=True)
        
        # Preserve original luminance
        luma_ratio = torch.where(new_luma > 0.001, original_luma / new_luma, 1.0)
        result = curved * luma_ratio
        
        return result
    
    def _apply_blend_mode(self, base, overlay, mode, opacity):
        """Apply various blend modes"""
        
        if mode == "multiply":
            blended = base * overlay
        elif mode == "screen":
            blended = 1.0 - (1.0 - base) * (1.0 - overlay)
        elif mode == "overlay":
            blended = torch.where(base < 0.5, 
                                2.0 * base * overlay,
                                1.0 - 2.0 * (1.0 - base) * (1.0 - overlay))
        elif mode == "soft_light":
            blended = torch.where(overlay < 0.5,
                                base - (1.0 - 2.0 * overlay) * base * (1.0 - base),
                                base + (2.0 * overlay - 1.0) * (torch.sqrt(torch.clamp(base, 0.0, 1.0)) - base))
        elif mode == "hard_light":
            blended = torch.where(overlay < 0.5,
                                2.0 * base * overlay,
                                1.0 - 2.0 * (1.0 - base) * (1.0 - overlay))
        elif mode == "color_dodge":
            blended = torch.where(overlay >= 0.999, 1.0, 
                                torch.clamp(base / torch.clamp(1.0 - overlay, 0.001, 1.0), 0.0, 1.0))
        elif mode == "color_burn":
            blended = torch.where(overlay <= 0.001, 0.0,
                                1.0 - torch.clamp((1.0 - base) / torch.clamp(overlay, 0.001, 1.0), 0.0, 1.0))
        elif mode == "darken":
            blended = torch.min(base, overlay)
        elif mode == "lighten":
            blended = torch.max(base, overlay)
        else:  # normal
            blended = overlay
        
        return base * (1.0 - opacity) + blended * opacity
    
    def _generate_curve_info(self, master_points, red_points, green_points, blue_points,
                           blend_mode, opacity, preserve_luminance):
        """Generate curve information string"""
        
        info_lines = [
            f"RGB Curves Applied:",
            f"Master: {len(master_points)} points",
            f"Red: {len(red_points)} points", 
            f"Green: {len(green_points)} points",
            f"Blue: {len(blue_points)} points",
            f"Blend: {blend_mode} ({opacity:.0%})",
            f"Luminance Preserved: {preserve_luminance}"
        ]
        
        return " | ".join(info_lines)