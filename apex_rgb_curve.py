#!/usr/bin/env python3
"""
Apex_RGB_Curve.py - Professional RGB curve color adjustment
YouTube Channel: Apex Artist
Individual and master RGB curve controls with multiple blend modes
"""
import torch
import torch.nn.functional as F
import math

class ApexRGBCurve:
    """
    Apex RGB Curve - Professional color grading with individual RGB channel curves
    Supports master curve, individual R/G/B curves, and multiple blend modes
    """
    
    def __init__(self):
        # Curve presets for quick access
        self.curve_presets = {
            "linear": [0, 64, 128, 192, 255],
            "slight_s": [0, 48, 128, 208, 255],
            "strong_s": [0, 32, 128, 224, 255],
            "brighten": [0, 80, 160, 224, 255],
            "darken": [0, 32, 96, 176, 255],
            "contrast": [0, 40, 128, 216, 255],
            "low_contrast": [0, 76, 128, 180, 255],
            "film_look": [16, 60, 140, 200, 240],
            "vintage": [20, 80, 120, 180, 235],
            "crushed_blacks": [32, 80, 128, 192, 255],
            "lifted_blacks": [0, 96, 144, 200, 255]
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_options = list(cls().curve_presets.keys())
        
        return {
            "required": {
                "image": ("IMAGE",),
                
                # Master curve controls
                "master_preset": (preset_options, {"default": "linear"}),
                "master_shadows": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "master_darks": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1}),
                "master_mids": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "master_lights": ("INT", {"default": 192, "min": 0, "max": 255, "step": 1}),
                "master_highlights": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                
                # Red channel
                "red_preset": (preset_options, {"default": "linear"}),
                "red_shadows": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "red_darks": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1}),
                "red_mids": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "red_lights": ("INT", {"default": 192, "min": 0, "max": 255, "step": 1}),
                "red_highlights": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                
                # Green channel
                "green_preset": (preset_options, {"default": "linear"}),
                "green_shadows": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green_darks": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1}),
                "green_mids": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "green_lights": ("INT", {"default": 192, "min": 0, "max": 255, "step": 1}),
                "green_highlights": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                
                # Blue channel
                "blue_preset": (preset_options, {"default": "linear"}),
                "blue_shadows": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue_darks": ("INT", {"default": 64, "min": 0, "max": 255, "step": 1}),
                "blue_mids": ("INT", {"default": 128, "min": 0, "max": 255, "step": 1}),
                "blue_lights": ("INT", {"default": 192, "min": 0, "max": 255, "step": 1}),
                "blue_highlights": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                
                # Blend settings
                "blend_mode": ([
                    "normal",
                    "multiply", 
                    "screen",
                    "overlay",
                    "soft_light",
                    "hard_light",
                    "color_dodge",
                    "color_burn",
                    "darken",
                    "lighten"
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Advanced options
                "use_preset_override": ("BOOLEAN", {"default": False, "tooltip": "Use preset values instead of manual points"}),
                "preserve_luminance": ("BOOLEAN", {"default": False, "tooltip": "Apply curves in luminance-preserving mode"}),
                "curve_smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smooth curve interpolation"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curve_info")
    FUNCTION = "apply_rgb_curves"
    CATEGORY = "Apex Artist/Color"

    def apply_rgb_curves(self, image, 
                        master_preset, master_shadows, master_darks, master_mids, master_lights, master_highlights,
                        red_preset, red_shadows, red_darks, red_mids, red_lights, red_highlights,
                        green_preset, green_shadows, green_darks, green_mids, green_lights, green_highlights,
                        blue_preset, blue_shadows, blue_darks, blue_mids, blue_lights, blue_highlights,
                        blend_mode, opacity, use_preset_override, preserve_luminance, curve_smoothing):
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            device = image.device
            batch_size, height, width, channels = image.shape
            
            # Get curve points for each channel
            if use_preset_override:
                master_points = self.curve_presets[master_preset]
                red_points = self.curve_presets[red_preset]
                green_points = self.curve_presets[green_preset]
                blue_points = self.curve_presets[blue_preset]
            else:
                master_points = [master_shadows, master_darks, master_mids, master_lights, master_highlights]
                red_points = [red_shadows, red_darks, red_mids, red_lights, red_highlights]
                green_points = [green_shadows, green_darks, green_mids, green_lights, green_highlights]
                blue_points = [blue_shadows, blue_darks, blue_mids, blue_lights, blue_highlights]
            
            # Create lookup tables for each channel
            master_lut = self._create_curve_lut(master_points, curve_smoothing, device)
            red_lut = self._create_curve_lut(red_points, curve_smoothing, device)
            green_lut = self._create_curve_lut(green_points, curve_smoothing, device)
            blue_lut = self._create_curve_lut(blue_points, curve_smoothing, device)
            
            # Convert to working space (0-1 range)
            working_image = image.clone()
            
            if preserve_luminance:
                # Apply curves in luminance-preserving mode
                result_image = self._apply_luminance_preserving_curves(
                    working_image, master_lut, red_lut, green_lut, blue_lut
                )
            else:
                # Apply curves directly to RGB channels
                result_image = self._apply_direct_curves(
                    working_image, master_lut, red_lut, green_lut, blue_lut
                )
            
            # Apply blend mode if not normal
            if blend_mode != "normal" and opacity > 0:
                result_image = self._apply_blend_mode(image, result_image, blend_mode, opacity)
            elif opacity < 1.0:
                result_image = image * (1.0 - opacity) + result_image * opacity
            
            # Clamp values to valid range
            result_image = torch.clamp(result_image, 0.0, 1.0)
            
            # Generate info string
            curve_info = self._generate_curve_info(
                master_points, red_points, green_points, blue_points,
                blend_mode, opacity, preserve_luminance
            )
            
            return (result_image, curve_info)
            
        except Exception as e:
            error_info = f"RGB Curve Error: {str(e)}"
            return (image, error_info)
    
    def _create_curve_lut(self, points, smoothing, device):
        """Create a 256-value lookup table from 5 control points"""
        
        # Input positions for the 5 points (0, 64, 128, 192, 255)
        input_pos = torch.tensor([0, 64, 128, 192, 255], dtype=torch.float32, device=device)
        output_values = torch.tensor([p / 255.0 for p in points], dtype=torch.float32, device=device)
        
        # Create lookup table
        lut = torch.zeros(256, dtype=torch.float32, device=device)
        
        # Create interpolation indices
        indices = torch.arange(256, dtype=torch.float32, device=device)
        
        # Simple linear interpolation
        for i in range(256):
            # Find which segment we're in
            for j in range(len(input_pos) - 1):
                if input_pos[j] <= i <= input_pos[j + 1]:
                    # Linear interpolation
                    if input_pos[j + 1] - input_pos[j] > 0:
                        t = (i - input_pos[j]) / (input_pos[j + 1] - input_pos[j])
                        
                        # Apply smoothing if requested
                        if smoothing > 0:
                            t = self._smooth_step(t, smoothing)
                        
                        lut[i] = output_values[j] * (1 - t) + output_values[j + 1] * t
                    else:
                        lut[i] = output_values[j]
                    break
        
        return lut
    
    def _smooth_step(self, t, smoothing):
        """Apply smoothing to interpolation parameter"""
        if smoothing <= 0:
            return t
        
        # Hermite interpolation for smooth curves
        smoothing = min(smoothing, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return t * (1.0 - smoothing) + smooth_t * smoothing
    
    def _apply_direct_curves(self, image, master_lut, red_lut, green_lut, blue_lut):
        """Apply curves directly to RGB channels"""
        
        result = image.clone()
        device = image.device
        
        # Convert to 0-255 range for LUT lookup
        image_255 = (image * 255).round().long()
        image_255 = torch.clamp(image_255, 0, 255)
        
        # Check if master curve is not linear
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
        
        # Apply opacity
        return base * (1.0 - opacity) + blended * opacity
    
    def _generate_curve_info(self, master_points, red_points, green_points, blue_points,
                           blend_mode, opacity, preserve_luminance):
        """Generate human-readable curve information"""
        
        def points_to_str(points):
            return f"[{', '.join(map(str, points))}]"
        
        info_lines = [
            f"RGB Curves Applied:",
            f"Master: {points_to_str(master_points)}",
            f"Red: {points_to_str(red_points)}",
            f"Green: {points_to_str(green_points)}",
            f"Blue: {points_to_str(blue_points)}",
            f"Blend: {blend_mode} ({opacity:.0%})",
            f"Luminance Preserved: {preserve_luminance}"
        ]
        
        return " | ".join(info_lines)