#!/usr/bin/env python3
"""
Apex_RGB_Curve.py - Professional RGB curve color adjustment
YouTube Channel: Apex Artist
Individual and master RGB curve controls with multiple blend modes
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from .apex_film_profiles import FilmStockProfiles
from .apex_color_science import FilmGrain, ColorScience
from . import CURVE_PRESETS, BLEND_MODES, DEFAULT_CURVE_POINTS

class ApexRGBCurve:
    """
    Apex RGB Curve - Professional color grading with individual RGB channel curves
    Supports master curve, individual R/G/B curves, and multiple blend modes
    """
    
    def __init__(self):
        from . import CURVE_PRESETS
        self.curve_presets = CURVE_PRESETS
            
            # Color Grading Inspired
            "warm_shadows": [(0, 0.1), (0.3, 0.35), (0.5, 0.5), (0.7, 0.65), (1, 0.9)],
            "cool_highlights": [(0, 0), (0.3, 0.25), (0.5, 0.5), (0.7, 0.8), (1, 1.1)],
            "cross_process": [(0, 0), (0.3, 0.4), (0.5, 0.45), (0.7, 0.6), (1, 0.9)],
            
            # Film Stock Emulation
            "kodak_portra": [(0, 0), (0.2, 0.2), (0.5, 0.55), (0.8, 0.85), (1, 0.95)],
            "fuji_provia": [(0, 0), (0.25, 0.2), (0.5, 0.5), (0.75, 0.8), (1, 1)],
            "ilford_delta": [(0.05, 0), (0.3, 0.25), (0.5, 0.5), (0.7, 0.75), (0.95, 1)],
            
            # HDR Style
            "hdr_natural": [(0, 0), (0.25, 0.3), (0.5, 0.5), (0.75, 0.7), (0.9, 0.85), (1, 0.9)],
            "hdr_dramatic": [(0, 0), (0.2, 0.2), (0.5, 0.6), (0.75, 0.8), (0.9, 0.9), (1, 1)],
            "hdr_detailed": [(0, 0), (0.2, 0.25), (0.4, 0.5), (0.6, 0.7), (0.8, 0.85), (1, 0.95)],
            
            # Time of Day
            "golden_hour": [(0, 0), (0.2, 0.25), (0.5, 0.6), (0.8, 0.85), (1, 0.95)],
            "blue_hour": [(0, 0), (0.3, 0.3), (0.5, 0.45), (0.7, 0.7), (1, 0.9)],
            "sunset_glow": [(0.05, 0), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8), (1, 0.95)],
            
            # Special Effects
            "high_key": [(0, 0.2), (0.25, 0.4), (0.5, 0.6), (0.75, 0.8), (1, 1)],
            "low_key": [(0, 0), (0.25, 0.2), (0.5, 0.4), (0.75, 0.6), (1, 0.8)],
            "infrared": [(0, 0), (0.3, 0.4), (0.5, 0.7), (0.7, 0.9), (1, 1)],
            
            # Modern Trends
            "teal_orange": [(0, 0), (0.2, 0.15), (0.5, 0.5), (0.8, 0.9), (1, 1)],
            "matte_fade": [(0.1, 0.1), (0.3, 0.35), (0.5, 0.5), (0.7, 0.7), (0.9, 0.85)],
            "clean_punch": [(0, 0), (0.25, 0.2), (0.5, 0.5), (0.75, 0.85), (1, 1)]
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        preset_options = list(cls().curve_presets.keys())
        film_profiles = FilmStockProfiles.get_profiles()
        
        return {
            "required": {
                "image": ("IMAGE",),
                
                # Film stock selection
                "film_simulation": (["None"] + list(film_profiles.keys()), {
                    "default": "None",
                    "tooltip": "Apply film stock characteristics"
                }),
                
                # Curve controls using CURVE type with histogram overlay
                "master_curve": ("CURVE_EDITOR", {
                    "default": DEFAULT_CURVE_POINTS,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Master Curve",
                    "color": (0.8, 0.8, 0.8),  # Gray for master curve
                    "show_histogram": True,
                    "show_preview": True,
                    "height": 256
                }),
                "red_curve": ("CURVE_EDITOR", {
                    "default": DEFAULT_CURVE_POINTS,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Red Channel",
                    "color": (1.0, 0.2, 0.2),  # Red curve
                    "show_histogram": True,
                    "height": 256
                }),
                "green_curve": ("CURVE_EDITOR", {
                    "default": DEFAULT_CURVE_POINTS,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Green Channel",
                    "color": (0.2, 1.0, 0.2),  # Green curve
                    "show_histogram": True,
                    "height": 256
                }),
                "blue_curve": ("CURVE_EDITOR", {
                    "default": DEFAULT_CURVE_POINTS,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "label": "Blue Channel",
                    "color": (0.2, 0.2, 1.0),  # Blue curve
                    "show_histogram": True,
                    "height": 256
                }),
                
                # Film simulation
                "film_profile": (list(FilmStockProfiles.get_profiles().keys()) + ["none"], {"default": "none"}),
                "enable_grain": ("BOOLEAN", {"default": True}),
                
                # Blend settings
                "blend_mode": (BLEND_MODES, {"default": "normal"})
                ], {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Advanced options
                "preserve_luminance": ("BOOLEAN", {"default": False, "tooltip": "Apply curves in luminance-preserving mode"}),
                "curve_smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Smooth curve interpolation"})
            },
            "optional": {
                "preset": (preset_options, {"default": "linear"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curve_info")
    FUNCTION = "apply_rgb_curves"
    CATEGORY = "Apex Artist/Color"

    def apply_rgb_curves(self, image, master_curve, red_curve, green_curve, blue_curve,
                        film_profile, enable_grain, blend_mode, opacity, 
                        preserve_luminance, curve_smoothing, preset=None):
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            batch_size, height, width, channels = image.shape
            
            # Apply film profile if selected
            if film_profile != "none":
                profile_data = FilmStockProfiles.get_profiles()[film_profile]
                film_curves = profile_data["curves"]
                characteristics = profile_data["characteristics"]
                
                # Merge film curves with user curves
                if preset is None:  # Only apply if no other preset is selected
                    master_curve = self._merge_curves(master_curve, film_curves["g"])
                    red_curve = self._merge_curves(red_curve, film_curves["r"])
                    green_curve = self._merge_curves(green_curve, film_curves["g"])
                    blue_curve = self._merge_curves(blue_curve, film_curves["b"])
            
            # Calculate histogram for visualization
            with torch.no_grad():
                # Convert to channels-first format for histogram calculation
                img_chw = image.permute(0, 3, 1, 2)[0]  # Take first batch
                
                # Calculate histograms for each channel
                histograms = []
                for channel in range(3):
                    hist = torch.histc(img_chw[channel], bins=256, min=0, max=1)
                    # Normalize histogram
                    hist = hist / hist.max()
                    histograms.append(hist)
                
                # Create RGB histogram visualization
                histogram_data = {
                    "histograms": histograms,
                    "width": 256,
                    "height": 128,
                    "colors": [(1,0,0), (0,1,0), (0,0,1)]  # RGB colors
                }
                
                # Store histogram data for UI visualization
                self._last_histogram = histogram_data
            
            # Handle preset if provided
            if preset and preset != "linear":
                preset_points = self.curve_presets[preset]
                preset_points_norm = [(x/255, y/255) for x, y in zip(
                    [0, 64, 128, 192, 255],
                    preset_points
                )]
                master_curve = preset_points_norm
                red_curve = preset_points_norm
                green_curve = preset_points_norm
                blue_curve = preset_points_norm
            
            # Convert curve points to lookup tables
            master_lut = self._curve_to_lut(master_curve, curve_smoothing)
            red_lut = self._curve_to_lut(red_curve, curve_smoothing)
            green_lut = self._curve_to_lut(green_curve, curve_smoothing)
            blue_lut = self._curve_to_lut(blue_curve, curve_smoothing)
            
            # Create lookup tables for each channel
            master_lut = self._create_curve_lut(master_curve, curve_smoothing)
            red_lut = self._create_curve_lut(red_curve, curve_smoothing)
            green_lut = self._create_curve_lut(green_curve, curve_smoothing)
            blue_lut = self._create_curve_lut(blue_curve, curve_smoothing)
            
            # Convert to working space (0-1 range)
            working_image = image.clone()
            
            # Apply film profile if selected
            if film_profile and film_profile != "none":
                profile = FilmStockProfiles.get_profiles()[film_profile]
                characteristics = profile["characteristics"]
                
                # Apply color science adjustments
                result_image = ColorScience.apply_color_science(working_image, characteristics)
                
                # Apply film grain if enabled
                if "grain_size" in characteristics:
                    height, width = working_image.shape[1:3]
                    grain = FilmGrain.generate_grain_pattern(
                        height, width, 
                        characteristics,
                        working_image.device
                    )
                    result_image = FilmGrain.apply_grain(result_image, grain, characteristics)
            
            if preserve_luminance:
                # Apply curves in luminance-preserving mode
                result_image = self._apply_luminance_preserving_curves(
                    result_image, master_lut, red_lut, green_lut, blue_lut
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
                master_curve, red_curve, green_curve, blue_curve,
                blend_mode, opacity, preserve_luminance
            )
            
            return (result_image, curve_info)
            
        except Exception as e:
            error_info = f"RGB Curve Error: {str(e)}"
            return (image, error_info)
    
    def _curve_to_lut(self, curve_points, smoothing):
        """Convert curve control points to a 256-value lookup table"""
        
        # Ensure we have points in ascending x order
        curve_points = sorted(curve_points, key=lambda x: x[0])
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in curve_points]
        y_coords = [p[1] for p in curve_points]
        
        # Create lookup table
        lut = torch.zeros(256, dtype=torch.float32)
        
        # Convert to numpy arrays for interpolation
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        
        if smoothing > 0:
            # Create a finer interpolation for smooth curves
            fine_x = np.linspace(0, 1, 1024)
            # Use cubic spline interpolation for smoothing
            fine_y = np.interp(fine_x, x_coords, y_coords)
            
            # Apply additional smoothing if needed
            if smoothing > 0.01:
                window = int(1024 * smoothing)
                if window % 2 == 0:
                    window += 1
                fine_y = np.convolve(fine_y, np.ones(window)/window, mode='same')
            
            # Interpolate back to 256 values
            lut_indices = np.linspace(0, 1023, 256).astype(int)
            lut = torch.from_numpy(fine_y[lut_indices]).float()
        else:
            # Direct linear interpolation to 256 values
            input_indices = np.linspace(0, 1, 256)
            lut = torch.from_numpy(np.interp(input_indices, x_coords, y_coords)).float()
        
        # Ensure valid range
        lut = torch.clamp(lut, 0.0, 1.0)
        
        return lut
    
    def _smooth_step(self, t, smoothing):
        """Apply smoothing to interpolation parameter"""
        if smoothing <= 0:
            return t
        
        # Hermite interpolation for smooth curves
        smoothing = min(smoothing, 1.0)
        smooth_t = t * t * (3.0 - 2.0 * t)
        return t * (1.0 - smoothing) + smooth_t * smoothing
    
    def _merge_curves(self, curve1, curve2, weight=0.5):
        """Merge two curves with optional weighting"""
        # Ensure both curves have x coordinates in ascending order
        curve1 = sorted(curve1, key=lambda x: x[0])
        curve2 = sorted(curve2, key=lambda x: x[0])
        
        # Get all unique x coordinates
        x_coords = sorted(set([p[0] for p in curve1 + curve2]))
        
        # Interpolate y values for both curves at each x coordinate
        merged_curve = []
        for x in x_coords:
            # Find y values for curve1
            y1 = np.interp(x, 
                          [p[0] for p in curve1],
                          [p[1] for p in curve1])
            
            # Find y values for curve2
            y2 = np.interp(x, 
                          [p[0] for p in curve2],
                          [p[1] for p in curve2])
            
            # Merge y values with weighting
            y_merged = y1 * (1 - weight) + y2 * weight
            merged_curve.append((x, y_merged))
        
        return merged_curve
    
    def _apply_direct_curves(self, image, master_lut, red_lut, green_lut, blue_lut):
        """Apply curves directly to RGB channels"""
        
        result = image.clone()
        
        # Convert to 0-255 range for LUT lookup
        image_255 = (image * 255).round().long()
        image_255 = torch.clamp(image_255, 0, 255)
        
        # Apply master curve to all channels
        if not torch.equal(master_lut, torch.linspace(0, 1, 256)):
            for c in range(3):
                result[:, :, :, c] = master_lut[image_255[:, :, :, c]]
        
        # Convert back to 0-255 for individual channel curves
        result_255 = (result * 255).round().long()
        result_255 = torch.clamp(result_255, 0, 255)
        
        # Apply individual channel curves
        if not torch.equal(red_lut, torch.linspace(0, 1, 256)):
            result[:, :, :, 0] = red_lut[result_255[:, :, :, 0]]
        
        if not torch.equal(green_lut, torch.linspace(0, 1, 256)):
            result[:, :, :, 1] = green_lut[result_255[:, :, :, 1]]
        
        if not torch.equal(blue_lut, torch.linspace(0, 1, 256)):
            result[:, :, :, 2] = blue_lut[result_255[:, :, :, 2]]
        
        return result
    
    def _apply_luminance_preserving_curves(self, image, master_lut, red_lut, green_lut, blue_lut):
        """Apply curves while preserving luminance"""
        
        # Calculate original luminance (Rec. 709)
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device)
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
                                base + (2.0 * overlay - 1.0) * (torch.sqrt(base) - base))
        elif mode == "hard_light":
            blended = torch.where(overlay < 0.5,
                                2.0 * base * overlay,
                                1.0 - 2.0 * (1.0 - base) * (1.0 - overlay))
        elif mode == "color_dodge":
            blended = torch.where(overlay >= 0.999, 1.0, 
                                torch.clamp(base / (1.0 - overlay), 0.0, 1.0))
        elif mode == "color_burn":
            blended = torch.where(overlay <= 0.001, 0.0,
                                1.0 - torch.clamp((1.0 - base) / overlay, 0.0, 1.0))
        elif mode == "darken":
            blended = torch.min(base, overlay)
        elif mode == "lighten":
            blended = torch.max(base, overlay)
        else:  # normal
            blended = overlay
        
        # Apply opacity
        return base * (1.0 - opacity) + blended * opacity
    
    def _generate_curve_info(self, master_curve, red_curve, green_curve, blue_curve,
                           blend_mode, opacity, preserve_luminance):
        """Generate human-readable curve information"""
        
        def points_to_str(curve):
            return f"[{', '.join(f'({x:.2f}, {y:.2f})' for x, y in curve)}]"
        
        info_lines = [
            f"RGB Curves Applied:",
            f"Master: {points_to_str(master_curve)}",
            f"Red: {points_to_str(red_curve)}",
            f"Green: {points_to_str(green_curve)}",
            f"Blue: {points_to_str(blue_curve)}",
            f"Blend: {blend_mode} ({opacity:.0%})",
            f"Luminance Preserved: {preserve_luminance}"
        ]
        
        return " | ".join(info_lines)

# 🧪 LOCAL TESTING - Remove before pushing to Git
NODE_CLASS_MAPPINGS = {
    "ApexRGBCurve": ApexRGBCurve
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexRGBCurve": "🎨 Apex RGB Curve"
}