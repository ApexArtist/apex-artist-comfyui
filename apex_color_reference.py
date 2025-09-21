#!/usr/bin/env python3
"""
Apex Color Reference Node - Automatic color matching and curve generation
based on reference images
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate

class ApexColorReference:
    """
    Analyzes reference images and generates matching curves for color grading
    Features:
    - Color histogram matching
    - Tone curve extraction
    - Channel-specific adjustments
    - Multiple matching methods
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "matching_method": ([
                    "histogram_matching",    # Direct histogram matching
                    "curve_extraction",      # Extract curves from reference
                    "statistical",           # Match statistical properties
                    "zone_based",           # Match by luminance zones
                    "selective"             # Match specific tonal ranges
                ], {"default": "histogram_matching"}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "preserve_luminance": ("BOOLEAN", {
                    "default": True,
                    "label": "Preserve Luminance"
                }),
                "match_zones": ([
                    "all",
                    "shadows_only",
                    "midtones_only",
                    "highlights_only",
                    "shadows_midtones",
                    "midtones_highlights"
                ], {"default": "all"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("matched_image", "match_info")
    FUNCTION = "match_colors"
    CATEGORY = "ApexArtist/Color"

    def match_colors(self, target_image, reference_image, matching_method="histogram_matching",
                    strength=1.0, preserve_luminance=True, match_zones="all"):
        try:
            # Ensure images are in correct format
            if len(target_image.shape) == 3:
                target_image = target_image.unsqueeze(0)
            if len(reference_image.shape) == 3:
                reference_image = reference_image.unsqueeze(0)
            
            # Get histograms and curves based on method
            if matching_method == "histogram_matching":
                matched_image = self._histogram_matching(target_image, reference_image, strength)
            elif matching_method == "curve_extraction":
                matched_image = self._extract_curves(target_image, reference_image, strength)
            elif matching_method == "statistical":
                matched_image = self._statistical_matching(target_image, reference_image, strength)
            elif matching_method == "zone_based":
                matched_image = self._zone_based_matching(target_image, reference_image, match_zones, strength)
            else:  # selective
                matched_image = self._selective_matching(target_image, reference_image, match_zones, strength)
            
            # Apply preserve luminance if requested
            if preserve_luminance:
                matched_image = self._preserve_luminance(target_image, matched_image)
            
            # Generate match info
            match_info = self._generate_match_info(matching_method, strength, preserve_luminance, match_zones)
            
            return (matched_image, match_info)
            
        except Exception as e:
            print(f"Color matching error: {str(e)}")
            return (target_image, f"Error: {str(e)}")

    def _histogram_matching(self, target, reference, strength):
        """Match histograms between target and reference images"""
        device = target.device
        result = target.clone()
        
        # Process each channel
        for c in range(3):
            # Calculate histograms
            target_flat = target[0,:,:,c].flatten()
            ref_flat = reference[0,:,:,c].flatten()
            
            # Get sorted values
            target_sorted = torch.sort(target_flat)[0]
            ref_sorted = torch.sort(ref_flat)[0]
            
            # Create mapping
            n_pixels = target_flat.shape[0]
            ref_n_pixels = ref_flat.shape[0]
            
            # Interpolate reference values to match target distribution
            indices = torch.linspace(0, ref_n_pixels - 1, n_pixels).long()
            indices = torch.clamp(indices, 0, ref_n_pixels - 1)
            mapped_values = ref_sorted[indices]
            
            # Apply mapping with strength
            original_values = target_sorted
            adjusted_values = original_values + strength * (mapped_values - original_values)
            
            # Create lookup table
            lut = torch.zeros(256, device=device, dtype=torch.float32)
            for i in range(n_pixels):
                orig_idx = int(torch.clamp(original_values[i] * 255, 0, 255))
                lut[orig_idx] = adjusted_values[i]
            
            # Apply LUT to channel
            img_255 = (target[0,:,:,c] * 255).long().clamp(0, 255)
            result[0,:,:,c] = lut[img_255]
        
        return torch.clamp(result, 0, 1)

    def _extract_curves(self, target, reference, strength):
        """Extract tone curves by analyzing image characteristics"""
        result = target.clone()
        
        # Process each channel
        for c in range(3):
            target_channel = target[0,:,:,c]
            ref_channel = reference[0,:,:,c]
            
            # Analyze key points in both images
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            target_points = torch.tensor([torch.quantile(target_channel, p/100) for p in percentiles])
            ref_points = torch.tensor([torch.quantile(ref_channel, p/100) for p in percentiles])
            
            # Create curve mapping
            curve_lut = self._create_curve_lut(target_points, ref_points, strength)
            
            # Apply curve
            img_255 = (target_channel * 255).long().clamp(0, 255)
            result[0,:,:,c] = curve_lut[img_255]
        
        return torch.clamp(result, 0, 1)

    def _statistical_matching(self, target, reference, strength):
        """Match statistical properties between images"""
        result = target.clone()
        
        # Process each channel
        for c in range(3):
            target_channel = target[0,:,:,c]
            ref_channel = reference[0,:,:,c]
            
            # Calculate statistics
            t_mean = torch.mean(target_channel)
            t_std = torch.std(target_channel)
            r_mean = torch.mean(ref_channel)
            r_std = torch.std(ref_channel)
            
            # Apply statistical transformation
            normalized = (target_channel - t_mean) / (t_std + 1e-8)
            adjusted = normalized * r_std + r_mean
            
            # Blend with original based on strength
            result[0,:,:,c] = target_channel + strength * (adjusted - target_channel)
        
        return torch.clamp(result, 0, 1)

    def _zone_based_matching(self, target, reference, zones, strength):
        """Match specific luminance zones between images"""
        result = target.clone()
        
        # Calculate luminance
        target_luma = 0.2989 * target[0,:,:,0] + 0.5870 * target[0,:,:,1] + 0.1140 * target[0,:,:,2]
        ref_luma = 0.2989 * reference[0,:,:,0] + 0.5870 * reference[0,:,:,1] + 0.1140 * reference[0,:,:,2]
        
        # Define zone boundaries
        zone_bounds = {
            "shadows": (0.0, 0.3),
            "midtones": (0.3, 0.7),
            "highlights": (0.7, 1.0)
        }
        
        # Select zones to match
        active_zones = []
        if zones == "all":
            active_zones = list(zone_bounds.keys())
        elif zones == "shadows_midtones":
            active_zones = ["shadows", "midtones"]
        elif zones == "midtones_highlights":
            active_zones = ["midtones", "highlights"]
        else:
            active_zones = [zones.replace("_only", "")]
        
        # Apply zone-based matching
        for zone in active_zones:
            low, high = zone_bounds[zone]
            
            # Create mask for zone
            zone_mask = (target_luma >= low) & (target_luma <= high)
            
            if zone_mask.any():
                for c in range(3):
                    target_zone = target[0,:,:,c][zone_mask]
                    ref_zone_mask = (ref_luma >= low) & (ref_luma <= high)
                    
                    if ref_zone_mask.any():
                        ref_zone = reference[0,:,:,c][ref_zone_mask]
                        
                        # Match statistics within zone
                        t_mean = torch.mean(target_zone)
                        r_mean = torch.mean(ref_zone)
                        
                        adjustment = strength * (r_mean - t_mean)
                        result[0,:,:,c][zone_mask] = torch.clamp(
                            target[0,:,:,c][zone_mask] + adjustment, 0, 1
                        )
        
        return result

    def _selective_matching(self, target, reference, zones, strength):
        """Selective color matching based on specified zones"""
        return self._zone_based_matching(target, reference, zones, strength)

    def _create_curve_lut(self, x_points, y_points, strength):
        """Create a lookup table from curve points"""
        device = x_points.device
        
        # Interpolate curve
        x_np = x_points.cpu().numpy()
        y_np = y_points.cpu().numpy()
        
        # Create spline interpolation
        spline = interpolate.PchipInterpolator(x_np, y_np)
        
        # Generate LUT
        lut_x = np.linspace(0, 1, 256)
        lut_y = spline(lut_x)
        
        # Apply strength
        identity = np.linspace(0, 1, 256)
        lut_y = identity + strength * (lut_y - identity)
        
        return torch.tensor(lut_y, device=device, dtype=torch.float32)

    def _preserve_luminance(self, original, adjusted):
        """Preserve original luminance"""
        # Calculate original luminance
        orig_luma = 0.2989 * original[:,:,:,0] + 0.5870 * original[:,:,:,1] + 0.1140 * original[:,:,:,2]
        
        # Calculate adjusted luminance
        adj_luma = 0.2989 * adjusted[:,:,:,0] + 0.5870 * adjusted[:,:,:,1] + 0.1140 * adjusted[:,:,:,2]
        
        # Calculate ratio
        ratio = torch.where(adj_luma > 0.001, orig_luma / adj_luma, torch.ones_like(orig_luma))
        
        # Apply ratio to preserve luminance
        result = adjusted * ratio.unsqueeze(-1)
        
        return torch.clamp(result, 0, 1)

    def _generate_match_info(self, method, strength, preserve_luminance, zones):
        """Generate information about the color matching"""
        info = [
            f"Color Match Method: {method}",
            f"Strength: {strength:.2f}",
            f"Preserve Luminance: {preserve_luminance}",
            f"Target Zones: {zones}"
        ]
        
        return " | ".join(info)