#!/usr/bin/env python3
"""
Apex Color Reference Node - Automatic color matching and curve generation
based on reference images
"""
import torch
import torch.nn.functional as F
import numpy as np

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
        """Optimized histogram matching using vectorized operations"""
        device = target.device
        batch_size = target.shape[0]
        
        # Process all channels at once using vectorized operations
        target_flat = target.view(batch_size, -1, 3)  # [B, H*W, 3]
        ref_flat = reference.view(batch_size, -1, 3)
        
        # Use torch.quantile for faster percentile calculation
        percentiles = torch.linspace(0, 1, 256, device=device)
        
        # Calculate quantiles for all channels simultaneously
        target_quantiles = torch.quantile(target_flat, percentiles.unsqueeze(-1), dim=1)  # [256, B, 3]
        ref_quantiles = torch.quantile(ref_flat, percentiles.unsqueeze(-1), dim=1)
        
        # Interpolate using torch operations (GPU-accelerated)
        target_indices = (target_flat * 255).long().clamp(0, 255)  # [B, H*W, 3]
        
        # Vectorized lookup for all channels
        mapped_values = ref_quantiles[target_indices, 0, :]  # Broadcasting magic
        original_values = target_quantiles[target_indices, 0, :]
        
        # Apply strength blending
        result_flat = original_values + strength * (mapped_values - original_values)
        
        return torch.clamp(result_flat.view_as(target), 0, 1)

    def _extract_curves(self, target, reference, strength):
        """Optimized curve extraction using GPU operations"""
        # Process all channels simultaneously
        percentiles = torch.tensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=target.device)
        
        # Calculate percentiles for all channels at once
        target_flat = target.view(target.shape[0], -1, 3)  # [B, H*W, 3]
        ref_flat = reference.view(reference.shape[0], -1, 3)
        
        target_points = torch.quantile(target_flat, percentiles.unsqueeze(-1), dim=1)  # [7, B, 3]
        ref_points = torch.quantile(ref_flat, percentiles.unsqueeze(-1), dim=1)
        
        # Create smooth curve using linear interpolation (faster than scipy)
        curve_lut = self._create_fast_curve_lut(target_points, ref_points, strength, target.device)
        
        # Apply curve using vectorized indexing
        target_indices = (target * 255).long().clamp(0, 255)
        result = curve_lut[target_indices]
        
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
        """Optimized zone-based matching using vectorized operations"""
        device = target.device
        
        # Pre-calculate luminance for both images (vectorized)
        luma_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 1, 1, 3)
        target_luma = torch.sum(target * luma_weights, dim=-1, keepdim=True)  # [B, H, W, 1]
        ref_luma = torch.sum(reference * luma_weights, dim=-1, keepdim=True)
        
        # Define zone boundaries as tensors for vectorized operations
        zone_bounds = {
            "shadows": (0.0, 0.3),
            "midtones": (0.3, 0.7),
            "highlights": (0.7, 1.0)
        }
        
        # Select active zones
        if zones == "all":
            active_zones = list(zone_bounds.keys())
        elif zones == "shadows_midtones":
            active_zones = ["shadows", "midtones"]
        elif zones == "midtones_highlights":
            active_zones = ["midtones", "highlights"]
        else:
            active_zones = [zones.replace("_only", "")]
        
        result = target.clone()
        
        # Process all zones with vectorized operations
        for zone in active_zones:
            low, high = zone_bounds[zone]
            
            # Create masks (vectorized)
            target_mask = (target_luma >= low) & (target_luma <= high)  # [B, H, W, 1]
            ref_mask = (ref_luma >= low) & (ref_luma <= high)
            
            # Calculate means using masked operations (much faster)
            if target_mask.any() and ref_mask.any():
                # Vectorized mean calculation for all channels
                target_zone_means = torch.sum(target * target_mask, dim=(1,2), keepdim=True) / \
                                  (torch.sum(target_mask, dim=(1,2), keepdim=True) + 1e-8)
                ref_zone_means = torch.sum(reference * ref_mask, dim=(1,2), keepdim=True) / \
                               (torch.sum(ref_mask, dim=(1,2), keepdim=True) + 1e-8)
                
                # Apply adjustment (vectorized)
                adjustment = strength * (ref_zone_means - target_zone_means)
                result = torch.where(target_mask, 
                                   torch.clamp(target + adjustment, 0, 1), 
                                   result)
        
        return result

    def _selective_matching(self, target, reference, zones, strength):
        """Selective color matching based on specified zones"""
        return self._zone_based_matching(target, reference, zones, strength)

    def _create_fast_curve_lut(self, x_points, y_points, strength, device):
        """Create LUT using fast GPU interpolation instead of scipy"""
        # Use torch.lerp for linear interpolation on GPU
        lut_size = 256
        lut_x = torch.linspace(0, 1, lut_size, device=device)
        
        # Simple linear interpolation between points (much faster than spline)
        x_flat = x_points.flatten()
        y_flat = y_points.flatten()
        
        # Use torch.searchsorted for fast interpolation
        indices = torch.searchsorted(x_flat, lut_x)
        indices = torch.clamp(indices, 1, len(x_flat) - 1)
        
        # Linear interpolation
        x0 = x_flat[indices - 1]
        x1 = x_flat[indices]
        y0 = y_flat[indices - 1]
        y1 = y_flat[indices]
        
        # Avoid division by zero
        weight = torch.where(x1 != x0, (lut_x - x0) / (x1 - x0), torch.zeros_like(lut_x))
        lut_y = y0 + weight * (y1 - y0)
        
        # Apply strength blending
        identity = torch.linspace(0, 1, lut_size, device=device)
        lut_y = identity + strength * (lut_y - identity)
        
        return lut_y.view(lut_size, 1, 1, 1)  # Shape for broadcasting

    def _create_curve_lut(self, x_points, y_points, strength):
        """Legacy function - replaced by fast version"""
        return self._create_fast_curve_lut(x_points, y_points, strength, x_points.device)

    def _preserve_luminance(self, original, adjusted):
        """Optimized luminance preservation using vectorized operations"""
        # Pre-compute luminance weights
        luma_weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                  device=original.device).view(1, 1, 1, 3)
        
        # Vectorized luminance calculation
        orig_luma = torch.sum(original * luma_weights, dim=-1, keepdim=True)
        adj_luma = torch.sum(adjusted * luma_weights, dim=-1, keepdim=True)
        
        # Safe division with epsilon
        ratio = torch.where(adj_luma > 1e-6, 
                          orig_luma / adj_luma, 
                          torch.ones_like(orig_luma))
        
        # Apply ratio (broadcasting automatically handles dimensions)
        result = adjusted * ratio
        
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