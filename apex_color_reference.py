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

    RETURN_TYPES = ("IMAGE", "CURVE", "CURVE", "CURVE", "CURVE", "STRING")
    RETURN_NAMES = ("matched_image", "master_curve", "red_curve", "green_curve", "blue_curve", "match_info")
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
                curves = self._histogram_matching(target_image, reference_image)
            elif matching_method == "curve_extraction":
                curves = self._extract_curves(target_image, reference_image)
            elif matching_method == "statistical":
                curves = self._statistical_matching(target_image, reference_image)
            elif matching_method == "zone_based":
                curves = self._zone_based_matching(target_image, reference_image, match_zones)
            else:  # selective
                curves = self._selective_matching(target_image, reference_image, match_zones)
            
            # Apply strength factor
            curves = self._adjust_curve_strength(curves, strength)
            
            # Apply curves to image
            matched_image = self._apply_curves(target_image, curves, preserve_luminance)
            
            # Generate match info
            match_info = self._generate_match_info(curves, matching_method, strength)
            
            return (matched_image,) + curves + (match_info,)
            
        except Exception as e:
            print(f"Color matching error: {str(e)}")
            return (target_image, None, None, None, None, str(e))

    def _histogram_matching(self, target, reference):
        """Match histograms between target and reference images"""
        master_curve = []
        rgb_curves = []
        
        # Process each channel
        for c in range(3):
            # Calculate histograms
            target_hist = torch.histc(target[0,:,:,c], bins=256, min=0, max=1)
            ref_hist = torch.histc(reference[0,:,:,c], bins=256, min=0, max=1)
            
            # Calculate cumulative distributions
            target_cdf = torch.cumsum(target_hist, dim=0)
            ref_cdf = torch.cumsum(ref_hist, dim=0)
            
            # Normalize CDFs
            target_cdf = target_cdf / target_cdf[-1]
            ref_cdf = ref_cdf / ref_cdf[-1]
            
            # Create lookup table
            lut = torch.zeros(256)
            for i in range(256):
                target_val = i / 255.0
                # Find closest point in target CDF
                idx = torch.argmin(torch.abs(target_cdf - target_val))
                # Find corresponding value in reference CDF
                matched_val = torch.argmin(torch.abs(ref_cdf - target_cdf[idx])) / 255.0
                lut[i] = matched_val
            
            # Convert LUT to curve points
            curve_points = [(i/255.0, lut[i].item()) for i in range(0, 256, 51)]
            rgb_curves.append(curve_points)
            
            # Contribute to master curve
            if c == 0:  # Initialize master curve
                master_curve = curve_points.copy()
            else:  # Average with existing master curve
                master_curve = [(x, (y + master_curve[i][1])/2) 
                              for i, (x, y) in enumerate(curve_points)]
        
        return (master_curve,) + tuple(rgb_curves)

    def _extract_curves(self, target, reference):
        """Extract tone curves by analyzing image characteristics"""
        curves = []
        
        # Process each channel including luminance
        for c in range(4):  # 3 RGB channels + luminance
            if c < 3:
                target_channel = target[0,:,:,c]
                ref_channel = reference[0,:,:,c]
            else:
                # Calculate luminance
                target_channel = 0.2989 * target[0,:,:,0] + 0.5870 * target[0,:,:,1] + 0.1140 * target[0,:,:,2]
                ref_channel = 0.2989 * reference[0,:,:,0] + 0.5870 * reference[0,:,:,1] + 0.1140 * reference[0,:,:,2]
            
            # Analyze key points in both images
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            target_points = torch.tensor([torch.quantile(target_channel, p/100) for p in percentiles])
            ref_points = torch.tensor([torch.quantile(ref_channel, p/100) for p in percentiles])
            
            # Create curve points
            curve_points = [(t.item(), r.item()) for t, r in zip(target_points, ref_points)]
            curves.append(curve_points)
        
        # Return master curve (luminance) and RGB curves
        return tuple(curves)

    def _statistical_matching(self, target, reference):
        """Match statistical properties between images"""
        curves = []
        
        # Process each channel
        for c in range(3):
            target_channel = target[0,:,:,c]
            ref_channel = reference[0,:,:,c]
            
            # Calculate statistics
            t_mean = torch.mean(target_channel)
            t_std = torch.std(target_channel)
            r_mean = torch.mean(ref_channel)
            r_std = torch.std(ref_channel)
            
            # Create transfer function
            def transfer(x):
                return ((x - t_mean) * (r_std / t_std) + r_mean).clamp(0, 1)
            
            # Create curve points
            points = torch.linspace(0, 1, 5)
            curve_points = [(x.item(), transfer(x).item()) for x in points]
            curves.append(curve_points)
        
        # Create master curve (average of RGB)
        master_points = []
        for i in range(len(curves[0])):
            x = curves[0][i][0]
            y = sum(c[i][1] for c in curves) / 3
            master_points.append((x, y))
        
        return (master_points,) + tuple(curves)

    def _zone_based_matching(self, target, reference, zones):
        """Match specific luminance zones between images"""
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
        
        curves = []
        # Process each channel
        for c in range(3):
            curve_points = [(0,0), (1,1)]  # Start with identity mapping
            
            for zone in active_zones:
                low, high = zone_bounds[zone]
                
                # Get pixels in zone
                target_mask = (target[0,:,:,c] >= low) & (target[0,:,:,c] <= high)
                ref_mask = (reference[0,:,:,c] >= low) & (reference[0,:,:,c] <= high)
                
                if target_mask.any() and ref_mask.any():
                    # Calculate mean values in zone
                    t_mean = target[0,:,:,c][target_mask].mean()
                    r_mean = reference[0,:,:,c][ref_mask].mean()
                    
                    # Add control point
                    curve_points.append((t_mean.item(), r_mean.item()))
            
            # Sort points by x coordinate
            curve_points = sorted(curve_points)
            curves.append(curve_points)
        
        # Create master curve (average of RGB)
        master_points = []
        for i in range(len(curves[0])):
            x = curves[0][i][0]
            y = sum(c[i][1] for c in curves) / 3
            master_points.append((x, y))
        
        return (master_points,) + tuple(curves)

    def _selective_matching(self, target, reference, zones):
        """Selective color matching based on specified zones"""
        # Similar to zone_based but with color-aware matching
        curves = self._zone_based_matching(target, reference, zones)
        
        # Enhance curves based on color relationships
        enhanced_curves = []
        for curve in curves:
            # Add more control points for smoother transition
            x_vals = [p[0] for p in curve]
            y_vals = [p[1] for p in curve]
            
            # Create spline interpolation
            spline = interpolate.PchipInterpolator(x_vals, y_vals)
            
            # Generate smoother curve
            x_new = np.linspace(0, 1, 10)
            y_new = spline(x_new)
            
            # Convert to points
            enhanced_curves.append(list(zip(x_new, y_new)))
        
        return tuple(enhanced_curves)

    def _adjust_curve_strength(self, curves, strength):
        """Adjust the strength of the curves"""
        adjusted_curves = []
        for curve in curves:
            # Adjust each point except endpoints
            adjusted_points = []
            for i, (x, y) in enumerate(curve):
                if i == 0 or i == len(curve)-1:
                    adjusted_points.append((x, y))
                else:
                    # Interpolate between identity (x=y) and target curve
                    adj_y = x + (y - x) * strength
                    adjusted_points.append((x, adj_y))
            adjusted_curves.append(adjusted_points)
        return tuple(adjusted_curves)

    def _apply_curves(self, image, curves, preserve_luminance):
        """Apply the curves to the image"""
        master_curve, r_curve, g_curve, b_curve = curves
        
        # Create lookup tables
        master_lut = self._create_lut(master_curve)
        r_lut = self._create_lut(r_curve)
        g_lut = self._create_lut(g_curve)
        b_lut = self._create_lut(b_curve)
        
        # Apply curves
        result = image.clone()
        
        if preserve_luminance:
            # Calculate original luminance
            luminance = 0.2989 * image[:,:,:,0] + 0.5870 * image[:,:,:,1] + 0.1140 * image[:,:,:,2]
        
        # Apply channel curves
        for c, lut in enumerate([r_lut, g_lut, b_lut]):
            result[:,:,:,c] = self._apply_lut(result[:,:,:,c], lut)
        
        # Apply master curve
        result = self._apply_lut(result, master_lut)
        
        if preserve_luminance:
            # Calculate new luminance
            new_luminance = 0.2989 * result[:,:,:,0] + 0.5870 * result[:,:,:,1] + 0.1140 * result[:,:,:,2]
            
            # Adjust to preserve original luminance
            ratio = torch.where(new_luminance > 0.001, luminance / new_luminance, torch.ones_like(luminance))
            result = result * ratio.unsqueeze(-1)
        
        return result

    def _create_lut(self, curve_points):
        """Create a lookup table from curve points"""
        x = torch.tensor([p[0] for p in curve_points])
        y = torch.tensor([p[1] for p in curve_points])
        
        # Create lookup table
        lut = torch.zeros(256)
        for i in range(256):
            input_val = i / 255.0
            # Find surrounding points
            mask = x <= input_val
            if not mask.any():
                lut[i] = y[0]
            elif mask.all():
                lut[i] = y[-1]
            else:
                idx = torch.where(mask)[0][-1]
                # Linear interpolation
                x1, x2 = x[idx], x[idx + 1]
                y1, y2 = y[idx], y[idx + 1]
                lut[i] = y1 + (input_val - x1) * (y2 - y1) / (x2 - x1)
        
        return lut

    def _apply_lut(self, image, lut):
        """Apply a lookup table to an image"""
        device = image.device
        lut = lut.to(device)
        
        # Scale to 0-255 range
        img_255 = (image * 255).long().clamp(0, 255)
        
        # Apply LUT
        return lut[img_255]

    def _generate_match_info(self, curves, method, strength):
        """Generate information about the color matching"""
        master, r, g, b = curves
        
        info = [
            f"Color Match Method: {method}",
            f"Match Strength: {strength:.2f}",
            f"Curve Points:",
            f"  Master: {len(master)} points",
            f"  Red: {len(r)} points",
            f"  Green: {len(g)} points",
            f"  Blue: {len(b)} points"
        ]
        
        return "\n".join(info)