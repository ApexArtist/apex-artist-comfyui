#!/usr/bin/env python3
"""
Apex Film Grading Node - Professional color grading and film effects
Combines DaVinci Resolve-style color grading with authentic film look effects
"""
import torch
import torch.nn.functional as F
import math

class ApexFilmGrading:
    """
    Professional film grading and color correction node
    Features DaVinci-style color wheels, film emulation, and cinematic effects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "film_preset": ([
                    "None",
                    "Kodak Vision3 250D", 
                    "Kodak Vision3 500T",
                    "Fuji Eterna 400T",
                    "Fuji Pro 400H",
                    "Kodak Portra 400",
                    "Kodak Gold 200",
                    "Agfa Vista 200",
                    "Cinestill 800T",
                    "Lomography Color 100",
                    "Vintage 70s",
                    "Modern Digital",
                    "Bleach Bypass",
                    "Teal & Orange",
                    "Film Noir"
                ], {"default": "None"}),
            },
            "optional": {
                # Color Grading Controls
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "whites": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "blacks": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 5.0}),
                "tint": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 5.0}),
                
                # Film Effects
                "bloom_intensity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "halation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "grain_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "vignette_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "vignette_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "chromatic_aberration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "film_curve_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("graded_image", "grading_info")
    FUNCTION = "apply_film_grading"
    CATEGORY = "ApexArtist/Color"
    
    def apply_film_grading(self, image, film_preset="None", exposure=0.0, contrast=1.0, 
                          highlights=0.0, shadows=0.0, whites=0.0, blacks=0.0,
                          saturation=1.0, vibrance=0.0, temperature=0.0, tint=0.0,
                          bloom_intensity=0.0, halation_strength=0.0, grain_amount=0.0,
                          grain_size=1.0, vignette_strength=0.0, vignette_size=1.0,
                          chromatic_aberration=0.0, film_curve_strength=1.0, mask=None):
        
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Apply film preset first
        result = self._apply_film_preset(image, film_preset, film_curve_strength)
        
        # Color grading pipeline
        result = self._apply_exposure(result, exposure)
        result = self._apply_white_balance(result, temperature, tint)
        result = self._apply_tone_curve(result, highlights, shadows, whites, blacks, contrast)
        result = self._apply_saturation_vibrance(result, saturation, vibrance)
        
        # Film effects pipeline
        if bloom_intensity > 0:
            result = self._apply_bloom(result, bloom_intensity)
        
        if halation_strength > 0:
            result = self._apply_halation(result, halation_strength)
        
        if chromatic_aberration > 0:
            result = self._apply_chromatic_aberration(result, chromatic_aberration)
        
        if vignette_strength > 0:
            result = self._apply_vignette(result, vignette_strength, vignette_size)
        
        if grain_amount > 0:
            result = self._apply_film_grain(result, grain_amount, grain_size)
        
        # Generate grading info
        preset_info = f"Preset: {film_preset}" if film_preset != "None" else "No Preset"
        effects_info = []
        if bloom_intensity > 0: effects_info.append(f"Bloom: {bloom_intensity:.1f}")
        if halation_strength > 0: effects_info.append(f"Halation: {halation_strength:.1f}")
        if grain_amount > 0: effects_info.append(f"Grain: {grain_amount:.1f}")
        if vignette_strength > 0: effects_info.append(f"Vignette: {vignette_strength:.1f}")
        
        effects_str = " | ".join(effects_info) if effects_info else "No Effects"
        grading_info = f"{preset_info} | {effects_str}"
        
        # Apply mask if provided
        if mask is not None:
            result = self._apply_mask(image, result, mask)
        
        return (torch.clamp(result, 0, 1), grading_info)
    
    def _apply_film_preset(self, image, preset, strength):
        """Apply film emulation presets"""
        if preset == "None" or strength == 0:
            return image
        
        # Film preset definitions with color matrices and curves
        presets = {
            "Kodak Vision3 250D": {
                "matrix": [1.05, -0.02, -0.03, -0.01, 1.02, -0.01, 0.02, -0.05, 1.03],
                "shadows": (1.02, 0.98, 0.95), "mids": (1.0, 1.0, 1.0), "highlights": (0.98, 1.01, 1.02),
                "saturation": 1.1, "contrast": 1.05
            },
            "Kodak Vision3 500T": {
                "matrix": [1.03, 0.01, -0.04, 0.02, 1.00, -0.02, -0.01, -0.03, 1.04],
                "shadows": (1.01, 0.99, 0.96), "mids": (1.0, 1.0, 1.0), "highlights": (0.99, 1.00, 1.01),
                "saturation": 1.08, "contrast": 1.03
            },
            "Fuji Eterna 400T": {
                "matrix": [0.98, 0.02, 0.00, 0.01, 1.01, -0.02, 0.01, -0.02, 1.01],
                "shadows": (1.00, 0.98, 0.97), "mids": (1.0, 1.0, 1.0), "highlights": (1.00, 1.02, 1.01),
                "saturation": 0.95, "contrast": 0.98
            },
            "Fuji Pro 400H": {
                "matrix": [1.02, 0.01, -0.01, 0.00, 1.03, 0.01, 0.02, -0.01, 0.99],
                "shadows": (1.01, 0.99, 0.98), "mids": (1.0, 1.0, 1.0), "highlights": (0.99, 1.01, 1.00),
                "saturation": 1.15, "contrast": 1.02
            },
            "Kodak Portra 400": {
                "matrix": [1.04, 0.00, -0.02, 0.01, 1.02, 0.01, 0.03, 0.00, 0.98],
                "shadows": (1.02, 0.99, 0.97), "mids": (1.0, 1.0, 1.0), "highlights": (0.98, 1.01, 1.02),
                "saturation": 1.12, "contrast": 1.04
            },
            "Kodak Gold 200": {
                "matrix": [1.08, 0.02, -0.01, 0.03, 1.05, 0.02, 0.05, 0.01, 0.96],
                "shadows": (1.05, 1.00, 0.95), "mids": (1.0, 1.0, 1.0), "highlights": (0.95, 1.02, 1.05),
                "saturation": 1.25, "contrast": 1.08
            },
            "Teal & Orange": {
                "matrix": [1.1, -0.1, 0.0, -0.05, 1.05, -0.05, -0.15, 0.1, 1.2],
                "shadows": (0.9, 1.0, 1.1), "mids": (1.0, 1.0, 1.0), "highlights": (1.1, 1.0, 0.9),
                "saturation": 1.3, "contrast": 1.1
            },
            "Film Noir": {
                "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "shadows": (0.95, 0.95, 0.95), "mids": (1.0, 1.0, 1.0), "highlights": (1.05, 1.05, 1.05),
                "saturation": 0.3, "contrast": 1.4
            }
        }
        
        if preset not in presets:
            return image
        
        preset_data = presets[preset]
        
        # Apply color matrix
        result = self._apply_color_matrix(image, preset_data["matrix"], strength)
        
        # Apply tone curves for shadows, mids, highlights
        result = self._apply_film_tone_curve(result, preset_data, strength)
        
        # Apply saturation and contrast
        sat_factor = 1.0 + (preset_data["saturation"] - 1.0) * strength
        contrast_factor = 1.0 + (preset_data["contrast"] - 1.0) * strength
        
        result = self._adjust_saturation(result, sat_factor)
        result = self._adjust_contrast(result, contrast_factor)
        
        return result
    
    def _apply_color_matrix(self, image, matrix, strength):
        """Apply 3x3 color transformation matrix"""
        # Interpolate matrix with identity
        identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        interp_matrix = []
        for i in range(9):
            interp_matrix.append(identity[i] + (matrix[i] - identity[i]) * strength)
        
        # Reshape image for matrix multiplication
        batch_size, height, width, channels = image.shape
        img_flat = image.view(-1, 3)  # [B*H*W, 3]
        
        # Create matrix tensor
        color_matrix = torch.tensor([
            [interp_matrix[0], interp_matrix[1], interp_matrix[2]],
            [interp_matrix[3], interp_matrix[4], interp_matrix[5]],
            [interp_matrix[6], interp_matrix[7], interp_matrix[8]]
        ], device=image.device, dtype=image.dtype)
        
        # Apply matrix multiplication
        result_flat = torch.matmul(img_flat, color_matrix.t())
        
        return result_flat.view(batch_size, height, width, channels)
    
    def _apply_film_tone_curve(self, image, preset_data, strength):
        """Apply film-specific tone curves for shadows, mids, highlights"""
        shadows_mult = torch.tensor(preset_data["shadows"], device=image.device)
        mids_mult = torch.tensor(preset_data["mids"], device=image.device)
        highlights_mult = torch.tensor(preset_data["highlights"], device=image.device)
        
        # Interpolate with strength
        shadows_mult = 1.0 + (shadows_mult - 1.0) * strength
        mids_mult = 1.0 + (mids_mult - 1.0) * strength
        highlights_mult = 1.0 + (highlights_mult - 1.0) * strength
        
        # Create tone masks
        shadows_mask = torch.clamp(1.0 - image * 2.0, 0, 1) ** 2
        highlights_mask = torch.clamp(image * 2.0 - 1.0, 0, 1) ** 2
        mids_mask = 1.0 - shadows_mask - highlights_mask
        
        # Apply tone-specific adjustments
        result = image * (
            shadows_mask * shadows_mult.view(1, 1, 1, 3) +
            mids_mask * mids_mult.view(1, 1, 1, 3) +
            highlights_mask * highlights_mult.view(1, 1, 1, 3)
        )
        
        return result
    
    def _apply_exposure(self, image, exposure):
        """Apply exposure adjustment"""
        if exposure == 0:
            return image
        return image * (2 ** exposure)
    
    def _apply_white_balance(self, image, temperature, tint):
        """Apply white balance adjustment"""
        if temperature == 0 and tint == 0:
            return image
        
        # Convert temperature/tint to RGB multipliers
        temp_factor = temperature / 100.0
        tint_factor = tint / 100.0
        
        # Temperature: blue <-> yellow
        r_temp = 1.0 + temp_factor * 0.3
        b_temp = 1.0 - temp_factor * 0.3
        
        # Tint: green <-> magenta
        g_tint = 1.0 + tint_factor * 0.3
        
        wb_multiplier = torch.tensor([r_temp, g_tint, b_temp], device=image.device)
        return image * wb_multiplier.view(1, 1, 1, 3)
    
    def _apply_tone_curve(self, image, highlights, shadows, whites, blacks, contrast):
        """Apply tone curve adjustments"""
        result = image
        
        # Shadows and highlights
        if shadows != 0:
            shadow_mask = torch.clamp(1.0 - image * 2.0, 0, 1)
            result = result + shadows * shadow_mask * 0.5
        
        if highlights != 0:
            highlight_mask = torch.clamp(image * 2.0 - 1.0, 0, 1)
            result = result + highlights * highlight_mask * 0.5
        
        # Whites and blacks
        if whites != 0:
            white_mask = image ** 2
            result = result + whites * white_mask * 0.3
        
        if blacks != 0:
            black_mask = (1.0 - image) ** 2
            result = result + blacks * black_mask * 0.3
        
        # Contrast
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5
        
        return result
    
    def _apply_saturation_vibrance(self, image, saturation, vibrance):
        """Apply saturation and vibrance adjustments"""
        if saturation == 1.0 and vibrance == 0:
            return image
        
        # Convert to HSV for saturation
        result = image
        
        if saturation != 1.0:
            result = self._adjust_saturation(result, saturation)
        
        if vibrance != 0:
            result = self._apply_vibrance(result, vibrance)
        
        return result
    
    def _adjust_saturation(self, image, factor):
        """Adjust color saturation"""
        # Calculate luminance
        luminance = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        # Interpolate between grayscale and color
        return luminance + factor * (image - luminance)
    
    def _adjust_contrast(self, image, factor):
        """Adjust contrast"""
        return (image - 0.5) * factor + 0.5
    
    def _apply_vibrance(self, image, vibrance):
        """Apply vibrance (smart saturation that protects skin tones)"""
        # Calculate current saturation level
        max_rgb = torch.max(image, dim=-1, keepdim=True)[0]
        min_rgb = torch.min(image, dim=-1, keepdim=True)[0]
        current_sat = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        
        # Vibrance effect is stronger on less saturated areas
        vibrance_mask = 1.0 - current_sat
        effective_vibrance = vibrance * vibrance_mask
        
        return self._adjust_saturation(image, 1.0 + effective_vibrance)
    
    def _apply_bloom(self, image, intensity):
        """Apply bloom effect"""
        device = image.device
        
        # Create bloom by blurring bright areas
        bright_mask = torch.clamp(image - 0.8, 0, 1) / 0.2
        bright_areas = image * bright_mask
        
        # Gaussian blur for bloom using 2D kernel
        kernel_size = 15
        sigma = 5.0
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        batch_size, height, width, channels = bright_areas.shape
        bright_reshaped = bright_areas.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        
        padding = kernel_size // 2
        bloom = F.conv2d(bright_reshaped, kernel_2d, padding=padding)
        
        # Ensure output matches input size
        if bloom.shape[2:] != (height, width):
            bloom = F.interpolate(bloom, size=(height, width), mode='bilinear', align_corners=False)
        
        bloom = bloom.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return image + bloom * intensity
    
    def _apply_halation(self, image, strength):
        """Apply halation effect (film characteristic)"""
        device = image.device
        
        # Halation is a red glow around bright areas
        bright_mask = torch.clamp(image - 0.7, 0, 1) / 0.3
        
        # Create red halation
        red_channel = image[..., 0:1] * bright_mask[..., 0:1]
        
        # Blur the red channel using 2D Gaussian
        kernel_size = 21
        sigma = 8.0
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        batch_size, height, width, _ = red_channel.shape
        red_reshaped = red_channel.permute(0, 3, 1, 2)
        
        padding = kernel_size // 2
        halation = F.conv2d(red_reshaped, kernel_2d, padding=padding)
        
        # Ensure output matches input size
        if halation.shape[2:] != (height, width):
            halation = F.interpolate(halation, size=(height, width), mode='bilinear', align_corners=False)
        
        halation = halation.permute(0, 2, 3, 1)
        
        # Add red halation to image
        result = image.clone()
        result[..., 0:1] = result[..., 0:1] + halation * strength * 0.3
        
        return result
    
    def _apply_chromatic_aberration(self, image, strength):
        """Apply chromatic aberration effect"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create slight offsets for R and B channels
        offset = int(strength * 3)
        if offset == 0:
            return image
        
        result = image.clone()
        
        # Shift red channel slightly
        if offset > 0:
            result[..., :-offset, :, 0] = image[..., offset:, :, 0]
            result[..., -offset:, :, 0] = image[..., -offset:, :, 0]
        
        # Shift blue channel opposite direction
        if offset > 0:
            result[..., offset:, :, 2] = image[..., :-offset, :, 2]
            result[..., :offset, :, 2] = image[..., :offset, :, 2]
        
        return result
    
    def _apply_vignette(self, image, strength, size):
        """Apply vignette effect"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create radial gradient
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate distance from center
        distance = torch.sqrt(xx**2 + yy**2) / size
        
        # Create vignette mask
        vignette_mask = 1.0 - torch.clamp(distance - 0.5, 0, 1) * strength
        vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(-1)
        
        return image * vignette_mask
    
    def _apply_film_grain(self, image, amount, grain_size):
        """Apply film grain effect"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Generate noise at different scale
        noise_height = int(height / grain_size)
        noise_width = int(width / grain_size)
        
        # Generate grain noise
        grain = torch.randn(batch_size, noise_height, noise_width, 1, device=device) * 0.1
        
        # Upscale grain to image size
        grain = F.interpolate(
            grain.permute(0, 3, 1, 2), 
            size=(height, width), 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        # Apply grain with luminance-based intensity
        luminance = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        grain_intensity = amount * (1.0 - torch.abs(luminance - 0.5) * 2.0)  # Stronger in mid-tones
        
        return image + grain * grain_intensity
    
    def _apply_mask(self, original, processed, mask):
        """Apply mask to blend original and processed images"""
        device = original.device
        batch, height, width, channels = original.shape
        
        # Ensure mask has correct dimensions
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(-1)
        
        # Resize mask if needed
        if mask.shape[1:3] != (height, width):
            mask = F.interpolate(mask.permute(0, 3, 1, 2), size=(height, width), 
                               mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        # Ensure mask is in range [0, 1]
        mask = torch.clamp(mask, 0, 1)
        
        # Apply mask: masked areas get processed image, unmasked areas keep original
        return original * (1 - mask) + processed * mask