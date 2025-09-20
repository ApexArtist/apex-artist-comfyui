"""
Film grain and color science utilities for Apex Artist nodes
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional

class FilmGrain:
    """Film grain simulation with realistic characteristics"""
    
    @staticmethod
    def generate_grain_pattern(height: int, width: int, 
                             characteristics: Dict,
                             device: torch.device) -> torch.Tensor:
        """
        Generate realistic film grain pattern
        Args:
            height: Image height
            width: Image width
            characteristics: Grain characteristics from film profile
            device: Target device for tensor
        """
        # Get grain parameters
        size = characteristics.get("grain_size", 1.5)
        intensity = characteristics.get("grain_intensity", 0.2)
        shadows = characteristics.get("shadow_grain", 1.2)
        highlights = characteristics.get("highlight_grain", 0.8)
        
        # Create base noise
        noise = torch.randn(height, width, device=device)
        
        # Apply grain size using gaussian blur
        if size > 1.0:
            kernel_size = int(2 * round(size) + 1)
            sigma = size / 3.0
            noise = gaussian_blur(noise.unsqueeze(0).unsqueeze(0), 
                                kernel_size, sigma).squeeze()
        
        # Normalize and adjust intensity
        noise = noise * intensity
        
        return noise

    @staticmethod
    def apply_grain(image: torch.Tensor, 
                   grain_pattern: torch.Tensor,
                   characteristics: Dict) -> torch.Tensor:
        """
        Apply grain pattern to image with tone-dependent intensity
        Args:
            image: Input image tensor [B,H,W,C]
            grain_pattern: Generated grain pattern
            characteristics: Grain characteristics
        """
        shadows = characteristics.get("shadow_grain", 1.2)
        highlights = characteristics.get("highlight_grain", 0.8)
        
        # Calculate luminance
        luminance = 0.2989 * image[...,0] + 0.5870 * image[...,1] + 0.1140 * image[...,2]
        
        # Create tone-dependent mask
        shadow_mask = torch.exp(-4 * luminance)
        highlight_mask = torch.exp(-4 * (1 - luminance))
        
        # Adjust grain intensity based on luminance
        grain_intensity = (grain_pattern * shadows * shadow_mask +
                         grain_pattern * highlights * highlight_mask)
        
        # Apply grain to each channel
        result = image.clone()
        for c in range(3):
            result[...,c] = image[...,c] + grain_intensity
        
        return torch.clamp(result, 0, 1)

def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply gaussian blur to tensor"""
    pad_size = kernel_size // 2
    
    # Create 1D Gaussian kernel
    kernel = torch.arange(kernel_size, device=x.device, dtype=torch.float32)
    kernel = kernel - kernel.mean()
    kernel = torch.exp(-kernel.pow(2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    # Separate 2D convolution into two 1D convolutions
    # Horizontal pass
    x = F.pad(x, (pad_size, pad_size, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel.view(1, 1, 1, -1), padding=0)
    
    # Vertical pass
    x = F.pad(x, (0, 0, pad_size, pad_size), mode='reflect')
    x = F.conv2d(x, kernel.view(1, 1, -1, 1), padding=0)
    
    return x

class ColorScience:
    """Advanced color science calculations for film simulation"""
    
    @staticmethod
    def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to XYZ color space"""
        # sRGB to XYZ matrix
        matrix = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device)
        
        # Apply gamma correction
        rgb_linear = torch.where(rgb > 0.04045,
                               ((rgb + 0.055) / 1.055) ** 2.4,
                               rgb / 12.92)
        
        return torch.matmul(rgb_linear, matrix.T)
    
    @staticmethod
    def xyz_to_lab(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert XYZ to LAB color space"""
        # D65 illuminant
        xyz_n = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device)
        
        # Scale by reference white
        xyz_scaled = xyz / xyz_n
        
        # Nonlinear transformation
        f = torch.where(xyz_scaled > 0.008856,
                       xyz_scaled.pow(1/3),
                       7.787 * xyz_scaled + 16/116)
        
        L = 116 * f[...,1] - 16
        a = 500 * (f[...,0] - f[...,1])
        b = 200 * (f[...,1] - f[...,2])
        
        return L, a, b
    
    @staticmethod
    def lab_to_xyz(L: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convert LAB to XYZ color space"""
        # D65 illuminant
        xyz_n = torch.tensor([0.95047, 1.0, 1.08883], device=L.device)
        
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        # Reverse nonlinear transformation
        xyz = torch.stack([fx, fy, fz], dim=-1)
        xyz = torch.where(xyz > 0.206893,
                         xyz.pow(3),
                         (xyz - 16/116) / 7.787)
        
        return xyz * xyz_n
    
    @staticmethod
    def xyz_to_rgb(xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to RGB color space"""
        # XYZ to sRGB matrix
        matrix = torch.tensor([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], device=xyz.device)
        
        rgb_linear = torch.matmul(xyz, matrix.T)
        
        # Apply inverse gamma correction
        rgb = torch.where(rgb_linear > 0.0031308,
                         1.055 * rgb_linear.pow(1/2.4) - 0.055,
                         12.92 * rgb_linear)
        
        return torch.clamp(rgb, 0, 1)
    
    @classmethod
    def apply_color_science(cls, image: torch.Tensor, 
                           characteristics: Dict) -> torch.Tensor:
        """Apply color science adjustments based on film characteristics"""
        # Convert to LAB space for more accurate color manipulation
        xyz = cls.rgb_to_xyz(image)
        L, a, b = cls.xyz_to_lab(xyz)
        
        # Apply color adjustments
        saturation = characteristics.get("saturation", 1.0)
        color_temp = characteristics.get("color_temp", 0.0)
        tint = characteristics.get("tint", 0.0)
        
        # Adjust saturation in a*b* plane
        a = a * saturation
        b = b * saturation
        
        # Adjust color temperature (blue-yellow axis)
        b = b + color_temp * 5
        
        # Adjust tint (green-magenta axis)
        a = a + tint * 5
        
        # Convert back to RGB
        xyz = cls.lab_to_xyz(L, a, b)
        rgb = cls.xyz_to_rgb(xyz)
        
        return rgb