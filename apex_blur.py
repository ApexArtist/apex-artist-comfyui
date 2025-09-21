#!/usr/bin/env python3
"""
Apex Blur Node - Professional blur effects with multiple algorithms
Supports various blur types for different artistic and technical needs
"""
import torch
import torch.nn.functional as F
import numpy as np
import math

class ApexBlur:
    """
    Professional blur node with multiple blur algorithms
    
    Features:
    - Gaussian blur (smooth, natural)
    - Box blur (fast, uniform)
    - Motion blur (directional)
    - Radial blur (zoom effect)
    - Surface blur (edge-preserving)
    - Lens blur (depth of field)
    - Spin blur (rotational)
    - Variable radius control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": ([
                    "gaussian",
                    "box", 
                    "motion",
                    "radial",
                    "surface",
                    "lens",
                    "spin",
                    "zoom"
                ], {"default": "gaussian"}),
                "radius": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("blurred_image", "blur_info")
    FUNCTION = "apply_blur"
    CATEGORY = "ApexArtist/Effects"

    def apply_blur(self, image, blur_type="gaussian", radius=5.0, strength=1.0, 
                   angle=0.0, center_x=0.5, center_y=0.5, edge_threshold=0.1):
        try:
            # Ensure image is in correct format [B, H, W, C]
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # Apply blur based on type
            if blur_type == "gaussian":
                blurred = self._gaussian_blur(image, radius)
            elif blur_type == "box":
                blurred = self._box_blur(image, radius)
            elif blur_type == "motion":
                blurred = self._motion_blur(image, radius, angle)
            elif blur_type == "radial":
                blurred = self._radial_blur(image, radius, center_x, center_y)
            elif blur_type == "surface":
                blurred = self._surface_blur(image, radius, edge_threshold)
            elif blur_type == "lens":
                blurred = self._lens_blur(image, radius, center_x, center_y)
            elif blur_type == "spin":
                blurred = self._spin_blur(image, radius, center_x, center_y)
            elif blur_type == "zoom":
                blurred = self._zoom_blur(image, radius, center_x, center_y)
            else:
                blurred = self._gaussian_blur(image, radius)
            
            # Apply strength blending
            if strength < 1.0:
                blurred = image + strength * (blurred - image)
            
            # Generate blur info
            blur_info = f"Blur: {blur_type} | Radius: {radius:.1f} | Strength: {strength:.2f}"
            if blur_type in ["motion"]:
                blur_info += f" | Angle: {angle:.0f}°"
            elif blur_type in ["radial", "lens", "spin", "zoom"]:
                blur_info += f" | Center: ({center_x:.2f}, {center_y:.2f})"
            
            return (torch.clamp(blurred, 0, 1), blur_info)
            
        except Exception as e:
            print(f"Blur error: {str(e)}")
            return (image, f"Error: {str(e)}")

    def _create_gaussian_kernel(self, radius, device):
        """Create 2D Gaussian kernel"""
        sigma = radius / 3.0
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        
        # Create coordinate grids
        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Calculate Gaussian values
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def _gaussian_blur(self, image, radius):
        """High-quality Gaussian blur"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(radius, device)
        kernel_size = kernel.shape[-1]
        padding = kernel_size // 2
        
        # Reshape image for convolution [B*C, 1, H, W]
        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        
        # Apply convolution with padding
        blurred = F.conv2d(img_reshaped, kernel, padding=padding)
        
        # Reshape back to [B, H, W, C]
        blurred = blurred.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return blurred

    def _box_blur(self, image, radius):
        """Fast box blur using separable convolution"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        kernel_size = int(2 * radius + 1)
        kernel_1d = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        padding = kernel_size // 2
        
        # Reshape for convolution
        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        
        # Horizontal pass
        blurred = F.conv2d(img_reshaped, kernel_1d, padding=(0, padding))
        
        # Vertical pass
        kernel_1d_v = kernel_1d.transpose(-1, -2)
        blurred = F.conv2d(blurred, kernel_1d_v, padding=(padding, 0))
        
        # Reshape back
        blurred = blurred.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return blurred

    def _motion_blur(self, image, radius, angle):
        """Directional motion blur"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Calculate motion vector
        dx = radius * math.cos(angle_rad)
        dy = radius * math.sin(angle_rad)
        
        # Create motion blur kernel
        kernel_size = int(2 * radius + 1)
        kernel = torch.zeros(kernel_size, kernel_size, device=device)
        
        center = kernel_size // 2
        steps = max(1, int(radius))
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            x = int(center + t * dx - dx/2)
            y = int(center + t * dy - dy/2)
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] += 1
        
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        else:
            kernel[center, center] = 1
        
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        
        # Apply convolution
        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        blurred = F.conv2d(img_reshaped, kernel, padding=padding)
        blurred = blurred.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return blurred

    def _radial_blur(self, image, radius, center_x, center_y):
        """Radial/zoom blur effect"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Adjust center
        center_x_adj = (center_x - 0.5) * 2
        center_y_adj = (center_y - 0.5) * 2
        
        xx = xx - center_x_adj
        yy = yy - center_y_adj
        
        # Calculate distance from center
        distance = torch.sqrt(xx**2 + yy**2)
        
        # Create blur by sampling multiple scales
        result = torch.zeros_like(image)
        samples = int(radius) + 1
        
        for i in range(samples):
            scale = 1.0 + (i / samples) * (radius / 10.0)
            
            # Create sampling grid
            grid_x = xx / scale + center_x_adj
            grid_y = yy / scale + center_y_adj
            
            # Normalize to [-1, 1] for grid_sample
            grid_x = grid_x
            grid_y = grid_y
            
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            grid = grid.expand(batch_size, -1, -1, -1)
            
            # Sample image
            img_for_sampling = image.permute(0, 3, 1, 2)
            sampled = F.grid_sample(img_for_sampling, grid, mode='bilinear', 
                                  padding_mode='border', align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1)
            
            result += sampled / samples
        
        return result

    def _surface_blur(self, image, radius, edge_threshold):
        """Edge-preserving surface blur"""
        # Apply Gaussian blur
        blurred = self._gaussian_blur(image, radius)
        
        # Calculate edge weights based on luminance difference
        luma_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)
        original_luma = torch.sum(image * luma_weights, dim=-1, keepdim=True)
        blurred_luma = torch.sum(blurred * luma_weights, dim=-1, keepdim=True)
        
        # Edge detection
        luma_diff = torch.abs(original_luma - blurred_luma)
        edge_weight = torch.exp(-luma_diff / edge_threshold)
        
        # Blend based on edge strength
        result = image + edge_weight * (blurred - image)
        
        return result

    def _lens_blur(self, image, radius, center_x, center_y):
        """Simulate lens blur with depth of field"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create distance map from center
        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate distance from focus point
        distance = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        
        # Create variable blur based on distance
        max_distance = torch.sqrt(2.0)  # Maximum possible distance
        blur_amount = (distance / max_distance) * radius
        
        # Apply varying Gaussian blur
        result = torch.zeros_like(image)
        
        # Sample multiple blur levels
        blur_levels = 5
        for i in range(blur_levels):
            current_radius = (i + 1) * radius / blur_levels
            level_blur = self._gaussian_blur(image, current_radius)
            
            # Create mask for this blur level
            level_min = (i / blur_levels) * radius
            level_max = ((i + 1) / blur_levels) * radius
            
            mask = ((blur_amount >= level_min) & (blur_amount < level_max)).float()
            mask = mask.unsqueeze(-1)
            
            result += level_blur * mask
        
        # Add original for areas with no blur
        no_blur_mask = (blur_amount < radius / blur_levels).float().unsqueeze(-1)
        result += image * no_blur_mask
        
        return result

    def _spin_blur(self, image, radius, center_x, center_y):
        """Rotational spin blur"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Adjust center
        center_x_adj = (center_x - 0.5) * 2
        center_y_adj = (center_y - 0.5) * 2
        
        xx = xx - center_x_adj
        yy = yy - center_y_adj
        
        result = torch.zeros_like(image)
        samples = int(radius) + 1
        
        for i in range(samples):
            angle = (i / samples) * radius * math.pi / 180  # Convert to radians
            
            # Rotation matrix
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Rotate coordinates
            xx_rot = xx * cos_a - yy * sin_a + center_x_adj
            yy_rot = xx * sin_a + yy * cos_a + center_y_adj
            
            grid = torch.stack([xx_rot, yy_rot], dim=-1).unsqueeze(0)
            grid = grid.expand(batch_size, -1, -1, -1)
            
            # Sample image
            img_for_sampling = image.permute(0, 3, 1, 2)
            sampled = F.grid_sample(img_for_sampling, grid, mode='bilinear',
                                  padding_mode='border', align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1)
            
            result += sampled / samples
        
        return result

    def _zoom_blur(self, image, radius, center_x, center_y):
        """Zoom blur effect (different from radial)"""
        device = image.device
        batch_size, height, width, channels = image.shape
        
        result = torch.zeros_like(image)
        samples = int(radius) + 1
        
        for i in range(samples):
            zoom_factor = 1.0 + (i / samples) * (radius / 20.0)
            
            # Calculate zoom transform
            scale = 1.0 / zoom_factor
            
            # Create sampling grid
            y_coords = torch.linspace(-1, 1, height, device=device)
            x_coords = torch.linspace(-1, 1, width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Apply zoom from center
            center_x_adj = (center_x - 0.5) * 2
            center_y_adj = (center_y - 0.5) * 2
            
            xx_zoom = (xx - center_x_adj) * scale + center_x_adj
            yy_zoom = (yy - center_y_adj) * scale + center_y_adj
            
            grid = torch.stack([xx_zoom, yy_zoom], dim=-1).unsqueeze(0)
            grid = grid.expand(batch_size, -1, -1, -1)
            
            # Sample image
            img_for_sampling = image.permute(0, 3, 1, 2)
            sampled = F.grid_sample(img_for_sampling, grid, mode='bilinear',
                                  padding_mode='border', align_corners=False)
            sampled = sampled.permute(0, 2, 3, 1)
            
            result += sampled / samples
        
        return result