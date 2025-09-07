# apex_depth_to_normal.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class ApexDepthToNormal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 5.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "coordinate_system": (["OpenGL (Blender)", "DirectX"], {
                    "default": "OpenGL (Blender)"
                }),
            },
            "optional": {
                "blur_radius": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 3.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "edge_enhance": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "depth_to_normal"
    CATEGORY = "ApexArtist"
    
    def depth_to_normal(self, depth_image, strength=1.0, coordinate_system="OpenGL (Blender)", 
                       blur_radius=0.0, invert_depth=False, edge_enhance=0.0):
        
        # Ensure we're working with the right tensor format [B, H, W, C]
        if depth_image.dim() == 4 and depth_image.shape[-1] > 1:
            # Convert to grayscale if RGB
            depth = torch.mean(depth_image, dim=-1, keepdim=False)
        else:
            depth = depth_image.squeeze(-1) if depth_image.shape[-1] == 1 else depth_image
        
        # Optional depth inversion
        if invert_depth:
            depth = 1.0 - depth
        
        # Optional smoothing
        if blur_radius > 0:
            kernel_size = max(3, int(blur_radius * 6) | 1)  # Ensure odd number
            depth_smooth = depth.unsqueeze(1)  # Add channel dim for conv
            depth_smooth = F.gaussian_blur(depth_smooth, kernel_size, blur_radius)
            depth = depth_smooth.squeeze(1)
        
        # Calculate gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        
        # Apply Sobel filters
        depth_padded = depth.unsqueeze(1)  # [B, 1, H, W]
        grad_x = F.conv2d(depth_padded, sobel_x, padding=1)
        grad_y = F.conv2d(depth_padded, sobel_y, padding=1)
        
        # Optional edge enhancement
        if edge_enhance > 0:
            edge_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                     dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
            edges = F.conv2d(depth_padded, edge_kernel, padding=1)
            grad_x = grad_x + edges * edge_enhance
            grad_y = grad_y + edges * edge_enhance
        
        # Calculate normal vectors
        normal_x = -grad_x * strength
        normal_y = -grad_y * strength
        normal_z = torch.ones_like(normal_x)
        
        # Handle coordinate system
        if coordinate_system == "DirectX":
            normal_y = -normal_y
        
        # Normalize vectors
        normal_length = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2 + 1e-8)
        normal_x = normal_x / normal_length
        normal_y = normal_y / normal_length  
        normal_z = normal_z / normal_length
        
        # Convert from [-1,1] to [0,1] and create RGB normal map
        normal_x = (normal_x + 1.0) * 0.5
        normal_y = (normal_y + 1.0) * 0.5
        normal_z = (normal_z + 1.0) * 0.5
        
        # Stack channels and remove extra dimensions [B, H, W, 3]
        normal_map = torch.stack([
            normal_x.squeeze(1), 
            normal_y.squeeze(1), 
            normal_z.squeeze(1)
        ], dim=-1)
        
        return (normal_map,)

# For local testing
def test_depth_to_normal(depth_image_path, output_path):
    """
    Test function for local development
    """
    # Load depth image
    img = Image.open(depth_image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    
    # Convert to ComfyUI format [B, H, W, C]
    depth_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(-1)
    
    # Create node instance and process
    node = ApexDepthToNormal()
    normal_map, = node.depth_to_normal(
        depth_image=depth_tensor,
        strength=1.5,
        coordinate_system="OpenGL (Blender)",
        blur_radius=0.5,
        invert_depth=False,
        edge_enhance=0.0
    )
    
    # Convert back to image and save
    normal_np = normal_map[0].numpy()  # Remove batch dimension
    normal_np = (normal_np * 255).astype(np.uint8)
    
    normal_img = Image.fromarray(normal_np, 'RGB')
    normal_img.save(output_path)
    print(f"Normal map saved to: {output_path}")

# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ApexArtist - Depth to Normal": ApexDepthToNormal
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexArtist - Depth to Normal": "Apex Artist - Depth to Normal"
}

# Example usage for local testing
if __name__ == "__main__":
    # Test with a depth image
    # test_depth_to_normal("path/to/your/depth_image.png", "output_normal_map.png")
    print("ApexArtist Depth to Normal node ready!")
    print("For local testing, uncomment and modify the test line above.")