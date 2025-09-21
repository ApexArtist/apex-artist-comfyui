#!/usr/bin/env python3#!/usr/bin/env python3#!/usr/bin/env python3

"""

Apex Depth to Normal Node - Professional depth map to normal map conversion""""""

Specialized for converting depth maps (DepthAnything, MiDaS, etc.) to high-quality normal maps

"""Apex Depth to Normal Node - Professional depth map to normal map conversionApex Image to Normal Node - Convert any image to a normal map

import torch

import torch.nn.functional as FSpecialized for converting depth maps (DepthAnything, MiDaS, etc.) to high-quality normal mapsSimple and effective normal map generation from height/displacement data



class ApexDepthToNormal:""""""

    """

    Professional depth map to normal map converterimport torchimport torch

    Optimized specifically for depth maps from:

    - DepthAnything v2import torch.nn.functional as Fimport torch.nn.functional as F

    - MiDaS

    - Other depth estimation models

    """

    class ApexDepthToNormal:class ApexImageToNormal:

    @classmethod

    def INPUT_TYPES(cls):    """    """

        return {

            "required": {    Professional depth map to normal map converter    Simple image to normal map converter

                "depth_map": ("IMAGE",),

                "strength": ("FLOAT", {    Optimized specifically for depth maps from:    Takes any image (RGB, grayscale, height map) and generates a normal map

                    "default": 12.0, 

                    "min": 0.1,     - DepthAnything v2    """

                    "max": 30.0, 

                    "step": 0.1    - MiDaS    

                }),

            },    - Other depth estimation models    @classmethod

            "optional": {

                "invert": ("BOOLEAN", {"default": False}),    """    def INPUT_TYPES(cls):

                "auto_invert_depth": ("BOOLEAN", {"default": False}),

                "blur": ("FLOAT", {            return {

                    "default": 0.0, 

                    "min": 0.0,     @classmethod

                    "max": 3.0,     def INPUT_TYPES(cls):

                    "step": 0.1        return {

                }),            "required": {

                "enhance_details": ("FLOAT", {                "depth_map": ("IMAGE",),

                    "default": 0.0,                "strength": ("FLOAT", {

                    "min": 0.0,                    "default": 12.0, 

                    "max": 3.0,                    "min": 0.1, 

                    "step": 0.1                    "max": 30.0, 

                }),                    "step": 0.1

            }                }),

        }            },

                "optional": {

    RETURN_TYPES = ("IMAGE", "STRING")                "invert": ("BOOLEAN", {"default": False}),

    RETURN_NAMES = ("normal_map", "info")                "auto_invert_depth": ("BOOLEAN", {"default": False}),

    FUNCTION = "depth_to_normal"                "blur": ("FLOAT", {

    CATEGORY = "ApexArtist/Compositing"                    "default": 0.0, 

                        "min": 0.0, 

    def depth_to_normal(self, depth_map, strength=12.0, invert=False, auto_invert_depth=False, blur=0.0, enhance_details=0.0):                    "max": 3.0, 

        print("ApexDepthToNormal: Processing depth map")                    "step": 0.1

                        }),

        # Convert to grayscale if RGB                "enhance_details": ("FLOAT", {

        if depth_map.shape[-1] == 3:                    "default": 0.0,

            # Use luminance formula                    "min": 0.0,

            gray = 0.299 * depth_map[..., 0:1] + 0.587 * depth_map[..., 1:2] + 0.114 * depth_map[..., 2:3]                    "max": 3.0,

        else:                    "step": 0.1

            gray = depth_map                }),

                    }

        # Smart depth map inversion detection        } 

        should_invert = invert

        if auto_invert_depth and not invert:                "invert": ("BOOLEAN", {"default": False}),                    "min": 0.0, 

            # Check if this looks like an inverted depth map

            h, w = gray.shape[1], gray.shape[2]                "auto_invert_depth": ("BOOLEAN", {"default": False}),                    "max": 2.0, 

            center_region = gray[:, h//4:3*h//4, w//4:3*w//4, :]

                            "blur": ("FLOAT", {                    "step": 0.1

            # Sample edge regions

            edge_samples = []                    "default": 0.0,                 }),

            edge_samples.append(gray[:, :h//8, :, :])  # top edge

            edge_samples.append(gray[:, -h//8:, :, :])  # bottom edge                    "min": 0.0,             }

            edge_samples.append(gray[:, :, :w//8, :])  # left edge

            edge_samples.append(gray[:, :, -w//8:, :])  # right edge                    "max": 3.0,         }

            

            center_mean = torch.mean(center_region)                    "step": 0.1    

            edge_mean = torch.mean(torch.stack([torch.mean(edge) for edge in edge_samples]))

                            }),    RETURN_TYPES = ("IMAGE",)

            # If center is significantly darker than edges, likely inverted depth

            if center_mean < edge_mean - 0.2:                "enhance_details": ("FLOAT", {    RETURN_NAMES = ("normal_map",)

                should_invert = True

                print("ApexDepthToNormal: Auto-detected inverted depth map, applying inversion")                    "default": 0.0,    FUNCTION = "image_to_normal"

        

        # Invert if requested or auto-detected                    "min": 0.0,    CATEGORY = "ApexArtist/Compositing"

        if should_invert:

            gray = 1.0 - gray                    "max": 3.0,    

        

        # Apply depth-specific preprocessing                    "step": 0.1    def image_to_normal(self, image, strength=1.0, invert=False, blur=0.0):

        gray = self._preprocess_depth(gray, enhance_details)

                        }),        # Convert to grayscale if RGB

        # Apply blur if requested

        if blur > 0.0:            }        if image.shape[-1] == 3:

            gray = self._apply_blur(gray, blur)

                }            # Use luminance formula

        # Calculate gradients optimized for depth maps

        dx, dy = self._calculate_gradients(gray, strength)                gray = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]

        

        # Create normal map    RETURN_TYPES = ("IMAGE", "STRING")        else:

        normal_map = self._create_normal_map(dx, dy)

            RETURN_NAMES = ("normal_map", "info")            gray = image

        # Generate info about processing

        invert_status = ""    FUNCTION = "depth_to_normal"        

        if should_invert and auto_invert_depth and not invert:

            invert_status = " | Auto-Inverted"    CATEGORY = "ApexArtist/Compositing"        # Invert if requested

        elif should_invert:

            invert_status = " | Inverted"            if invert:

            

        info = f"Depth to Normal | Strength: {strength:.1f} | Details: {enhance_details:.1f}{invert_status}"    def depth_to_normal(self, depth_map, strength=12.0, invert=False, auto_invert_depth=False, blur=0.0, enhance_details=0.0):            gray = 1.0 - gray

        

        return (normal_map, info)        print("ApexDepthToNormal: Processing depth map")        

    

    def _preprocess_depth(self, gray, enhance_details):                # Apply blur if requested

        """Specialized preprocessing for depth maps"""

        # Histogram stretching for better dynamic range        # Convert to grayscale if RGB        if blur > 0.0:

        min_val = torch.min(gray)

        max_val = torch.max(gray)        if depth_map.shape[-1] == 3:            gray = self._apply_blur(gray, blur)

        range_val = max_val - min_val + 1e-8

        stretched = (gray - min_val) / range_val            # Use luminance formula        

        

        # Apply adaptive histogram equalization if details enhancement is requested            gray = 0.299 * depth_map[..., 0:1] + 0.587 * depth_map[..., 1:2] + 0.114 * depth_map[..., 2:3]        # Calculate gradients using Sobel operators

        if enhance_details > 0:

            kernel_size = 15        else:        dx, dy = self._calculate_gradients(gray, strength)

            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=gray.device) / (kernel_size**2)

                        gray = depth_map        

            padded_input = stretched.permute(0, 3, 1, 2)

            local_mean = F.conv2d(padded_input, kernel, padding=kernel_size//2)                # Create normal map

            local_mean = local_mean.permute(0, 2, 3, 1)

                    # Smart depth map inversion detection        normal_map = self._create_normal_map(dx, dy)

            # Enhance details relative to local mean

            enhanced = stretched + enhance_details * 0.3 * (stretched - local_mean)        should_invert = invert        

            enhanced = torch.clamp(enhanced, 0, 1)

        else:        if auto_invert_depth and not invert:        return (normal_map,)

            enhanced = stretched

                    # Check if this looks like an inverted depth map    

        return enhanced

                # Standard depth maps: close objects = bright (white), far objects = dark (black)    def _apply_blur(self, image, blur_amount):

    def _apply_blur(self, image, blur_amount):

        """Apply Gaussian blur"""            # Inverted depth maps: close objects = dark (black), far objects = bright (white)        """Apply Gaussian blur"""

        device = image.device

        batch, height, width, channels = image.shape                    device = image.device

        

        # Create Gaussian kernel            # Sample center region (where main subject usually is)        batch, height, width, channels = image.shape

        kernel_size = int(blur_amount * 4) * 2 + 1

        sigma = blur_amount            h, w = gray.shape[1], gray.shape[2]        

        

        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2            center_region = gray[:, h//4:3*h//4, w//4:3*w//4, :]        # Create Gaussian kernel

        kernel = torch.exp(-(x**2) / (2 * sigma**2))

        kernel = kernel / kernel.sum()                    kernel_size = int(blur_amount * 4) * 2 + 1

        

        # Use standard 2D Gaussian kernel            # Sample edge regions (usually background/far objects)        sigma = blur_amount

        kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)

        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)            edge_samples = []        

        

        # Reshape for convolution            edge_samples.append(gray[:, :h//8, :, :])  # top edge        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2

        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)

                    edge_samples.append(gray[:, -h//8:, :, :])  # bottom edge        kernel = torch.exp(-(x**2) / (2 * sigma**2))

        # Apply 2D convolution with proper padding

        padding = kernel_size // 2            edge_samples.append(gray[:, :, :w//8, :])  # left edge        kernel = kernel / kernel.sum()

        blurred = F.conv2d(img_reshaped, kernel_2d, padding=padding)

                    edge_samples.append(gray[:, :, -w//8:, :])  # right edge        kernel = kernel.view(1, 1, kernel_size)

        return blurred.reshape(batch, channels, height, width).permute(0, 2, 3, 1)

                        

    def _calculate_gradients(self, image, strength):

        """Calculate gradients optimized for depth maps"""            center_mean = torch.mean(center_region)        # Reshape for convolution

        device = image.device

                    edge_mean = torch.mean(torch.stack([torch.mean(edge) for edge in edge_samples]))        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)

        # Enhanced 3x3 Sobel for depth maps

        sobel_x = torch.tensor([                    

            [-1, 0, 1],

            [-2, 0, 2],             # If center is significantly darker than edges, likely inverted depth        # Apply separable convolution

            [-1, 0, 1]

        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0            # (because main subject should typically be closer/brighter in normal depth)        padding = kernel_size // 2

        

        sobel_y = torch.tensor([            if center_mean < edge_mean - 0.2:        blurred = F.conv2d(img_reshaped, kernel, padding=(0, padding))

            [-1, -2, -1],

            [0, 0, 0],                should_invert = True        blurred = F.conv2d(blurred, kernel.transpose(-1, -2), padding=(padding, 0))

            [1, 2, 1]

        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0                print("ApexDepthToNormal: Auto-detected inverted depth map, applying inversion")        

        

        padding = 1                return blurred.reshape(batch, channels, height, width).permute(0, 2, 3, 1)

        strength_multiplier = 3.0  # Higher strength for depth maps

                # Invert if requested or auto-detected    

        # Reshape for convolution

        batch, height, width, channels = image.shape        if should_invert:    def _calculate_gradients(self, image, strength):

        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)

                    gray = 1.0 - gray        """Calculate image gradients using Sobel operators"""

        # Calculate gradients with enhanced strength

        dx = F.conv2d(img_reshaped, sobel_x, padding=padding) * strength * strength_multiplier                device = image.device

        dy = F.conv2d(img_reshaped, sobel_y, padding=padding) * strength * strength_multiplier

                # Apply depth-specific preprocessing        

        # Enhance gradients with adaptive scaling

        gradient_magnitude = torch.sqrt(dx**2 + dy**2)        gray = self._preprocess_depth(gray, enhance_details)        # Sobel kernels

        max_gradient = torch.max(gradient_magnitude) + 1e-8

                        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 

        # Enhance weaker gradients more to reveal subtle surface details

        enhancement_factor = 1.0 + (1.0 - gradient_magnitude / max_gradient) * 0.5        # Apply blur if requested                              device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        dx = dx * enhancement_factor

        dy = dy * enhancement_factor        if blur > 0.0:        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 

        

        # Reshape back            gray = self._apply_blur(gray, blur)                              device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        dx = dx.reshape(batch, channels, height, width).permute(0, 2, 3, 1)

        dy = dy.reshape(batch, channels, height, width).permute(0, 2, 3, 1)                

        

        return dx, dy        # Calculate gradients optimized for depth maps        # Reshape for convolution

    

    def _create_normal_map(self, dx, dy):        dx, dy = self._calculate_gradients(gray, strength)        batch, height, width, channels = image.shape

        """Create RGB normal map from gradients"""

        # Calculate Z component with proper scaling for surface detail                img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)

        z_base = 0.3  # Lower base Z for better surface definition

                # Create normal map        

        # Calculate gradient magnitude for adaptive Z scaling

        gradient_magnitude = torch.sqrt(dx**2 + dy**2)        normal_map = self._create_normal_map(dx, dy)        # Apply convolution

        max_gradient = torch.max(gradient_magnitude) + 1e-8

        normalized_gradient = gradient_magnitude / max_gradient                dx = F.conv2d(img_reshaped, sobel_x, padding=1) * strength

        

        # Inverse relationship: stronger gradients = lower Z        # Generate info about processing        dy = F.conv2d(img_reshaped, sobel_y, padding=1) * strength

        adaptive_z = z_base * (2.0 - normalized_gradient)

                invert_status = ""        

        # Normalize the normal vectors

        length = torch.sqrt(dx**2 + dy**2 + adaptive_z**2 + 1e-8)        if should_invert and auto_invert_depth and not invert:        # Reshape back

        nx = -dx / length  # Flip X for correct surface orientation

        ny = dy / length   # Y gradient (no flip needed)            invert_status = " | Auto-Inverted"        dx = dx.reshape(batch, channels, height, width).permute(0, 2, 3, 1)

        nz = adaptive_z / length

                elif should_invert:        dy = dy.reshape(batch, channels, height, width).permute(0, 2, 3, 1)

        # Map to RGB with proper normalization

        r = nx * 0.5 + 0.5  # X gradient -> Red            invert_status = " | Inverted"        

        g = ny * 0.5 + 0.5  # Y gradient -> Green 

        b = nz * 0.5 + 0.5  # Z component -> Blue                    return dx, dy

        

        # Apply clamping        info = f"Depth to Normal | Strength: {strength:.1f} | Details: {enhance_details:.1f}{invert_status}"    

        r = torch.clamp(r, 0, 1)

        g = torch.clamp(g, 0, 1)            def _create_normal_map(self, dx, dy):

        b = torch.clamp(b, 0, 1)

                return (normal_map, info)        """Create RGB normal map from gradients"""

        # Combine into RGB normal map

        normal_map = torch.cat([r, g, b], dim=-1)            # Z component (pointing up)

        

        return normal_map    def _preprocess_depth(self, gray, enhance_details):        dz = torch.ones_like(dx)

        """Specialized preprocessing for depth maps"""        

        # Depth maps benefit from contrast enhancement and edge preservation        # Normalize the normal vectors

                length = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

        # Histogram stretching for better dynamic range        nx = dx / length

        min_val = torch.min(gray)        ny = dy / length

        max_val = torch.max(gray)        nz = dz / length

        range_val = max_val - min_val + 1e-8        

        stretched = (gray - min_val) / range_val        # Convert from [-1,1] to [0,1] for RGB

                r = (nx + 1.0) * 0.5  # X gradient -> Red

        # Apply adaptive histogram equalization equivalent        g = (ny + 1.0) * 0.5  # Y gradient -> Green  

        # This enhances local contrast while preserving global structure        b = (nz + 1.0) * 0.5  # Z up -> Blue

        if enhance_details > 0:        

            # Calculate local mean with larger kernel for depth maps        # Combine into RGB normal map

            kernel_size = 15        normal_map = torch.cat([r, g, b], dim=-1)

            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=gray.device) / (kernel_size**2)        

                    return torch.clamp(normal_map, 0, 1)

            # Ensure proper padding to maintain size

            padded_input = stretched.permute(0, 3, 1, 2)# Node registration

            local_mean = F.conv2d(padded_input, kernel, padding=kernel_size//2)NODE_CLASS_MAPPINGS = {

            local_mean = local_mean.permute(0, 2, 3, 1)    "ApexImageToNormal": ApexImageToNormal

            }

            # Enhance details relative to local mean

            enhanced = stretched + enhance_details * 0.3 * (stretched - local_mean)NODE_DISPLAY_NAME_MAPPINGS = {

            enhanced = torch.clamp(enhanced, 0, 1)    "ApexImageToNormal": "🎯 Apex Image to Normal"

        else:}

            enhanced = stretched    

            def depth_to_normal(self, depth_image, strength=2.0, coordinate_system="OpenGL (Blender/Unity)", 

        return enhanced                       quality="High (5x5 Sobel)", blur_radius=0.5, invert_depth=False, 

                           edge_enhance=0.0, seamless_edges=True, detail_scale=1.0):

    def _apply_blur(self, image, blur_amount):        try:

        """Apply Gaussian blur"""            print(f"ApexDepthToNormal: Processing image with shape {depth_image.shape}")

        device = image.device            

        batch, height, width, channels = image.shape            # Ensure image is in correct format [B, H, W, C]

                    if len(depth_image.shape) == 3:

        # Create Gaussian kernel                depth_image = depth_image.unsqueeze(0)

        kernel_size = int(blur_amount * 4) * 2 + 1            

        sigma = blur_amount            device = depth_image.device

                    batch_size, height, width, channels = depth_image.shape

        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2            print(f"ApexDepthToNormal: Device={device}, Size=({batch_size},{height},{width},{channels})")

        kernel = torch.exp(-(x**2) / (2 * sigma**2))            

        kernel = kernel / kernel.sum()            # Convert to grayscale if needed

                    if channels == 3:

        # Use standard 2D Gaussian kernel instead of separable convolution                # Use luminance weights for better depth perception

        kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)                weights = torch.tensor([0.299, 0.587, 0.114], device=device)

        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]                depth = torch.sum(depth_image * weights, dim=-1, keepdim=True)

                    else:

        # Reshape for convolution                depth = depth_image

        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)            

                    # Invert depth if requested (for inverted depth maps)

        # Apply 2D convolution with proper padding            if invert_depth:

        padding = kernel_size // 2                depth = 1.0 - depth

        blurred = F.conv2d(img_reshaped, kernel_2d, padding=padding)            

                    # Apply preprocessing

        return blurred.reshape(batch, channels, height, width).permute(0, 2, 3, 1)            depth = self._preprocess_depth(depth, blur_radius, edge_enhance, seamless_edges)

                

    def _calculate_gradients(self, image, strength):            # Scale depth for detail control

        """Calculate gradients optimized for depth maps"""            depth = depth * detail_scale

        device = image.device            

                    # Calculate gradients based on quality setting

        # For depth maps: use precise Sobel operators for better surface detection            if quality == "Fast (3x3 Sobel)":

        # Enhanced 3x3 Sobel for better detail capture                dx, dy = self._sobel_3x3(depth, strength)

        sobel_x = torch.tensor([            elif quality == "Ultra (Custom gradients)":

            [-1, 0, 1],                dx, dy = self._custom_gradients(depth, strength)

            [-2, 0, 2],             else:  # High (5x5 Sobel)

            [-1, 0, 1]                dx, dy = self._sobel_5x5(depth, strength)

        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0            

                    # Create normal map based on coordinate system

        sobel_y = torch.tensor([            normal_map = self._create_normal_map(dx, dy, coordinate_system)

            [-1, -2, -1],            

            [0, 0, 0],            # Generate info

            [1, 2, 1]            normal_info = self._generate_normal_info(coordinate_system, quality, strength, detail_scale)

        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0            

                    print(f"ApexDepthToNormal: Generated normal map with shape {normal_map.shape}")

        padding = 1            print(f"ApexDepthToNormal: Normal info: {normal_info}")

        strength_multiplier = 3.0  # Higher strength for depth maps            

                    return (torch.clamp(normal_map, 0, 1), normal_info)

        # Reshape for convolution            

        batch, height, width, channels = image.shape        except Exception as e:

        img_reshaped = image.permute(0, 3, 1, 2).reshape(-1, 1, height, width)            print(f"ApexDepthToNormal ERROR: {str(e)}")

                    import traceback

        # Calculate gradients with enhanced strength            traceback.print_exc()

        dx = F.conv2d(img_reshaped, sobel_x, padding=padding) * strength * strength_multiplier            # Return a fallback normal map (flat normal pointing up)

        dy = F.conv2d(img_reshaped, sobel_y, padding=padding) * strength * strength_multiplier            fallback = torch.zeros_like(depth_image)

                    if len(fallback.shape) == 4:

        # Enhance gradients with adaptive scaling for depth maps                fallback = fallback.repeat(1, 1, 1, 3) if fallback.shape[-1] == 1 else fallback

        gradient_magnitude = torch.sqrt(dx**2 + dy**2)            else:

        max_gradient = torch.max(gradient_magnitude) + 1e-8                fallback = fallback.unsqueeze(-1).repeat(1, 1, 1, 3) if len(fallback.shape) == 3 else fallback

                    fallback[:, :, :, 2] = 1.0  # Blue channel = Z up

        # Enhance weaker gradients more to reveal subtle surface details            return (fallback, f"Error: {str(e)}")

        enhancement_factor = 1.0 + (1.0 - gradient_magnitude / max_gradient) * 0.5

        dx = dx * enhancement_factor    def _preprocess_depth(self, depth, blur_radius, edge_enhance, seamless_edges):

        dy = dy * enhancement_factor        """Advanced depth preprocessing for better normal generation"""

                device = depth.device

        # Reshape back using original dimensions        

        dx = dx.reshape(batch, channels, height, width).permute(0, 2, 3, 1)        # Apply Gaussian blur for smoothing

        dy = dy.reshape(batch, channels, height, width).permute(0, 2, 3, 1)        if blur_radius > 0:

                    depth = self._gaussian_blur_depth(depth, blur_radius)

        return dx, dy        

            # Edge enhancement using unsharp masking

    def _create_normal_map(self, dx, dy):        if edge_enhance > 0:

        """Create RGB normal map from gradients with enhanced processing"""            # Create slightly blurred version

        # Calculate Z component with proper scaling for surface detail            blurred = self._gaussian_blur_depth(depth, 1.0)

        # Lower Z values = more pronounced surface features            # Enhance edges

        z_base = 0.3  # Much lower base Z for better surface definition            depth = depth + edge_enhance * (depth - blurred)

                    depth = torch.clamp(depth, 0, 1)

        # Calculate gradient magnitude for adaptive Z scaling        

        gradient_magnitude = torch.sqrt(dx**2 + dy**2)        # Seamless edge handling for tileable textures

                if seamless_edges:

        # Adaptive Z that responds to gradient strength            depth = self._make_seamless(depth)

        # Strong gradients get lower Z (more surface detail)        

        # Weak gradients get higher Z (flatter areas)        return depth

        max_gradient = torch.max(gradient_magnitude) + 1e-8

        normalized_gradient = gradient_magnitude / max_gradient    def _gaussian_blur_depth(self, depth, radius):

                """Optimized Gaussian blur for depth preprocessing"""

        # Inverse relationship: stronger gradients = lower Z        device = depth.device

        adaptive_z = z_base * (2.0 - normalized_gradient)        sigma = radius / 2.0

                kernel_size = int(2 * np.ceil(2 * sigma) + 1)

        # Normalize the normal vectors        

        length = torch.sqrt(dx**2 + dy**2 + adaptive_z**2 + 1e-8)        # Create 1D Gaussian kernel

        nx = -dx / length  # Flip X for correct surface orientation        x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2

        ny = dy / length   # Y gradient (no flip needed)        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))

        nz = adaptive_z / length        kernel_1d = kernel_1d / kernel_1d.sum()

                kernel_1d = kernel_1d.view(1, 1, kernel_size)

        # Map to RGB with DSINE-style normalization        

        # Convert from [-1,1] range to [0,1] RGB range        # Apply separable convolution

        r = nx * 0.5 + 0.5  # X gradient -> Red        batch_size, height, width, channels = depth.shape

        g = ny * 0.5 + 0.5  # Y gradient -> Green         depth_reshaped = depth.permute(0, 3, 1, 2).reshape(-1, 1, height, width)

        b = nz * 0.5 + 0.5  # Z component -> Blue        

                padding = kernel_size // 2

        # Apply DSINE-style clamping        

        r = torch.clamp(r, 0, 1)        # Horizontal pass

        g = torch.clamp(g, 0, 1)        blurred = F.conv2d(depth_reshaped, kernel_1d, padding=(0, padding))

        b = torch.clamp(b, 0, 1)        

                # Vertical pass  

        # Combine into RGB normal map        kernel_1d_v = kernel_1d.transpose(-1, -2)

        normal_map = torch.cat([r, g, b], dim=-1)        blurred = F.conv2d(blurred, kernel_1d_v, padding=(padding, 0))

                

        return normal_map        return blurred.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)

    def _make_seamless(self, depth):
        """Make depth map seamless for tiling"""
        batch_size, height, width, channels = depth.shape
        
        # Apply cosine falloff at edges for seamless tiling
        y_coords = torch.linspace(0, 1, height, device=depth.device)
        x_coords = torch.linspace(0, 1, width, device=depth.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Create edge falloff mask
        edge_size = 0.1  # 10% edge blending
        y_mask = torch.where(y_coords < edge_size, 
                           torch.cos((edge_size - y_coords) * np.pi / (2 * edge_size)),
                           torch.ones_like(y_coords))
        y_mask = torch.where(y_coords > 1 - edge_size,
                           torch.cos((y_coords - 1 + edge_size) * np.pi / (2 * edge_size)),
                           y_mask)
        
        x_mask = torch.where(x_coords < edge_size,
                           torch.cos((edge_size - x_coords) * np.pi / (2 * edge_size)),
                           torch.ones_like(x_coords))
        x_mask = torch.where(x_coords > 1 - edge_size,
                           torch.cos((x_coords - 1 + edge_size) * np.pi / (2 * edge_size)),
                           x_mask)
        
        # Combine masks
        edge_mask = y_mask.unsqueeze(1) * x_mask.unsqueeze(0)
        edge_mask = edge_mask.unsqueeze(0).unsqueeze(-1)
        
        return depth * edge_mask + depth.mean() * (1 - edge_mask)

    def _sobel_3x3(self, depth, strength):
        """Fast 3x3 Sobel gradient calculation"""
        device = depth.device
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        batch_size, height, width, channels = depth.shape
        depth_reshaped = depth.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        
        dx = F.conv2d(depth_reshaped, sobel_x, padding=1) * strength
        dy = F.conv2d(depth_reshaped, sobel_y, padding=1) * strength
        
        dx = dx.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        dy = dy.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return dx, dy

    def _sobel_5x5(self, depth, strength):
        """High-quality 5x5 Sobel gradient calculation"""
        device = depth.device
        
        # 5x5 Sobel kernels for better quality
        sobel_x_5x5 = torch.tensor([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 32.0
        
        sobel_y_5x5 = torch.tensor([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 32.0
        
        # Apply convolution
        batch_size, height, width, channels = depth.shape
        depth_reshaped = depth.permute(0, 3, 1, 2).reshape(-1, 1, height, width)
        
        dx = F.conv2d(depth_reshaped, sobel_x_5x5, padding=2) * strength
        dy = F.conv2d(depth_reshaped, sobel_y_5x5, padding=2) * strength
        
        dx = dx.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        dy = dy.reshape(batch_size, channels, height, width).permute(0, 2, 3, 1)
        
        return dx, dy

    def _custom_gradients(self, depth, strength):
        """Ultra-quality custom gradient calculation"""
        device = depth.device
        
        # Multi-scale gradient approach
        # Calculate gradients at different scales and combine
        dx1, dy1 = self._sobel_3x3(depth, strength * 0.6)
        dx2, dy2 = self._sobel_5x5(depth, strength * 0.4)
        
        # Apply slight Gaussian blur to 5x5 gradients for smoothness
        dx2 = self._gaussian_blur_depth(dx2, 0.3)
        dy2 = self._gaussian_blur_depth(dy2, 0.3)
        
        # Combine scales
        dx = dx1 + dx2
        dy = dy1 + dy2
        
        return dx, dy

    def _create_normal_map(self, dx, dy, coordinate_system):
        """Create properly normalized normal map"""
        device = dx.device
        
        # Calculate Z component (surface normal)
        dz = torch.ones_like(dx)
        
        # Normalize the normal vectors
        length = torch.sqrt(dx**2 + dy**2 + dz**2)
        dx = dx / (length + 1e-8)
        dy = dy / (length + 1e-8) 
        dz = dz / (length + 1e-8)
        
        # Adjust for coordinate system
        if coordinate_system == "DirectX (3ds Max/Maya)":
            # DirectX: X=right, Y=up, Z=forward
            r = (dx + 1) * 0.5   # X gradient → Red
            g = (1 - dy) * 0.5   # Flip Y gradient → Green  
            b = (dz + 1) * 0.5   # Z component → Blue
        elif coordinate_system == "Mikkt (Game Engines)":
            # Mikkt tangent space (used by many game engines)
            r = (dx + 1) * 0.5   # X gradient → Red
            g = (dy + 1) * 0.5   # Y gradient → Green
            b = (dz + 1) * 0.5   # Z component → Blue
        else:  # OpenGL (Blender/Unity)
            # OpenGL: X=right, Y=up, Z=out
            r = (dx + 1) * 0.5   # X gradient → Red
            g = (dy + 1) * 0.5   # Y gradient → Green
            b = (dz + 1) * 0.5   # Z component → Blue
        
        # Combine channels
        normal_map = torch.cat([r, g, b], dim=-1)
        
        return normal_map

    def _generate_normal_info(self, coordinate_system, quality, strength, detail_scale):
        """Generate information about the normal map generation"""
        info = [
            f"Normal Map Generated",
            f"Coordinate System: {coordinate_system}",
            f"Quality: {quality}",
            f"Strength: {strength:.1f}",
            f"Detail Scale: {detail_scale:.1f}"
        ]
        
        return " | ".join(info)
        
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
# ComfyUI registration - no test functions needed in production
NODE_CLASS_MAPPINGS = {
    "ApexDepthToNormal": ApexDepthToNormal
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexDepthToNormal": "🎯 Apex Depth to Normal"
}