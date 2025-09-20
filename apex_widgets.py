"""
Custom widgets for ComfyUI nodes
"""
import numpy as np
from PIL import Image, ImageDraw

def create_histogram_overlay(histogram_data):
    """Create a histogram overlay image for the curve editor"""
    width = histogram_data["width"]
    height = histogram_data["height"]
    colors = histogram_data["colors"]
    histograms = histogram_data["histograms"]
    
    # Create base image with alpha channel (RGBA)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw each channel's histogram
    for hist, color in zip(histograms, colors):
        # Convert histogram to numpy for easier manipulation
        hist_np = hist.numpy()
        
        # Scale to fit height
        hist_np = hist_np * (height - 2)  # Leave 1px border
        
        # Draw histogram bars with alpha
        for x in range(width):
            y_height = int(hist_np[int(x * 256 / width)])
            rgb_color = tuple(int(c * 255) for c in color)
            draw.line(
                [(x, height), (x, height - y_height)],
                fill=rgb_color + (128,)  # Add 50% alpha
            )
    
    return overlay