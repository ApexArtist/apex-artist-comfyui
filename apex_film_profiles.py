"""
Film stock LUTs and curve preview generation for Apex Artist nodes
"""
import torch
import numpy as np
from PIL import Image, ImageDraw

class FilmStockProfiles:
    """Film stock emulation profiles based on classic film characteristics"""
    
    @classmethod
    def get_profiles(cls):
        return {
            # Kodak Color Films
            "kodak_portra_160": {
                "curves": {
                    "r": [(0, 0.05), (0.2, 0.25), (0.5, 0.55), (0.8, 0.85), (1, 0.95)],
                    "g": [(0, 0.05), (0.2, 0.23), (0.5, 0.52), (0.8, 0.82), (1, 0.93)],
                    "b": [(0, 0.08), (0.2, 0.22), (0.5, 0.48), (0.8, 0.8), (1, 0.9)]
                },
                "characteristics": {
                    "saturation": 0.95,
                    "contrast": 0.85,
                    "grain_size": 1.4,
                    "grain_intensity": 0.15,
                    "shadow_grain": 1.2,
                    "highlight_grain": 0.8,
                    "color_temp": 0.0,
                    "tint": -0.02
                }
            },
            "kodak_portra_400": {
                "curves": {
                    "r": [(0, 0.08), (0.2, 0.28), (0.5, 0.58), (0.8, 0.88), (1, 0.98)],
                    "g": [(0, 0.07), (0.2, 0.25), (0.5, 0.55), (0.8, 0.85), (1, 0.95)],
                    "b": [(0, 0.1), (0.2, 0.24), (0.5, 0.52), (0.8, 0.82), (1, 0.92)]
                },
                "characteristics": {
                    "saturation": 0.9,
                    "contrast": 0.8,
                    "grain_size": 1.8,
                    "grain_intensity": 0.25,
                    "shadow_grain": 1.3,
                    "highlight_grain": 0.7,
                    "color_temp": 0.1,
                    "tint": -0.05
                }
            },
            "kodak_portra_800": {
                "curves": {
                    "r": [(0, 0.1), (0.2, 0.3), (0.5, 0.6), (0.8, 0.9), (1, 1)],
                    "g": [(0, 0.09), (0.2, 0.27), (0.5, 0.57), (0.8, 0.87), (1, 0.97)],
                    "b": [(0, 0.12), (0.2, 0.26), (0.5, 0.54), (0.8, 0.84), (1, 0.94)]
                },
                "characteristics": {
                    "saturation": 0.85,
                    "contrast": 0.75,
                    "grain": 0.6
                }
            },
            "kodak_ektar_100": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.25), (0.5, 0.6), (0.8, 0.9), (1, 1)],
                    "g": [(0, 0), (0.2, 0.23), (0.5, 0.58), (0.8, 0.88), (1, 0.98)],
                    "b": [(0, 0), (0.2, 0.22), (0.5, 0.55), (0.8, 0.85), (1, 0.95)]
                },
                "characteristics": {
                    "saturation": 1.2,
                    "contrast": 1.1,
                    "grain": 0.1
                }
            },
            
            # Fujifilm Color Films
            "fuji_provia_100f": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.22), (0.5, 0.52), (0.8, 0.85), (1, 1)],
                    "g": [(0, 0), (0.2, 0.21), (0.5, 0.51), (0.8, 0.84), (1, 0.98)],
                    "b": [(0, 0), (0.2, 0.23), (0.5, 0.53), (0.8, 0.86), (1, 1)]
                },
                "characteristics": {
                    "saturation": 1.1,
                    "contrast": 1.0,
                    "grain": 0.15
                }
            },
            "fuji_velvia_50": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.25), (0.5, 0.65), (0.8, 0.9), (1, 1)],
                    "g": [(0, 0), (0.2, 0.23), (0.5, 0.63), (0.8, 0.88), (1, 0.98)],
                    "b": [(0, 0), (0.2, 0.28), (0.5, 0.68), (0.8, 0.92), (1, 1)]
                },
                "characteristics": {
                    "saturation": 1.4,
                    "contrast": 1.2,
                    "grain": 0.1
                }
            },
            "fuji_astia_100f": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.21), (0.5, 0.51), (0.8, 0.82), (1, 0.95)],
                    "g": [(0, 0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.81), (1, 0.93)],
                    "b": [(0, 0), (0.2, 0.22), (0.5, 0.52), (0.8, 0.83), (1, 0.96)]
                },
                "characteristics": {
                    "saturation": 0.9,
                    "contrast": 0.85,
                    "grain": 0.15
                }
            },
            
            # Black & White Films
            "ilford_delta_100": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.82), (1, 0.95)],
                    "g": [(0, 0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.82), (1, 0.95)],
                    "b": [(0, 0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.82), (1, 0.95)]
                },
                "characteristics": {
                    "saturation": 0,
                    "contrast": 1.1,
                    "grain": 0.2
                }
            },
            "ilford_hp5_plus": {
                "curves": {
                    "r": [(0, 0.05), (0.2, 0.25), (0.5, 0.55), (0.8, 0.85), (1, 0.98)],
                    "g": [(0, 0.05), (0.2, 0.25), (0.5, 0.55), (0.8, 0.85), (1, 0.98)],
                    "b": [(0, 0.05), (0.2, 0.25), (0.5, 0.55), (0.8, 0.85), (1, 0.98)]
                },
                "characteristics": {
                    "saturation": 0,
                    "contrast": 0.9,
                    "grain": 0.5
                }
            },
            "kodak_tmax_100": {
                "curves": {
                    "r": [(0, 0), (0.2, 0.22), (0.5, 0.52), (0.8, 0.85), (1, 1)],
                    "g": [(0, 0), (0.2, 0.22), (0.5, 0.52), (0.8, 0.85), (1, 1)],
                    "b": [(0, 0), (0.2, 0.22), (0.5, 0.52), (0.8, 0.85), (1, 1)]
                },
                "characteristics": {
                    "saturation": 0,
                    "contrast": 1.2,
                    "grain": 0.15
                }
            },
            
            # Vintage Films
            "kodak_gold_200": {
                "curves": {
                    "r": [(0, 0.05), (0.2, 0.3), (0.5, 0.6), (0.8, 0.85), (1, 0.95)],
                    "g": [(0, 0.05), (0.2, 0.25), (0.5, 0.55), (0.8, 0.8), (1, 0.9)],
                    "b": [(0, 0.1), (0.2, 0.2), (0.5, 0.45), (0.8, 0.75), (1, 0.85)]
                },
                "characteristics": {
                    "saturation": 1.1,
                    "contrast": 0.9,
                    "grain": 0.3
                }
            },
            "fuji_superia_400": {
                "curves": {
                    "r": [(0, 0.08), (0.2, 0.28), (0.5, 0.58), (0.8, 0.88), (1, 0.98)],
                    "g": [(0, 0.07), (0.2, 0.27), (0.5, 0.57), (0.8, 0.87), (1, 0.97)],
                    "b": [(0, 0.09), (0.2, 0.29), (0.5, 0.59), (0.8, 0.89), (1, 0.99)]
                },
                "characteristics": {
                    "saturation": 1.05,
                    "contrast": 0.85,
                    "grain": 0.4
                }
            }
        }

    @staticmethod
    def generate_preview(curve_points, width=256, height=256, color=(255,255,255)):
        """Generate a preview image for a curve"""
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Scale points to image dimensions
        points = [(int(x * width), int((1-y) * height)) for x, y in curve_points]
        
        # Draw grid
        grid_color = (128, 128, 128, 64)
        for i in range(4):
            x = (i + 1) * width // 4
            draw.line([(x, 0), (x, height)], grid_color, 1)
            y = (i + 1) * height // 4
            draw.line([(0, y), (width, y)], grid_color, 1)
        
        # Draw curve
        if len(points) > 1:
            # Create smoother curve with more points
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            # Interpolate additional points
            x_new = np.linspace(0, width, 100)
            y_new = np.interp(x_new, x_points, y_points)
            
            # Draw smooth curve
            smooth_points = list(zip(x_new, y_new))
            for i in range(len(smooth_points)-1):
                draw.line([smooth_points[i], smooth_points[i+1]], color + (200,), 2)
        
        # Draw control points
        for point in points:
            draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], 
                        fill=color + (255,))
        
        return image
    
    @classmethod
    def get_film_preview(cls, film_name):
        """Generate preview images for a film stock profile"""
        if film_name not in cls.get_profiles():
            return None
        
        profile = cls.get_profiles()[film_name]
        previews = {}
        
        # Generate preview for each channel
        for channel, curve in profile["curves"].items():
            color = {"r": (255,0,0), "g": (0,255,0), "b": (0,0,255)}[channel]
            previews[channel] = cls.generate_preview(curve, color=color)
        
        # Create composite preview
        composite = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        for preview in previews.values():
            composite = Image.alpha_composite(composite, preview)
        
        return composite

def create_curve_preview(curve_points, width=256, height=256, color=(255,255,255), 
                       show_histogram=False, histogram_data=None):
    """Create a preview image for a curve with optional histogram overlay"""
    # Create base image
    image = FilmStockProfiles.generate_preview(curve_points, width, height, color)
    
    # Add histogram if requested
    if show_histogram and histogram_data is not None:
        histogram_overlay = create_histogram_overlay(histogram_data)
        image = Image.alpha_composite(image, histogram_overlay)
    
    return image