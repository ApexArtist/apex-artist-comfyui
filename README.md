# Apex Artist - ComfyUI Image Resize Node

Professional image resizing node for ComfyUI with advanced algorithms and aspect ratio handling.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Features

### 🎯 **Multiple Resize Algorithms**
- **Lanczos** - Highest quality for most use cases
- **Bicubic** - Excellent for smooth gradients
- **Bilinear** - Fast with decent quality
- **Nearest** - Perfect for pixel art
- **Box** - Optimized for downscaling
- **Hamming** - Alternative high-quality option

### 📐 **Smart Aspect Ratio Handling**
- **Keep Proportions** - Maintains original aspect ratio
- **Stretch Mode** - Direct resize ignoring proportions
- Automatic dimension rounding to multiples of 8

### ✂️ **Crop & Fill Options**
When maintaining aspect ratios:
- **Crop Center** - Crops from center of image
- **Crop Top** - Keeps top portion of image
- **Crop Bottom** - Keeps bottom portion of image
- **Pad Center** - Adds padding centered
- **Pad Top** - Adds padding at bottom
- **Pad Bottom** - Adds padding at top

### 🎨 **Advanced Options**
- **Padding Colors**: Black, White, Gray, Transparent
- **Resize Factor**: Quick scaling multiplier (0.1x - 8.0x)
- **Post-Sharpening**: Optional enhancement after resize
- **Batch Processing**: Handles multiple images efficiently

## Installation

### Method 1: Git Clone (Recommended)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/apex-artist-comfyui.git
```

### Method 2: Manual Download
1. Download the repository as ZIP
2. Extract to `ComfyUI/custom_nodes/apex-artist-comfyui/`
3. Restart ComfyUI

### Method 3: ComfyUI Manager
1. Open ComfyUI Manager
2. Search for "Apex Artist"
3. Click Install

## Usage

1. **Add the Node**: Find "Apex Artist - Image Resize" in the node menu under "Apex Artist" category
2. **Connect Input**: Connect your image output to the image input
3. **Set Dimensions**: Configure width and height (auto-rounded to multiples of 8)
4. **Choose Algorithm**: Select resize method based on your needs
5. **Configure Handling**: Set aspect ratio and crop/pad preferences

### Quick Start Example
```
Image → Apex Artist → Save Image
```

## Node Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| image | IMAGE | - | Input image tensor |
| width | INT | 512 | Target width (64-8192) |
| height | INT | 512 | Target height (64-8192) |
| resize_method | COMBO | lanczos | Resizing algorithm |
| keep_proportion | BOOLEAN | True | Maintain aspect ratio |
| upscale_method | COMBO | crop_center | How to handle aspect ratio |
| pad_color | COMBO | black | Padding color (optional) |
| sharpen_after_resize | BOOLEAN | False | Post-process sharpening (optional) |
| resize_factor | FLOAT | 1.0 | Scale multiplier (optional) |

## Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| resized_image | IMAGE | Processed image tensor |
| final_width | INT | Actual output width |
| final_height | INT | Actual output height |
| resize_info | STRING | Detailed operation information |

## Algorithm Guide

### When to Use Each Algorithm:

- **Lanczos**: Best general-purpose choice, excellent quality
- **Bicubic**: Great for photographic images with smooth transitions
- **Bilinear**: Fast processing, good for real-time applications
- **Nearest**: Essential for pixel art, maintains sharp edges
- **Box**: Optimal for significant downscaling
- **Hamming**: Alternative high-quality option, similar to Lanczos

### Aspect Ratio Handling:

- **Crop Methods**: Remove parts of image to fit exact dimensions
- **Pad Methods**: Add borders to maintain full image content
- **Stretch**: Distort image to fit exact dimensions

## Examples

### High-Quality Photo Resize
```
Settings:
- Algorithm: Lanczos
- Keep Proportion: True
- Method: Pad Center
- Pad Color: Black
```

### Pixel Art Upscaling
```
Settings:
- Algorithm: Nearest
- Keep Proportion: True
- Method: Crop Center
```

### Quick Batch Processing
```
Settings:
- Algorithm: Bilinear
- Resize Factor: 0.5
- Keep Proportion: True
```

## Requirements

- ComfyUI (latest version)
- Python 3.8+
- PIL (Pillow) - included with ComfyUI
- PyTorch - included with ComfyUI
- NumPy - included with ComfyUI

## Troubleshooting

### Common Issues:

**Node not appearing in menu:**
- Restart ComfyUI completely
- Check console for error messages
- Verify file placement in custom_nodes directory

**Memory errors with large images:**
- Use smaller resize factors
- Process images individually instead of batches

**Quality issues:**
- Try Lanczos or Bicubic for better quality
- Enable post-sharpening for crisp results
- Avoid excessive upscaling (>4x)

## 🤝 **Contributing**

We welcome contributions from the community! Here's how you can help:

### **🌟 Ways to Contribute**
- 🐛 **Report bugs** and suggest fixes
- 💡 **Request features** that would improve your workflow  
- 📝 **Improve documentation** and examples
- 🔧 **Submit code** improvements and optimizations
- ⭐ **Star the repo** to show support

### **📋 Development Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/apex-artist-comfyui.git
cd apex-artist-comfyui

# Install development dependencies
pip install -e .

# Make your changes and test
# Submit a Pull Request
```

### **✅ Contribution Guidelines**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request with detailed description

---

## 📜 **License & Legal**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.

**Summary**: You are free to use, modify, distribute, and sell this software. Just include the original license notice.

---

## 📈 **Changelog**

### **v1.0.0** *(Latest)*
- 🎉 **Initial Release**
- ✨ 6 professional resize algorithms
- 🧠 Intelligent aspect ratio handling  
- ✂️ Advanced crop and pad options
- ⚡ Optimized batch processing
- 📖 Comprehensive documentation

### **🔮 Roadmap**
- 🎨 Additional artistic filters
- 🔄 Batch size optimization
- 📊 Performance monitoring
- 🎯 Custom algorithm presets
- 🖼️ Preview thumbnails

---

## ❤️ **Acknowledgments**

**Built with love for the ComfyUI community**

- 🙏 **ComfyUI Team** - For the amazing framework
- 👥 **Community Contributors** - For feedback and suggestions  
- 🔧 **Python/PIL Developers** - For the robust image processing tools
- 🎨 **Digital Artists** - For inspiring better tools

---

<div align="center">

### **⭐ Star this repo if it helped your workflow! ⭐**

**[🏠 Homepage](https://github.com/yourusername/apex-artist-comfyui)** • 
**[📥 Releases](https://github.com/yourusername/apex-artist-comfyui/releases)** • 
**[🐛 Issues](https://github.com/yourusername/apex-artist-comfyui/issues)** • 
**[💡 Discussions](https://github.com/yourusername/apex-artist-comfyui/discussions)**

*Made with ❤️ for creators, by creators*

</div>
