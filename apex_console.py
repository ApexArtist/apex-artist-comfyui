import json
import datetime
import torch
import numpy as np

class ApexConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Use the same approach as ComfyUI's built-in nodes that accept any input
                "input_data": ("*",),
            },
            "optional": {
                "custom_label": ("STRING", {"default": "APEX CONSOLE"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "show_timestamp": ("BOOLEAN", {"default": True}),
                "max_lines": ("INT", {"default": 30, "min": 10, "max": 100}),
                "theme": (["matrix", "cyberpunk", "classic", "dracula"], {"default": "matrix"}),
                "detailed_info": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "display_console"
    OUTPUT_NODE = True
    CATEGORY = "Apex Artist/Console"
    
    # This tells ComfyUI to always execute when inputs change
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    
    def get_log_emoji(self, log_level: str) -> str:
        emojis = {
            "DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", 
            "ERROR": "❌", "SUCCESS": "✅"
        }
        return emojis.get(log_level, "📝")
    
    def analyze_data(self, data, detailed=True):
        """Analyze any type of data"""
        try:
            data_type = type(data).__name__
            
            # STRING DATA
            if isinstance(data, str):
                if not data:
                    return "📝 Empty String"
                try:
                    json_data = json.loads(data)
                    formatted = json.dumps(json_data, indent=2) if detailed else str(json_data)
                    return f"📝 JSON String ({len(data)} chars):\n{formatted}"
                except:
                    preview = data[:200] + "..." if len(data) > 200 and detailed else data
                    return f"📝 Text String ({len(data)} chars): {preview}"
            
            # NUMERIC DATA
            elif isinstance(data, (int, float)):
                return f"🔢 Number: {data}"
            
            # BOOLEAN
            elif isinstance(data, bool):
                return f"🔘 Boolean: {'✅ TRUE' if data else '❌ FALSE'}"
            
            # TORCH TENSOR
            elif hasattr(data, 'shape') and hasattr(data, 'dtype') and 'torch' in str(type(data)):
                shape = tuple(data.shape)
                dtype = str(data.dtype)
                device = str(data.device) if hasattr(data, 'device') else 'unknown'
                
                # Calculate memory usage
                if hasattr(data, 'element_size') and hasattr(data, 'nelement'):
                    size_mb = data.element_size() * data.nelement() / (1024 * 1024)
                    memory_info = f"~{size_mb:.1f}MB"
                else:
                    memory_info = "unknown"
                
                # Get tensor stats if possible
                try:
                    min_val = data.min().item()
                    max_val = data.max().item()
                    mean_val = data.mean().item()
                    
                    if detailed:
                        return f"""🎨 PyTorch Tensor: {shape}
   Type: {dtype} | Device: {device}
   Memory: {memory_info}
   Range: [{min_val:.4f}, {max_val:.4f}]
   Mean: {mean_val:.4f}"""
                    else:
                        return f"🎨 Tensor: {shape} | {dtype} | {memory_info}"
                except:
                    return f"🎨 Tensor: {shape} | {dtype} | {device} | {memory_info}"
            
            # NUMPY ARRAY
            elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
                shape = data.shape
                dtype = str(data.dtype)
                size_mb = data.nbytes / (1024 * 1024)
                
                try:
                    min_val = data.min()
                    max_val = data.max()
                    mean_val = data.mean()
                    
                    if detailed:
                        return f"""🖼️ NumPy Array: {shape}
   Type: {dtype}
   Memory: ~{size_mb:.1f}MB
   Range: [{min_val:.4f}, {max_val:.4f}]
   Mean: {mean_val:.4f}"""
                    else:
                        return f"🖼️ Array: {shape} | {dtype} | ~{size_mb:.1f}MB"
                except:
                    return f"🖼️ Array: {shape} | {dtype} | ~{size_mb:.1f}MB"
            
            # LIST
            elif isinstance(data, list):
                length = len(data)
                if length == 0:
                    return "📋 Empty List"
                
                # Check for list of tensors
                if length > 0 and hasattr(data[0], 'shape'):
                    first_shape = tuple(data[0].shape)
                    return f"📋 Tensor List: {length} items | First shape: {first_shape}"
                
                # Regular list
                if detailed and length <= 10:
                    preview = [str(item)[:30] for item in data[:5]]
                    if length > 5:
                        preview.append(f"... (+{length-5} more)")
                    return f"📋 List ({length} items):\n   {preview}"
                else:
                    return f"📋 List: {length} items"
            
            # DICTIONARY (ComfyUI data structures)
            elif isinstance(data, dict):
                keys = list(data.keys())
                
                # Latent space
                if 'samples' in data:
                    samples = data['samples']
                    if hasattr(samples, 'shape'):
                        shape = tuple(samples.shape)
                        return f"🎨 Latent Space: {shape} | Batch: {shape[0]} | Channels: {shape[1]}"
                
                # Model data
                if any(key in keys for key in ['model', 'clip', 'vae']):
                    components = [k for k in keys if k in ['model', 'clip', 'vae']]
                    return f"🤖 Model Bundle: {', '.join(components)}"
                
                # Conditioning
                if any('cond' in str(key).lower() for key in keys):
                    return f"🎛️ Conditioning Data: {len(keys)} components"
                
                # Image dimensions/metadata
                if 'width' in keys and 'height' in keys:
                    width = data.get('width', 'unknown')
                    height = data.get('height', 'unknown')
                    other_keys = [k for k in keys if k not in ['width', 'height']]
                    return f"📐 Image Metadata: {width}x{height}" + (f" + {other_keys}" if other_keys else "")
                
                # Regular dict
                if detailed and len(keys) <= 10:
                    key_preview = ", ".join([str(k)[:20] for k in keys[:5]])
                    if len(keys) > 5:
                        key_preview += f"... (+{len(keys)-5} more)"
                    return f"📊 Dictionary ({len(keys)} keys):\n   {key_preview}"
                else:
                    return f"📊 Dictionary: {len(keys)} keys"
            
            # TUPLE
            elif isinstance(data, tuple):
                types = [type(item).__name__ for item in data[:3]]
                type_str = ", ".join(types)
                if len(data) > 3:
                    type_str += f"... (+{len(data)-3} more)"
                return f"📦 Tuple ({len(data)} items): ({type_str})"
            
            # UNKNOWN OBJECT
            else:
                # Try to extract useful info
                info_parts = []
                
                if hasattr(data, 'shape'):
                    info_parts.append(f"shape: {getattr(data, 'shape')}")
                if hasattr(data, '__len__'):
                    try:
                        info_parts.append(f"length: {len(data)}")
                    except:
                        pass
                if hasattr(data, 'dtype'):
                    info_parts.append(f"dtype: {getattr(data, 'dtype')}")
                
                info_str = " | ".join(info_parts) if info_parts else "no accessible properties"
                return f"🔧 {data_type} Object: {info_str}"
                
        except Exception as e:
            return f"❌ Analysis Error: {str(e)}"
    
    def display_console(self, input_data, **kwargs):
        """Main console display function"""
        
        try:
            # Get settings
            custom_label = kwargs.get('custom_label', 'APEX CONSOLE')
            log_level = kwargs.get('log_level', 'INFO')
            show_timestamp = kwargs.get('show_timestamp', True)
            theme = kwargs.get('theme', 'matrix')
            max_lines = kwargs.get('max_lines', 30)
            detailed_info = kwargs.get('detailed_info', True)
            
            # Build console output
            console_lines = []
            
            # Header
            console_lines.append(f"🎯 {custom_label}")
            console_lines.append("═" * 50)
            
            # Timestamp
            if show_timestamp:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                console_lines.append(f"⏰ {timestamp}")
                console_lines.append("")
            
            # Analyze input
            emoji = self.get_log_emoji(log_level)
            analysis = self.analyze_data(input_data, detailed_info)
            
            console_lines.append(f"{emoji} INPUT ANALYSIS:")
            console_lines.append(analysis)
            console_lines.append("")
            
            # Footer info
            console_lines.append("─" * 50)
            console_lines.append(f"🐍 Python Type: {type(input_data).__name__}")
            console_lines.append(f"🎨 Theme: {theme.title()}")
            console_lines.append("═" * 50)
            
            # Limit lines
            if len(console_lines) > max_lines:
                console_lines = console_lines[:3] + ["⚠️ Output truncated..."] + console_lines[-(max_lines-4):]
            
            console_output = "\n".join(console_lines)
            
            # Debug print to ComfyUI console
            print(f"\n🎯 APEX CONSOLE OUTPUT:\n{console_output}\n")
            
            return {
                "ui": {
                    "console_text": [console_output],
                    "theme": [theme],
                    "timestamp": [datetime.datetime.now().isoformat()],
                    "data_type": [type(input_data).__name__]
                }
            }
            
        except Exception as e:
            error_output = f"""🎯 {kwargs.get('custom_label', 'APEX CONSOLE')}
════════════════════════════════════════════════════
❌ CONSOLE ERROR
{str(e)}
════════════════════════════════════════════════════"""
            
            return {
                "ui": {
                    "console_text": [error_output],
                    "theme": [kwargs.get('theme', 'matrix')],
                    "timestamp": [datetime.datetime.now().isoformat()],
                    "data_type": ["Error"]
                }
            }

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ApexConsole": ApexConsole
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexConsole": "Apex Console 🎯"
}
