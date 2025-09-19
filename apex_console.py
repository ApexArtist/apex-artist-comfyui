import json
import datetime
import torch
import numpy as np

class ApexConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Use ComfyUI's standard "any" type format
                "input_data": (["*"], {"tooltip": "Connect any data type here"}),
            },
            "optional": {
                "custom_label": ("STRING", {"default": "APEX CONSOLE"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "show_timestamp": ("BOOLEAN", {"default": True}),
                "max_lines": ("INT", {"default": 30, "min": 10, "max": 100}),
                "theme": (["matrix", "cyberpunk", "classic", "dracula", "neon"], {"default": "matrix"}),
                "auto_scroll": ("BOOLEAN", {"default": True}),
                "detailed_info": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    
    FUNCTION = "display_console"
    OUTPUT_NODE = True
    CATEGORY = "Apex Artist/Console"
    
    # Allow any input type
    INPUT_IS_LIST = False
    
    @classmethod
    def VALIDATE_INPUTS(cls, input_data, **kwargs):
        # Accept any input type
        return True
    
    def get_log_emoji(self, log_level: str) -> str:
        """Get emoji for log level"""
        emojis = {
            "DEBUG": "🔍", "INFO": "ℹ️", "WARNING": "⚠️", 
            "ERROR": "❌", "SUCCESS": "✅"
        }
        return emojis.get(log_level, "📝")
    
    def analyze_any_data(self, data, detailed=True):
        """Analyze any type of data and return formatted info"""
        try:
            data_type = type(data).__name__
            
            # STRING DATA
            if isinstance(data, str):
                if not data:
                    return "📝 Empty String"
                
                # Try JSON parsing
                try:
                    json_data = json.loads(data)
                    if detailed:
                        formatted = json.dumps(json_data, indent=2)
                        return f"📝 JSON String ({len(data)} chars):\n{formatted}"
                    else:
                        return f"📝 JSON String ({len(data)} chars)"
                except:
                    if len(data) > 100 and detailed:
                        return f"📝 Text String ({len(data)} chars):\n{data[:100]}..."
                    else:
                        return f"📝 Text String: {data}"
            
            # NUMERIC DATA
            elif isinstance(data, (int, float)):
                if isinstance(data, int):
                    return f"🔢 Integer: {data:,}"
                else:
                    return f"📊 Float: {data:.6f}"
            
            # BOOLEAN DATA
            elif isinstance(data, bool):
                status = "✅ TRUE" if data else "❌ FALSE"
                return f"🔘 Boolean: {status}"
            
            # TENSOR DATA (PyTorch)
            elif hasattr(data, 'shape') and hasattr(data, 'dtype'):
                shape = tuple(data.shape)
                dtype = str(data.dtype)
                
                # Check if it's a torch tensor
                if hasattr(data, 'device'):
                    device = str(data.device)
                    size_mb = data.element_size() * data.nelement() / (1024 * 1024)
                    
                    if detailed:
                        return f"🎨 Tensor: {shape}\n   Type: {dtype} | Device: {device}\n   Memory: ~{size_mb:.1f}MB"
                    else:
                        return f"🎨 Tensor: {shape} | {dtype} | ~{size_mb:.1f}MB"
                else:
                    # NumPy array
                    size_mb = data.nbytes / (1024 * 1024)
                    if detailed:
                        return f"🖼️ NumPy Array: {shape}\n   Type: {dtype} | Memory: ~{size_mb:.1f}MB"
                    else:
                        return f"🖼️ Array: {shape} | {dtype}"
            
            # LIST DATA
            elif isinstance(data, list):
                length = len(data)
                if length == 0:
                    return "📋 Empty List"
                
                # Check if it's a list of tensors (common in ComfyUI)
                if length > 0 and hasattr(data[0], 'shape'):
                    first_shape = tuple(data[0].shape)
                    return f"📋 Tensor List: {length} items | Shape: {first_shape}"
                else:
                    if detailed and length <= 5:
                        items_preview = ", ".join([str(item)[:20] for item in data[:3]])
                        if length > 3:
                            items_preview += "..."
                        return f"📋 List ({length} items): [{items_preview}]"
                    else:
                        return f"📋 List: {length} items"
            
            # DICTIONARY DATA
            elif isinstance(data, dict):
                keys = list(data.keys())
                
                # Special handling for ComfyUI data structures
                if 'samples' in data and hasattr(data['samples'], 'shape'):
                    # Latent space data
                    shape = tuple(data['samples'].shape)
                    return f"🎨 Latent Space: {shape} | Batch: {shape[0]}"
                
                elif any(key in keys for key in ['model', 'clip', 'vae']):
                    # Model bundle
                    return f"🤖 Model Bundle: {', '.join(keys)}"
                
                elif any('cond' in str(key).lower() for key in keys):
                    # Conditioning data
                    return f"🎛️ Conditioning: {len(keys)} components"
                
                else:
                    if detailed and len(keys) <= 10:
                        keys_str = ", ".join([str(k)[:15] for k in keys[:5]])
                        if len(keys) > 5:
                            keys_str += f"... (+{len(keys)-5} more)"
                        return f"📊 Dictionary ({len(keys)} keys):\n   {keys_str}"
                    else:
                        return f"📊 Dictionary: {len(keys)} keys"
            
            # TUPLE DATA
            elif isinstance(data, tuple):
                return f"📦 Tuple: {len(data)} items | {tuple(type(item).__name__ for item in data[:3])}"
            
            # UNKNOWN/CUSTOM OBJECTS
            else:
                # Try to get useful info about the object
                attrs = []
                if hasattr(data, 'shape'):
                    attrs.append(f"shape: {getattr(data, 'shape')}")
                if hasattr(data, '__len__'):
                    try:
                        attrs.append(f"length: {len(data)}")
                    except:
                        pass
                
                attr_str = " | ".join(attrs) if attrs else "unknown structure"
                return f"🔧 {data_type}: {attr_str}"
                
        except Exception as e:
            return f"❌ Analysis Error: {str(e)}"
    
    def display_console(self, input_data, **kwargs):
        """Main console display function"""
        
        try:
            # Get settings with safe defaults
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
            console_lines.append("═" * 45)
            
            # Timestamp
            if show_timestamp:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                console_lines.append(f"⏰ {timestamp}")
                console_lines.append("")
            
            # Analyze the input data
            emoji = self.get_log_emoji(log_level)
            analysis = self.analyze_any_data(input_data, detailed_info)
            
            # Add analysis to console
            console_lines.append(f"{emoji} Data Analysis:")
            console_lines.append(f"{analysis}")
            
            # Additional metadata
            console_lines.append("")
            console_lines.append("─" * 45)
            
            # Memory info for large objects
            try:
                import sys
                size_bytes = sys.getsizeof(input_data)
                if hasattr(input_data, 'nbytes'):  # numpy
                    size_bytes = input_data.nbytes
                elif hasattr(input_data, 'element_size') and hasattr(input_data, 'nelement'):  # torch
                    size_bytes = input_data.element_size() * input_data.nelement()
                
                if size_bytes > 1024:
                    size_mb = size_bytes / (1024 * 1024)
                    console_lines.append(f"💾 Memory: ~{size_mb:.2f}MB")
                else:
                    console_lines.append(f"💾 Memory: {size_bytes} bytes")
            except:
                console_lines.append("💾 Memory: Unknown")
            
            # Python type info
            console_lines.append(f"🐍 Type: {type(input_data).__module__}.{type(input_data).__name__}")
            
            # Theme and settings
            console_lines.append(f"🎨 Theme: {theme.title()}")
            
            console_lines.append("═" * 45)
            
            # Limit output length
            if len(console_lines) > max_lines:
                console_lines = console_lines[:2] + ["⚠️ Output truncated..."] + console_lines[-max_lines+3:]
            
            console_output = "\n".join(console_lines)
            
            # Debug print
            print(f"\n🎯 APEX CONSOLE:\n{console_output}\n")
            
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
═════════════════════════════════════════════
❌ CONSOLE ERROR
{str(e)}
═════════════════════════════════════════════"""
            
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
