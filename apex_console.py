import json
import datetime

class ApexConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*",),
                "custom_label": ("STRING", {"default": "APEX CONSOLE"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "theme": (["matrix", "cyberpunk", "classic"], {"default": "matrix"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "display_console"
    OUTPUT_NODE = True
    CATEGORY = "Apex Artist/Console"
    
    def display_console(self, prompt=None, extra_pnginfo=None, **kwargs):
        """Simple console display that actually works"""
        
        # Get the actual input data - it comes through kwargs
        any_input = kwargs.get('any_input', "No input connected")
        custom_label = kwargs.get('custom_label', 'APEX CONSOLE')
        log_level = kwargs.get('log_level', 'INFO')
        theme = kwargs.get('theme', 'matrix')
        
        # Simple data analysis
        if any_input == "No input connected":
            data_info = "📭 No data connected"
        elif isinstance(any_input, str):
            data_info = f"📝 String ({len(any_input)} chars): {any_input[:100]}"
        elif isinstance(any_input, (int, float)):
            data_info = f"🔢 Number: {any_input}"
        elif isinstance(any_input, dict):
            if 'samples' in any_input:
                shape = any_input['samples'].shape if hasattr(any_input['samples'], 'shape') else 'unknown'
                data_info = f"🎨 Latent: {shape}"
            else:
                data_info = f"📊 Dict with {len(any_input)} keys: {list(any_input.keys())[:3]}"
        elif isinstance(any_input, list):
            data_info = f"📋 List with {len(any_input)} items"
        elif hasattr(any_input, 'shape'):
            data_info = f"🖼️ Tensor/Array: {any_input.shape}"
        else:
            data_info = f"🔧 {type(any_input).__name__}"
        
        # Build simple output
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        console_output = f"""🎯 {custom_label}
═══════════════════════════════════════════
⏰ {timestamp}

{log_level} ► {data_info}

🐍 Type: {type(any_input).__name__}
🎨 Theme: {theme}
═══════════════════════════════════════════"""
        
        print(f"\n{console_output}\n")
        
        return {
            "ui": {
                "text": [console_output]
            }
        }

NODE_CLASS_MAPPINGS = {
    "ApexConsole": ApexConsole
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexConsole": "🎯 Apex Console"
}
