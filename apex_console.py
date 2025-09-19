import json
import datetime
from typing import Dict, Any

class ApexConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "string_input": ("STRING", {"forceInput": True, "default": ""}),
                "int_input": ("INT", {"forceInput": True, "default": 0}),
                "float_input": ("FLOAT", {"forceInput": True, "default": 0.0}),
                "boolean_input": ("BOOLEAN", {"forceInput": True, "default": False}),
                "image_input": ("IMAGE", {"forceInput": True}),
                "latent_input": ("LATENT", {"forceInput": True}),
                "conditioning_input": ("CONDITIONING", {"forceInput": True}),
                "model_input": ("MODEL", {"forceInput": True}),
                
                "custom_label": ("STRING", {"default": "Apex Console"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "auto_scroll": ("BOOLEAN", {"default": True}),
                "show_timestamp": ("BOOLEAN", {"default": True}),
                "max_lines": ("INT", {"default": 50, "min": 10, "max": 200}),
                "clear_on_run": ("BOOLEAN", {"default": True}),
                "theme": (["matrix", "cyberpunk", "classic", "dracula"], {"default": "matrix"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("console_log",)
    FUNCTION = "display_console"
    OUTPUT_NODE = True
    CATEGORY = "Apex Artist/Console"
    
    # Console themes with colors and styles
    THEMES = {
        "matrix": {
            "bg": "#0d1117", "text": "#00ff41", "accent": "#39ff14", 
            "error": "#ff4444", "warning": "#ffaa00", "success": "#00ff88",
            "timestamp": "#666666", "border": "#00ff41"
        },
        "cyberpunk": {
            "bg": "#0a0a0a", "text": "#ff00ff", "accent": "#00ffff", 
            "error": "#ff0080", "warning": "#ffff00", "success": "#80ff00",
            "timestamp": "#8080ff", "border": "#ff00ff"
        },
        "classic": {
            "bg": "#000000", "text": "#00ff00", "accent": "#ffffff", 
            "error": "#ff0000", "warning": "#ffff00", "success": "#00ff00",
            "timestamp": "#808080", "border": "#00ff00"
        },
        "dracula": {
            "bg": "#282a36", "text": "#f8f8f2", "accent": "#bd93f9", 
            "error": "#ff5555", "warning": "#ffb86c", "success": "#50fa7b",
            "timestamp": "#6272a4", "border": "#bd93f9"
        }
    }
    
    def get_log_emoji(self, log_level: str) -> str:
        """Get emoji for log level"""
        emojis = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        return emojis.get(log_level, "📝")
    
    def get_data_emoji(self, data_type: str) -> str:
        """Get emoji for data type"""
        emojis = {
            "string": "📝",
            "int": "🔢", 
            "float": "📊",
            "boolean": "🔘",
            "image": "🖼️",
            "latent": "🎨",
            "conditioning": "🎛️",
            "model": "🤖",
            "timestamp": "⏰",
            "memory": "💾",
            "processing": "⚙️",
            "dimensions": "📐",
            "file": "📁"
        }
        return emojis.get(data_type, "📋")
    
    def format_console_line(self, message: str, log_level: str, theme: str, show_timestamp: bool) -> str:
        """Format a single console line with colors and emojis"""
        emoji = self.get_log_emoji(log_level)
        
        if show_timestamp:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            time_part = f"⏰ {timestamp}"
        else:
            time_part = ""
            
        # Format: [TIMESTAMP] EMOJI LEVEL: MESSAGE
        if time_part:
            formatted = f"{time_part} {emoji} {log_level}: {message}"
        else:
            formatted = f"{emoji} {log_level}: {message}"
            
        return formatted
    
    def analyze_inputs(self, **kwargs) -> list:
        """Analyze all inputs and create console entries"""
        console_entries = []
        
        # Process each input type
        if kwargs.get('string_input'):
            try:
                # Try to parse as JSON for pretty formatting
                json_data = json.loads(kwargs['string_input'])
                formatted_json = json.dumps(json_data, indent=2)
                console_entries.append({
                    "type": "string",
                    "level": "INFO", 
                    "message": f"📝 JSON Data:\n{formatted_json}"
                })
            except:
                console_entries.append({
                    "type": "string",
                    "level": "INFO",
                    "message": f"📝 Text: {kwargs['string_input']}"
                })
        
        if kwargs.get('int_input', 0) != 0:
            console_entries.append({
                "type": "int",
                "level": "INFO",
                "message": f"🔢 Integer Value: {kwargs['int_input']:,}"
            })
            
        if kwargs.get('float_input', 0.0) != 0.0:
            console_entries.append({
                "type": "float", 
                "level": "INFO",
                "message": f"📊 Float Value: {kwargs['float_input']:.6f}"
            })
            
        if kwargs.get('boolean_input') is not None:
            status = "✅ TRUE" if kwargs['boolean_input'] else "❌ FALSE"
            console_entries.append({
                "type": "boolean",
                "level": "SUCCESS" if kwargs['boolean_input'] else "WARNING",
                "message": f"🔘 Boolean: {status}"
            })
            
        if kwargs.get('image_input') is not None:
            img = kwargs['image_input']
            if hasattr(img, 'shape'):
                h, w = img.shape[1], img.shape[2] if len(img.shape) > 2 else (0, 0)
                console_entries.append({
                    "type": "image",
                    "level": "INFO", 
                    "message": f"🖼️ Image: {w}×{h} pixels | Memory: ~{(w*h*3*4/1024/1024):.1f}MB"
                })
                
        if kwargs.get('latent_input') is not None:
            latent = kwargs['latent_input']
            if 'samples' in latent:
                shape = latent['samples'].shape
                console_entries.append({
                    "type": "latent",
                    "level": "INFO",
                    "message": f"🎨 Latent: {shape} | Batch: {shape[0]} | Channels: {shape[1]}"
                })
                
        return console_entries
    
    def display_console(self, **kwargs):
        """Main console display function"""
        # Get settings
        custom_label = kwargs.get('custom_label', 'Apex Console')
        log_level = kwargs.get('log_level', 'INFO')
        show_timestamp = kwargs.get('show_timestamp', True)
        theme = kwargs.get('theme', 'matrix')
        max_lines = kwargs.get('max_lines', 50)
        
        # Analyze inputs
        console_entries = self.analyze_inputs(**kwargs)
        
        # Create console output
        console_lines = []
        
        # Header
        header = f"{'='*60}"
        console_lines.append(f"🎯 {custom_label.upper()}")
        console_lines.append(header)
        
        # Process entries
        for entry in console_entries:
            formatted_line = self.format_console_line(
                entry['message'], 
                entry['level'], 
                theme, 
                show_timestamp
            )
            console_lines.append(formatted_line)
        
        # System info
        if console_entries:
            console_lines.append("─" * 60)
            console_lines.append(f"📊 Processed {len(console_entries)} data inputs")
            console_lines.append(f"🎨 Theme: {theme.title()}")
            
        # Limit lines
        if len(console_lines) > max_lines:
            console_lines = console_lines[-max_lines:]
            console_lines.insert(0, "⚠️ Output truncated to max lines...")
            
        # Join output
        console_output = "\n".join(console_lines)
        
        # Print to terminal with colors (for development)
        print(f"\n🎯 APEX CONSOLE:\n{console_output}\n")
        
        # Return for UI display
        return {
            "ui": {
                "text": [console_output]
            },
            "result": (console_output,)
        }
