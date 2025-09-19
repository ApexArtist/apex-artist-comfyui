import json
import datetime
import os

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
                
                "custom_label": ("STRING", {"default": "APEX CONSOLE"}),
                "log_level": (["DEBUG", "INFO", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "auto_scroll": ("BOOLEAN", {"default": True}),
                "show_timestamp": ("BOOLEAN", {"default": True}),
                "max_lines": ("INT", {"default": 50, "min": 10, "max": 200}),
                "clear_on_run": ("BOOLEAN", {"default": True}),
                "theme": (["matrix", "cyberpunk", "classic", "dracula"], {"default": "matrix"}),
            }
        }
    
    # Remove return types to prevent Preview Any connection
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    
    FUNCTION = "display_console"
    OUTPUT_NODE = True  # Makes it a terminal/display node
    CATEGORY = "Apex Artist/Console"
    
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
    
    def format_data_with_emoji(self, data_type: str, value) -> str:
        """Format data with appropriate emoji and styling"""
        
        if data_type == "string" and value:
            try:
                # Try to parse as JSON for pretty formatting
                json_data = json.loads(value)
                formatted = json.dumps(json_data, indent=2)
                return f"📝 JSON Data:\n{formatted}"
            except:
                return f"📝 Text: {value}"
                
        elif data_type == "int" and value != 0:
            return f"🔢 Integer: {value:,}"
            
        elif data_type == "float" and value != 0.0:
            return f"📊 Float: {value:.6f}"
            
        elif data_type == "boolean":
            status = "✅ TRUE" if value else "❌ FALSE"
            return f"🔘 Boolean: {status}"
            
        elif data_type == "image" and value is not None:
            if hasattr(value, 'shape') and len(value.shape) >= 3:
                h, w = value.shape[1], value.shape[2]
                memory_mb = (w * h * 3 * 4) / (1024 * 1024)
                return f"🖼️ Image: {w}×{h} pixels | ~{memory_mb:.1f}MB"
            else:
                return f"🖼️ Image data received"
                
        elif data_type == "latent" and value is not None:
            if isinstance(value, dict) and 'samples' in value:
                shape = value['samples'].shape
                return f"🎨 Latent: {shape} | Batch: {shape[0]} | Channels: {shape[1]}"
            else:
                return f"🎨 Latent data received"
                
        elif data_type == "conditioning" and value is not None:
            if isinstance(value, list) and len(value) > 0:
                return f"🎛️ Conditioning: {len(value)} items"
            else:
                return f"🎛️ Conditioning data received"
                
        elif data_type == "model" and value is not None:
            return f"🤖 Model data received"
            
        return None  # Skip empty/default values
    
    def display_console(self, **kwargs):
        """Main console display function"""
        
        # Get settings
        custom_label = kwargs.get('custom_label', 'APEX CONSOLE')
        log_level = kwargs.get('log_level', 'INFO')
        show_timestamp = kwargs.get('show_timestamp', True)
        theme = kwargs.get('theme', 'matrix')
        max_lines = kwargs.get('max_lines', 50)
        
        # Build console output
        console_lines = []
        
        # Header
        console_lines.append("=" * 60)
        console_lines.append(f"🎯 {custom_label}")
        console_lines.append("=" * 60)
        
        # Process each input type
        data_count = 0
        input_types = ['string', 'int', 'float', 'boolean', 'image', 'latent', 'conditioning', 'model']
        
        for data_type in input_types:
            input_key = f"{data_type}_input"
            value = kwargs.get(input_key)
            
            # Format the data
            formatted_data = self.format_data_with_emoji(data_type, value)
            
            if formatted_data:  # Only add if we have actual data
                data_count += 1
                
                # Add timestamp if enabled
                if show_timestamp:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    time_part = f"⏰ {timestamp}"
                else:
                    time_part = ""
                
                # Get log level emoji
                emoji = self.get_log_emoji(log_level)
                
                # Build the line
                if time_part:
                    line = f"{time_part} {emoji} {formatted_data}"
                else:
                    line = f"{emoji} {formatted_data}"
                    
                console_lines.append(line)
        
        # Footer
        console_lines.append("─" * 60)
        if data_count > 0:
            console_lines.append(f"📊 Processed {data_count} data inputs")
            console_lines.append(f"🎨 Theme: {theme.title()} | 📅 {datetime.datetime.now().strftime('%Y-%m-%d')}")
        else:
            console_lines.append("💤 No active data inputs")
            console_lines.append("🔌 Connect nodes to display console output")
        
        console_lines.append("=" * 60)
        
        # Limit lines if needed
        if len(console_lines) > max_lines:
            console_lines = ["⚠️ Output truncated to max lines..."] + console_lines[-max_lines:]
        
        # Join all lines
        console_output = "\n".join(console_lines)
        
        # Print to ComfyUI terminal for debugging
        print(f"\n🎯 APEX CONSOLE OUTPUT:\n{console_output}\n")
        
        # Return UI data only (no regular outputs to prevent Preview Any)
        return {
            "ui": {
                "console_text": [console_output],
                "theme": [theme],
                "timestamp": [datetime.datetime.now().isoformat()]
            }
        }

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ApexConsole": ApexConsole
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexConsole": "Apex Console 🎯"
}
