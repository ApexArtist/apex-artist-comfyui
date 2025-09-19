import json
import torch
import numpy as np
from datetime import datetime
from PIL import Image

class ApexConsole:
    @classmethod
    def INPUT_TYPES(s):
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
                "log_level": (["INFO", "DEBUG", "WARNING", "ERROR", "SUCCESS"], {"default": "INFO"}),
                "auto_scroll": ("BOOLEAN", {"default": True}),
                "show_timestamp": ("BOOLEAN", {"default": True}),
                "max_lines": ("INT", {"default": 50, "min": 10, "max": 200, "step": 1}),
                "clear_on_run": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_output",)
    OUTPUT_NODE = True
    FUNCTION = "display_console"
    CATEGORY = "ApexArtist/Debug"
    
    def __init__(self):
        self.log_history = []
    
    def get_log_colors(self, log_level):
        """Get color scheme for different log levels"""
        colors = {
            "INFO": {"bg": "#1e293b", "text": "#e2e8f0", "border": "#475569", "accent": "#3b82f6"},
            "DEBUG": {"bg": "#0f172a", "text": "#94a3b8", "border": "#334155", "accent": "#6366f1"},
            "WARNING": {"bg": "#451a03", "text": "#fbbf24", "border": "#92400e", "accent": "#f59e0b"},
            "ERROR": {"bg": "#450a0a", "text": "#f87171", "border": "#991b1b", "accent": "#ef4444"},
            "SUCCESS": {"bg": "#052e16", "text": "#34d399", "border": "#166534", "accent": "#10b981"}
        }
        return colors.get(log_level, colors["INFO"])
    
    def format_any_data(self, data, data_type, label="Data"):
        """Universal data formatter - handles ANY ComfyUI data type"""
        
        if data is None:
            return f"🚫 {label}: None"
        
        try:
            if data_type == "STRING":
                return self.format_string_data(data, label)
            elif data_type == "INT":
                return f"🔢 {label}: {data:,}"
            elif data_type == "FLOAT":
                return f"📊 {label}: {data:.6f}"
            elif data_type == "BOOLEAN":
                icon = "✅" if data else "❌"
                return f"{icon} {label}: {data}"
            elif data_type == "IMAGE":
                return self.format_image_data(data, label)
            elif data_type == "LATENT":
                return self.format_latent_data(data, label)
            elif data_type == "CONDITIONING":
                return self.format_conditioning_data(data, label)
            elif data_type == "MODEL":
                return self.format_model_data(data, label)
            else:
                return self.format_unknown_data(data, label, data_type)
                
        except Exception as e:
            return f"⚠️ {label} ({data_type}): Error formatting - {str(e)}"
    
    def is_status_message(self, text):
        """Detect status-style messages"""
        status_indicators = ['✅', '❌', '⚠️', '🎯', '📊', 'success', 'complete', 'finished', 'done', 'loaded', 'failed', 'error']
        return any(indicator in text.lower() for indicator in status_indicators)
    
    def is_performance_log(self, text):
        """Detect performance/timing logs"""
        perf_indicators = ['time', 'seconds', 'ms', 'memory', 'vram', 'gpu', 'mb', 'gb', 'parameters', 'params']
        return any(indicator in text.lower() for indicator in perf_indicators)
    
    def is_error_message(self, text):
        """Detect error messages"""
        error_indicators = ['❌', 'error', 'failed', 'exception', 'traceback', 'warning', '⚠️']
        return any(indicator in text.lower() for indicator in error_indicators)
    
    def format_status_message(self, data, label):
        """Format status messages with enhanced styling"""
        if '✅' in data or any(word in data.lower() for word in ['success', 'complete', 'finished', 'done', 'loaded']):
            icon = '🎉'
            status = 'SUCCESS'
        elif '⚠️' in data or 'warning' in data.lower():
            icon = '⚠️'
            status = 'WARNING'
        elif '❌' in data or any(word in data.lower() for word in ['error', 'failed', 'exception']):
            icon = '🚨'
            status = 'ERROR'
        else:
            icon = '📋'
            status = 'INFO'
        
        return f"{icon} {label} [{status}]:\n{self.indent_text(data)}"
    
    def format_performance_log(self, data, label):
        """Format performance/timing logs"""
        return f"⏱️ {label} [PERFORMANCE]:\n{self.indent_text(data)}"
    
    def format_string_data(self, data, label):
        """Enhanced string formatting for all types of logs"""
        if not data:
            return f"📝 {label}: (empty string)"
        
        # Skip zero values that are meaningless
        if data == "0" or data == "0.0" or data == "False":
            return f"📝 {label}: {data}"
        
        # Detect and format different message types
        if self.is_status_message(data):
            return self.format_status_message(data, label)
        elif self.is_performance_log(data):
            return self.format_performance_log(data, label)
        elif self.is_error_message(data):
            return self.format_status_message(data, label)  # Uses same formatting but different detection
        
        # Try to parse as JSON
        try:
            if data.strip().startswith(('{', '[')):
                json_data = json.loads(data)
                return f"📄 {label} [JSON]:\n{self.format_json_data(json_data)}"
        except:
            pass
        
        # Multi-line string
        if '\n' in data:
            lines = data.split('\n')
            return f"📝 {label} ({len(lines)} lines):\n{self.indent_text(data)}"
        
        # Single line string
        if len(data) > 100:
            return f"📝 {label} ({len(data)} chars): {data[:97]}..."
        return f"📝 {label}: {data}"
    
    def format_image_data(self, data, label):
        """Format IMAGE tensor information"""
        if isinstance(data, torch.Tensor):
            batch, height, width, channels = data.shape
            dtype = str(data.dtype).replace('torch.', '')
            memory_mb = (data.numel() * data.element_size()) / (1024 * 1024)
            
            stats = ""
            if data.numel() > 0:
                stats = f"\n   📈 Range: [{data.min():.3f}, {data.max():.3f}]"
                stats += f"\n   📊 Mean: {data.mean():.3f}, Std: {data.std():.3f}"
            
            return f"""🖼️ {label}:
   📐 Shape: {batch}x{height}x{width}x{channels}
   🔢 Type: {dtype}
   💾 Memory: {memory_mb:.1f} MB{stats}"""
        
        return f"🖼️ {label}: {type(data).__name__}"
    
    def format_latent_data(self, data, label):
        """Format LATENT dictionary information"""
        if isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
            if isinstance(samples, torch.Tensor):
                batch, channels, height, width = samples.shape
                dtype = str(samples.dtype).replace('torch.', '')
                memory_mb = (samples.numel() * samples.element_size()) / (1024 * 1024)
                
                stats = ""
                if samples.numel() > 0:
                    stats = f"\n   📈 Range: [{samples.min():.3f}, {samples.max():.3f}]"
                    stats += f"\n   📊 Mean: {samples.mean():.3f}, Std: {samples.std():.3f}"
                
                extra_keys = [k for k in data.keys() if k != 'samples']
                extra_info = f"\n   🔑 Extra keys: {extra_keys}" if extra_keys else ""
                
                return f"""🎭 {label}:
   📐 Samples: {batch}x{channels}x{height}x{width}
   🔢 Type: {dtype}
   💾 Memory: {memory_mb:.1f} MB{stats}{extra_info}"""
        
        return f"🎭 {label}: {type(data).__name__}"
    
    def format_conditioning_data(self, data, label):
        """Format CONDITIONING list information"""
        if isinstance(data, list) and len(data) > 0:
            info = f"🎯 {label} ({len(data)} items):"
            
            for i, item in enumerate(data[:3]):  # Show first 3 items
                if isinstance(item, list) and len(item) >= 2:
                    cond_tensor, pooled = item[0], item[1] if len(item) > 1 else None
                    
                    if isinstance(cond_tensor, torch.Tensor):
                        shape = list(cond_tensor.shape)
                        info += f"\n   [{i}] Tensor: {shape}, Type: {str(cond_tensor.dtype).replace('torch.', '')}"
                    
                    if isinstance(pooled, dict):
                        info += f"\n       Pooled: {len(pooled)} keys"
                else:
                    info += f"\n   [{i}] {type(item).__name__}"
            
            if len(data) > 3:
                info += f"\n   ... and {len(data) - 3} more items"
            
            return info
        
        return f"🎯 {label}: {type(data).__name__}"
    
    def format_model_data(self, data, label):
        """Format MODEL information"""
        model_info = f"🧠 {label}: {type(data).__name__}"
        
        # Try to get model information
        try:
            if hasattr(data, 'model'):
                model_info += f"\n   📦 Inner model: {type(data.model).__name__}"
            
            if hasattr(data, 'model_config'):
                model_info += f"\n   ⚙️ Config available"
            
            # Estimate memory usage
            total_params = 0
            if hasattr(data, 'parameters'):
                for param in data.parameters():
                    total_params += param.numel()
                memory_mb = (total_params * 4) / (1024 * 1024)  # Assume float32
                model_info += f"\n   📊 Params: ~{total_params:,} ({memory_mb:.1f} MB)"
                
        except:
            pass
        
        return model_info
    
    def format_json_data(self, data, indent=1):
        """Format JSON data with nice indentation and icons"""
        indent_str = "   " * indent
        
        if isinstance(data, dict):
            if not data:
                return "{}"
            
            result = ""
            for key, value in data.items():
                icon = self.get_value_icon(value)
                if isinstance(value, (dict, list)):
                    result += f"\n{indent_str}{icon} {key}:"
                    result += self.format_json_data(value, indent + 1)
                else:
                    result += f"\n{indent_str}{icon} {key}: {value}"
            return result
            
        elif isinstance(data, list):
            if not data:
                return "[]"
            
            result = ""
            for i, item in enumerate(data):
                icon = self.get_value_icon(item)
                if isinstance(item, (dict, list)):
                    result += f"\n{indent_str}{icon} [{i}]:"
                    result += self.format_json_data(item, indent + 1)
                else:
                    result += f"\n{indent_str}{icon} [{i}]: {item}"
            return result
            
        return str(data)
    
    def get_value_icon(self, value):
        """Get appropriate icon for different value types"""
        if isinstance(value, bool):
            return "✅" if value else "❌"
        elif isinstance(value, (int, float)):
            return "🔢"
        elif isinstance(value, str):
            return "📝"
        elif isinstance(value, list):
            return "📋"
        elif isinstance(value, dict):
            return "📦"
        else:
            return "🔹"
    
    def format_unknown_data(self, data, label, data_type):
        """Format unknown data types"""
        type_name = type(data).__name__
        
        # Try to get basic info
        info = f"❓ {label} ({data_type}):\n   🏷️ Type: {type_name}"
        
        # Try to get size/length
        try:
            if hasattr(data, '__len__'):
                info += f"\n   📏 Length: {len(data)}"
        except:
            pass
        
        # Try to get tensor info if it's tensor-like
        try:
            if hasattr(data, 'shape'):
                info += f"\n   📐 Shape: {data.shape}"
            if hasattr(data, 'dtype'):
                info += f"\n   🔢 Dtype: {data.dtype}"
        except:
            pass
        
        # String representation (truncated)
        try:
            str_repr = str(data)
            if len(str_repr) > 200:
                str_repr = str_repr[:197] + "..."
            info += f"\n   📄 Value: {str_repr}"
        except:
            info += f"\n   📄 Value: <unable to convert to string>"
        
        return info
    
    def indent_text(self, text, indent="   "):
        """Add indentation to text"""
        return '\n'.join(indent + line for line in text.split('\n'))
    
    def add_emoji_colors(self, message):
        """Add colors to emojis and special characters"""
        emoji_colors = {
            '✅': '#10b981', '❌': '#ef4444', '🔢': '#3b82f6', '📝': '#8b5cf6',
            '📄': '#06b6d4', '📋': '#f59e0b', '📦': '#ec4899', '🖼️': '#84cc16',
            '🎭': '#f97316', '🎯': '#eab308', '🧠': '#6366f1', '📐': '#14b8a6',
            '💾': '#64748b', '📈': '#22c55e', '📊': '#0ea5e9', '🔑': '#f59e0b',
            '⚠️': '#f59e0b', '🚫': '#ef4444', '❓': '#6b7280', '🔹': '#64748b',
            '📏': '#8b5cf6', '🏷️': '#ec4899', '🎉': '#10b981', '🚨': '#ef4444',
            '⏱️': '#3b82f6'
        }
        
        for emoji, color in emoji_colors.items():
            message = message.replace(emoji, f'<span style="color: {color};">{emoji}</span>')
        
        return message
    
    def create_console_html(self, log_entries, title, colors, auto_scroll, max_lines):
        """Create the HTML console display"""
        if len(log_entries) > max_lines:
            log_entries = log_entries[-max_lines:]
        
        log_html = ""
        for entry in log_entries:
            entry_colors = self.get_log_colors(entry["level"])
            timestamp_html = f'<span style="color: {entry_colors["accent"]}; font-size: 0.8em;">[{entry["timestamp"]}]</span> ' if entry["timestamp"] else ""
            level_html = f'<span style="color: {entry_colors["accent"]}; font-weight: bold;">[{entry["level"]}]</span> '
            
            # Process message for HTML display
            message = entry["message"].replace('\n', '<br>')
            message = self.add_emoji_colors(message)
            
            log_html += f'''
            <div style="
                margin-bottom: 8px; 
                padding: 12px; 
                background: {entry_colors["bg"]}; 
                border-left: 4px solid {entry_colors["accent"]}; 
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.5;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                {timestamp_html}{level_html}<span style="color: {entry_colors["text"]};">{message}</span>
            </div>
            '''
        
        scroll_script = """
        <script>
        setTimeout(() => {
            const console = document.getElementById('apex-console');
            if (console) {
                console.scrollTop = console.scrollHeight;
            }
        }, 100);
        </script>
        """ if auto_scroll else ""
        
        html = f'''
        <div style="
            width: 100%; 
            max-width: 900px; 
            margin: 10px auto;
            background: {colors["bg"]}; 
            border: 2px solid {colors["border"]}; 
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(135deg, {colors["accent"]}, {colors["border"]}); 
                color: white; 
                padding: 16px 20px; 
                font-weight: bold; 
                font-family: 'Segoe UI', 'Helvetica', sans-serif;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <span style="font-size: 18px;">🖥️</span>
                <span>{title}</span>
                <span style="margin-left: auto; font-size: 12px; opacity: 0.8;">
                    {len(self.log_history)} total entries
                </span>
            </div>
            <div id="apex-console" style="
                max-height: 500px; 
                overflow-y: auto; 
                padding: 20px; 
                background: {colors["bg"]};
            ">
                {log_html if log_html else '<div style="color: #64748b; font-style: italic; text-align: center; padding: 20px;">No data to display</div>'}
            </div>
        </div>
        {scroll_script}
        '''
        
        return html
    
    def display_console(self, custom_label="Apex Console", log_level="INFO", auto_scroll=True, 
                       show_timestamp=True, max_lines=50, clear_on_run=False, **kwargs):
        
        if clear_on_run:
            self.log_history = []
        
        colors = self.get_log_colors(log_level)
        
        # Collect all non-None inputs
        inputs_found = []
        data_types = {
            'string_input': 'STRING', 'int_input': 'INT', 'float_input': 'FLOAT',
            'boolean_input': 'BOOLEAN', 'image_input': 'IMAGE', 'latent_input': 'LATENT',
            'conditioning_input': 'CONDITIONING', 'model_input': 'MODEL'
        }
        
        for input_name, data_type in data_types.items():
            if input_name in kwargs and kwargs[input_name] is not None:
                value = kwargs[input_name]
                
                # Skip meaningless default values
                skip_value = False
                if isinstance(value, str) and value == "":
                    skip_value = True
                elif isinstance(value, (int, float)) and value == 0 and input_name in ['int_input', 'float_input']:
                    skip_value = True
                elif isinstance(value, bool) and value == False and input_name == 'boolean_input':
                    skip_value = True
                
                if not skip_value:
                    formatted_data = self.format_any_data(value, data_type, input_name.replace('_input', '').replace('_', ' ').title())
                    inputs_found.append(formatted_data)
        
        # Create log entry
        if inputs_found:
            message = '\n\n'.join(inputs_found)
        else:
            message = "🔍 No data inputs detected"
        
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S") if show_timestamp else "",
            "level": log_level,
            "message": message
        }
        
        self.log_history.append(log_entry)
        
        # Create HTML console
        console_html = self.create_console_html([log_entry], custom_label, colors, auto_scroll, max_lines)
        
        return {
            "ui": {"text": [console_html]},
            "result": (message,)
        }

# Utility class for easy logging from other nodes
class ApexLogger:
    @staticmethod
    def format_resize_info(resolution_set, original_size, new_size, candidates, chosen_index):
        """Helper to format resize information for console"""
        return json.dumps({
            "action": "Smart Resize",
            "resolution_set": resolution_set,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "new_size": f"{new_size[0]}x{new_size[1]}",
            "candidates": [f"{w}x{h}" for w, h in candidates],
            "chosen_index": chosen_index
        }, indent=2)
    
    @staticmethod
    def format_simple_message(message, data=None):
        """Helper to format simple messages"""
        if data:
            return json.dumps({"message": message, "data": data}, indent=2)
        return message
    
    @staticmethod
    def format_performance_data(action, time_taken, memory_used=None, additional_info=None):
        """Helper to format performance data"""
        data = {
            "action": action,
            "time_seconds": time_taken,
        }
        if memory_used:
            data["memory_mb"] = memory_used
        if additional_info:
            data.update(additional_info)
        return json.dumps(data, indent=2)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ApexConsole": ApexConsole,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexConsole": "🖥️ Apex Console",
}
