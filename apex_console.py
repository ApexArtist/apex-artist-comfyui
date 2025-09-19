import datetime

class ApexConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "label": ("STRING", {"default": "APEX CONSOLE"}),
                "theme": (["matrix", "cyberpunk", "classic", "terminal"], {"default": "matrix"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "Apex Artist/Console"
    
    def display_text(self, text_input, label="APEX CONSOLE", theme="matrix"):
        """Display text input in console widget"""
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        console_output = f"""🎯 {label}
═══════════════════════════════════════════════════
⏰ {timestamp}

{text_input}

═══════════════════════════════════════════════════"""
        
        return {
            "ui": {
                "console_text": [console_output],
                "theme": [theme]
            }
        }

NODE_CLASS_MAPPINGS = {
    "ApexConsole": ApexConsole
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApexConsole": "🎯 Apex Console"
}
