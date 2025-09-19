import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create console text widget
                const consoleWidget = this.addWidget(
                    "text", 
                    "console_display", 
                    "🎯 Connect any data to analyze...", 
                    function(v) {}, 
                    {
                        multiline: true,
                        serialize: false
                    }
                );
                
                // Enhanced styling
                consoleWidget.inputEl.style.cssText = `
                    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
                    color: #00ff41 !important;
                    font-family: 'Fira Code', 'SF Mono', 'Consolas', 'Monaco', monospace !important;
                    font-size: 11px !important;
                    line-height: 1.4 !important;
                    border: 2px solid #00ff41 !important;
                    border-radius: 8px !important;
                    padding: 12px !important;
                    margin: 6px 0 !important;
                    resize: vertical !important;
                    min-height: 150px !important;
                    max-height: 600px !important;
                    height: 220px !important;
                    white-space: pre-wrap !important;
                    overflow-y: auto !important;
                    box-shadow: 
                        0 0 20px rgba(0, 255, 65, 0.4),
                        inset 0 0 20px rgba(0, 0, 0, 0.8) !important;
                    backdrop-filter: blur(3px) !important;
                    transition: all 0.3s ease !important;
                `;
                
                consoleWidget.inputEl.readOnly = true;
                this.consoleWidget = consoleWidget;
                
                // Adjust node size
                this.size = [420, 380];
                
                return result;
            };
            
            // Enhanced execution handler
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (message.console_text && this.consoleWidget) {
                    const consoleText = message.console_text[0];
                    const theme = message.theme ? message.theme[0] : "matrix";
                    const dataType = message.data_type ? message.data_type[0] : "unknown";
                    
                    // Update content
                    this.consoleWidget.value = consoleText;
                    
                    // Apply theme
                    this.applyConsoleTheme(theme);
                    
                    // Add data type indicator to node title
                    this.title = `Apex Console 🎯 [${dataType}]`;
                    
                    // Auto-scroll
                    setTimeout(() => {
                        if (this.consoleWidget.inputEl) {
                            this.consoleWidget.inputEl.scrollTop = this.consoleWidget.inputEl.scrollHeight;
                        }
                    }, 50);
                    
                    // Brief glow effect
                    this.consoleWidget.inputEl.style.animation = "consoleGlow 0.8s ease-out";
                    setTimeout(() => {
                        this.consoleWidget.inputEl.style.animation = "";
                    }, 800);
                }
                return onExecuted?.apply(this, arguments);
            };
            
            // Theme application
            nodeType.prototype.applyConsoleTheme = function(theme) {
                if (!this.consoleWidget?.inputEl) return;
                
                const themes = {
                    matrix: { bg: "linear-gradient(135deg, #001100 0%, #003300 50%, #001100 100%)", color: "#00ff41", border: "#00ff41" },
                    cyberpunk: { bg: "linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a0a1a 100%)", color: "#ff00ff", border: "#ff00ff" },
                    classic: { bg: "#000000", color: "#00ff00", border: "#00ff00" },
                    dracula: { bg: "linear-gradient(135deg, #282a36 0%, #44475a 50%, #282a36 100%)", color: "#f8f8f2", border: "#bd93f9" },
                    neon: { bg: "linear-gradient(135deg, #001122 0%, #002244 50%, #001122 100%)", color: "#00ffff", border: "#00ffff" }
                };
                
                const t = themes[theme] || themes.matrix;
                const el = this.consoleWidget.inputEl;
                
                el.style.background = t.bg;
                el.style.color = t.color;
                el.style.borderColor = t.border;
                el.style.boxShadow = `0 0 20px ${t.border}40, inset 0 0 20px rgba(0, 0, 0, 0.8)`;
            };
        }
    }
});
