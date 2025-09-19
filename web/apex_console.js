import { app } from "../../scripts/app.js";

// Load CSS
const link = document.createElement("link");
link.rel = "stylesheet";
link.type = "text/css";
link.href = new URL("./apex_console.css", import.meta.url).href;
document.head.appendChild(link);

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Remove the large external widget and create an embedded one
                this.consoleOutput = "";
                this.consoleTheme = "matrix";
                
                // Create a text widget for console display (embedded in node)
                const consoleWidget = this.addWidget(
                    "text", 
                    "console_output", 
                    "", 
                    function(v) {}, 
                    {
                        multiline: true,
                        serialize: false
                    }
                );
                
                // Style the widget
                consoleWidget.inputEl.className = "apex-console-embedded";
                consoleWidget.inputEl.style.cssText = `
                    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
                    color: #00ff41 !important;
                    font-family: 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace !important;
                    font-size: 10px !important;
                    line-height: 1.3 !important;
                    border: 1px solid #00ff41 !important;
                    border-radius: 6px !important;
                    padding: 8px !important;
                    resize: vertical !important;
                    min-height: 120px !important;
                    max-height: 400px !important;
                    height: 180px !important;
                    white-space: pre-wrap !important;
                    overflow-y: auto !important;
                    box-shadow: 
                        0 0 15px rgba(0, 255, 65, 0.3),
                        inset 0 0 15px rgba(0, 0, 0, 0.8) !important;
                `;
                
                // Make widget read-only but still interactive
                consoleWidget.inputEl.readOnly = true;
                consoleWidget.inputEl.placeholder = "🎯 APEX CONSOLE - Connect inputs to see output...";
                
                // Store reference to console widget
                this.consoleWidget = consoleWidget;
                
                // Adjust node size to accommodate console
                this.size = [400, 320];
                
                return result;
            };
            
            // Handle execution updates
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (message.console_text && this.consoleWidget) {
                    const consoleText = message.console_text[0];
                    const theme = message.theme ? message.theme[0] : "matrix";
                    
                    // Update console content
                    this.consoleWidget.value = consoleText;
                    this.consoleOutput = consoleText;
                    this.consoleTheme = theme;
                    
                    // Apply theme styling
                    this.applyConsoleTheme(theme);
                    
                    // Auto-scroll to bottom
                    setTimeout(() => {
                        if (this.consoleWidget.inputEl) {
                            this.consoleWidget.inputEl.scrollTop = this.consoleWidget.inputEl.scrollHeight;
                        }
                    }, 10);
                }
                return onExecuted?.apply(this, arguments);
            };
            
            // Add theme application method
            nodeType.prototype.applyConsoleTheme = function(theme) {
                if (!this.consoleWidget || !this.consoleWidget.inputEl) return;
                
                const themes = {
                    matrix: {
                        bg: "linear-gradient(135deg, #001100 0%, #003300 50%, #001100 100%)",
                        color: "#00ff41",
                        border: "#00ff41",
                        shadow: "rgba(0, 255, 65, 0.4)"
                    },
                    cyberpunk: {
                        bg: "linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a0a1a 100%)",
                        color: "#ff00ff",
                        border: "#ff00ff", 
                        shadow: "rgba(255, 0, 255, 0.4)"
                    },
                    classic: {
                        bg: "#000000",
                        color: "#00ff00",
                        border: "#00ff00",
                        shadow: "rgba(0, 255, 0, 0.4)"
                    },
                    dracula: {
                        bg: "linear-gradient(135deg, #282a36 0%, #44475a 50%, #282a36 100%)",
                        color: "#f8f8f2",
                        border: "#bd93f9",
                        shadow: "rgba(189, 147, 249, 0.4)"
                    }
                };
                
                const themeColors = themes[theme] || themes.matrix;
                
                this.consoleWidget.inputEl.style.background = themeColors.bg;
                this.consoleWidget.inputEl.style.color = themeColors.color;
                this.consoleWidget.inputEl.style.borderColor = themeColors.border;
                this.consoleWidget.inputEl.style.boxShadow = `
                    0 0 15px ${themeColors.shadow},
                    inset 0 0 15px rgba(0, 0, 0, 0.8)
                `;
            };
            
            // Handle resizing
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                if (onResize) {
                    onResize.apply(this, arguments);
                }
                
                // Adjust console widget size
                if (this.consoleWidget && this.consoleWidget.inputEl) {
                    const newHeight = Math.max(120, size[1] - 200);
                    this.consoleWidget.inputEl.style.height = `${newHeight}px`;
                }
            };
        }
    }
});
