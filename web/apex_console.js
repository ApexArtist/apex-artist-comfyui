import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create display widget
                this.displayWidget = this.addWidget("text", "display", "🎯 Waiting for input...", () => {}, {
                    multiline: true,
                    serialize: false
                });
                
                // Style the widget
                setTimeout(() => {
                    if (this.displayWidget?.inputEl) {
                        const el = this.displayWidget.inputEl;
                        el.style.backgroundColor = "#001100";
                        el.style.color = "#00ff41";
                        el.style.fontFamily = "monospace";
                        el.style.fontSize = "11px";
                        el.style.height = "200px";
                        el.style.border = "1px solid #00ff41";
                        el.style.borderRadius = "4px";
                        el.style.padding = "10px";
                        el.style.resize = "vertical";
                        el.style.whiteSpace = "pre-wrap";
                        el.readOnly = true;
                    }
                }, 50);
                
                this.size = [400, 280];
                return result;
            };
            
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (message?.console_text && this.displayWidget) {
                    this.displayWidget.value = message.console_text[0];
                    
                    // Apply theme
                    const theme = message.theme?.[0] || "matrix";
                    this.applyTheme(theme);
                }
                return onExecuted?.apply(this, arguments);
            };
            
            // Theme colors
            nodeType.prototype.applyTheme = function(theme) {
                if (!this.displayWidget?.inputEl) return;
                
                const themes = {
                    matrix: { bg: "#001100", color: "#00ff41", border: "#00ff41" },
                    cyberpunk: { bg: "#0a0a0a", color: "#ff00ff", border: "#ff00ff" },
                    classic: { bg: "#000000", color: "#00ff00", border: "#00ff00" },
                    terminal: { bg: "#0c0c0c", color: "#ffffff", border: "#666666" }
                };
                
                const t = themes[theme] || themes.matrix;
                const el = this.displayWidget.inputEl;
                
                el.style.backgroundColor = t.bg;
                el.style.color = t.color;
                el.style.borderColor = t.border;
            };
        }
    }
});
