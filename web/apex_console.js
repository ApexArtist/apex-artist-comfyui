import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                try {
                    // Create console text widget with error checking
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
                    
                    // Wait for DOM element to be ready
                    setTimeout(() => {
                        if (consoleWidget && consoleWidget.inputEl) {
                            this.setupConsoleStyles(consoleWidget);
                        }
                    }, 100);
                    
                    this.consoleWidget = consoleWidget;
                    
                    // Adjust node size
                    this.size = [420, 380];
                    
                } catch (error) {
                    console.error("ApexConsole: Error in onNodeCreated:", error);
                }
                
                return result;
            };
            
            // Add setup method for console styles
            nodeType.prototype.setupConsoleStyles = function(consoleWidget) {
                try {
                    if (!consoleWidget || !consoleWidget.inputEl) {
                        console.warn("ApexConsole: Console widget or inputEl not available");
                        return;
                    }
                    
                    const el = consoleWidget.inputEl;
                    
                    // Apply styles safely
                    el.style.setProperty('background', 'linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%)', 'important');
                    el.style.setProperty('color', '#00ff41', 'important');
                    el.style.setProperty('font-family', '"Fira Code", "SF Mono", "Consolas", "Monaco", monospace', 'important');
                    el.style.setProperty('font-size', '11px', 'important');
                    el.style.setProperty('line-height', '1.4', 'important');
                    el.style.setProperty('border', '2px solid #00ff41', 'important');
                    el.style.setProperty('border-radius', '8px', 'important');
                    el.style.setProperty('padding', '12px', 'important');
                    el.style.setProperty('margin', '6px 0', 'important');
                    el.style.setProperty('resize', 'vertical', 'important');
                    el.style.setProperty('min-height', '150px', 'important');
                    el.style.setProperty('max-height', '600px', 'important');
                    el.style.setProperty('height', '220px', 'important');
                    el.style.setProperty('white-space', 'pre-wrap', 'important');
                    el.style.setProperty('overflow-y', 'auto', 'important');
                    el.style.setProperty('box-shadow', '0 0 20px rgba(0, 255, 65, 0.4), inset 0 0 20px rgba(0, 0, 0, 0.8)', 'important');
                    el.style.setProperty('backdrop-filter', 'blur(3px)', 'important');
                    el.style.setProperty('transition', 'all 0.3s ease', 'important');
                    
                    el.readOnly = true;
                    
                } catch (error) {
                    console.error("ApexConsole: Error setting up styles:", error);
                }
            };
            
            // Enhanced execution handler with error checking
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                try {
                    if (message && message.console_text && this.consoleWidget) {
                        const consoleText = message.console_text[0];
                        const theme = message.theme ? message.theme[0] : "matrix";
                        const dataType = message.data_type ? message.data_type[0] : "unknown";
                        
                        // Update content safely
                        if (this.consoleWidget.value !== undefined) {
                            this.consoleWidget.value = consoleText;
                        }
                        
                        // Apply theme safely
                        this.applyConsoleTheme(theme);
                        
                        // Update node title safely
                        if (this.title !== undefined) {
                            this.title = `Apex Console 🎯 [${dataType}]`;
                        }
                        
                        // Auto-scroll safely
                        setTimeout(() => {
                            try {
                                if (this.consoleWidget && this.consoleWidget.inputEl) {
                                    this.consoleWidget.inputEl.scrollTop = this.consoleWidget.inputEl.scrollHeight;
                                }
                            } catch (scrollError) {
                                console.warn("ApexConsole: Auto-scroll failed:", scrollError);
                            }
                        }, 50);
                        
                        // Glow effect safely
                        this.addGlowEffect();
                    }
                } catch (error) {
                    console.error("ApexConsole: Error in onExecuted:", error);
                }
                
                return onExecuted?.apply(this, arguments);
            };
            
            // Safe glow effect method
            nodeType.prototype.addGlowEffect = function() {
                try {
                    if (this.consoleWidget && this.consoleWidget.inputEl && this.consoleWidget.inputEl.style) {
                        this.consoleWidget.inputEl.style.animation = "consoleGlow 0.8s ease-out";
                        setTimeout(() => {
                            if (this.consoleWidget && this.consoleWidget.inputEl && this.consoleWidget.inputEl.style) {
                                this.consoleWidget.inputEl.style.animation = "";
                            }
                        }, 800);
                    }
                } catch (error) {
                    console.warn("ApexConsole: Glow effect failed:", error);
                }
            };
            
            // Safe theme application
            nodeType.prototype.applyConsoleTheme = function(theme) {
                try {
                    if (!this.consoleWidget || !this.consoleWidget.inputEl || !this.consoleWidget.inputEl.style) {
                        return;
                    }
                    
                    const themes = {
                        matrix: { 
                            bg: "linear-gradient(135deg, #001100 0%, #003300 50%, #001100 100%)", 
                            color: "#00ff41", 
                            border: "#00ff41" 
                        },
                        cyberpunk: { 
                            bg: "linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a0a1a 100%)", 
                            color: "#ff00ff", 
                            border: "#ff00ff" 
                        },
                        classic: { 
                            bg: "#000000", 
                            color: "#00ff00", 
                            border: "#00ff00" 
                        },
                        dracula: { 
                            bg: "linear-gradient(135deg, #282a36 0%, #44475a 50%, #282a36 100%)", 
                            color: "#f8f8f2", 
                            border: "#bd93f9" 
                        },
                        neon: { 
                            bg: "linear-gradient(135deg, #001122 0%, #002244 50%, #001122 100%)", 
                            color: "#00ffff", 
                            border: "#00ffff" 
                        }
                    };
                    
                    const t = themes[theme] || themes.matrix;
                    const el = this.consoleWidget.inputEl;
                    
                    el.style.setProperty('background', t.bg, 'important');
                    el.style.setProperty('color', t.color, 'important');
                    el.style.setProperty('border-color', t.border, 'important');
                    el.style.setProperty('box-shadow', `0 0 20px ${t.border}40, inset 0 0 20px rgba(0, 0, 0, 0.8)`, 'important');
                    
                } catch (error) {
                    console.warn("ApexConsole: Theme application failed:", error);
                }
            };
            
            // Safe resize handler
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                try {
                    if (onResize) {
                        onResize.apply(this, arguments);
                    }
                    
                    // Adjust console widget size safely
                    if (this.consoleWidget && this.consoleWidget.inputEl && this.consoleWidget.inputEl.style && size && size[1]) {
                        const newHeight = Math.max(120, size[1] - 200);
                        this.consoleWidget.inputEl.style.setProperty('height', `${newHeight}px`, 'important');
                    }
                } catch (error) {
                    console.warn("ApexConsole: Resize failed:", error);
                }
            };
        }
    }
});
