import { app } from "../../scripts/app.js";

// Load CSS file
const cssPath = new URL("./apex_console.css", import.meta.url).href;
const link = document.createElement("link");
link.rel = "stylesheet";
link.type = "text/css";
link.href = cssPath;
document.head.appendChild(link);

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create embedded console widget
                const consoleWidget = this.addDOMWidget(
                    "console_display",
                    "div",
                    document.createElement("div")
                );
                
                // Initial setup
                consoleWidget.element.className = "apex-console theme-matrix";
                consoleWidget.element.innerHTML = `
                    <div class="apex-console-header">
                        🎯 APEX CONSOLE INITIALIZING...
                    </div>
                    <div class="apex-console-body">
                        <div style="color: #666; font-style: italic; text-align: center; margin-top: 50px;">
                            🔌 Connect data inputs to see console output
                        </div>
                    </div>
                `;
                
                // Set widget size
                consoleWidget.element.style.width = "600px";
                consoleWidget.element.style.height = "350px";
                consoleWidget.element.style.resize = "both";
                consoleWidget.element.style.overflow = "hidden";
                
                return result;
            };
            
            // Handle execution and update console
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const consoleWidget = this.widgets?.find(w => w.name === "console_display");
                
                if (consoleWidget && message.console_text) {
                    const consoleText = message.console_text[0];
                    const theme = message.theme ? message.theme[0] : "matrix";
                    
                    // Apply theme
                    consoleWidget.element.className = `apex-console theme-${theme}`;
                    
                    // Format text with HTML styling
                    const formattedText = consoleText
                        .replace(/\n/g, '<br>')
                        .replace(/(⏰ \d{2}:\d{2}:\d{2}\.\d{3})/g, '<span class="apex-timestamp">$1</span>')
                        .replace(/(🎯|ℹ️|⚠️|❌|✅|🔍)/g, '<span class="apex-log-icon">$1</span>')
                        .replace(/(📝|🔢|📊|🔘|🖼️|🎨|🎛️|🤖|📋)/g, '<span class="apex-data-icon">$1</span>')
                        .replace(/(=+)/g, '<span class="apex-separator">$1</span>')
                        .replace(/(─+)/g, '<span class="apex-divider">$1</span>');
                    
                    // Update console content
                    consoleWidget.element.innerHTML = `
                        <div class="apex-console-content">
                            ${formattedText}
                        </div>
                    `;
                    
                    // Auto-scroll to bottom
                    const content = consoleWidget.element.querySelector('.apex-console-content');
                    if (content) {
                        content.scrollTop = content.scrollHeight;
                    }
                }
                
                return onExecuted?.apply(this, arguments);
            };
        }
    }
});
