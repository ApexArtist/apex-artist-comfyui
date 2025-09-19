import { app } from "../../scripts/app.js";

// Register the console widget
app.registerExtension({
    name: "ApexArtist.Console",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create console display widget
                const consoleWidget = this.addDOMWidget(
                    "console_display", 
                    "div", 
                    document.createElement("div")
                );
                
                // Style the console
                consoleWidget.element.innerHTML = `
                    <div style="
                        background: linear-gradient(135deg, #0d1117 0%, #1a1a1a 100%);
                        color: #00ff41;
                        font-family: 'Fira Code', 'Courier New', monospace;
                        font-size: 11px;
                        padding: 15px;
                        border: 2px solid #00ff41;
                        border-radius: 8px;
                        height: 300px;
                        width: 500px;
                        overflow-y: auto;
                        white-space: pre-wrap;
                        line-height: 1.4;
                        box-shadow: 0 0 20px rgba(0, 255, 65, 0.3);
                        position: relative;
                    ">
                        <div style="color: #00ff88; font-weight: bold; margin-bottom: 10px;">
                            🎯 APEX CONSOLE READY
                        </div>
                        <div style="color: #666;">
                            Waiting for data inputs...
                        </div>
                    </div>
                `;
                
                // Update console on execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    if (message?.text) {
                        const consoleText = message.text[0];
                        consoleWidget.element.querySelector('div').innerHTML = 
                            consoleText.replace(/\n/g, '<br>');
                        
                        // Auto-scroll to bottom
                        consoleWidget.element.scrollTop = consoleWidget.element.scrollHeight;
                    }
                    return onExecuted?.apply(this, arguments);
                };
                
                return result;
            };
        }
    }
});
