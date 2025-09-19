import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ApexArtist.Console",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ApexConsole") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Simple text widget
                this.consoleWidget = this.addWidget("text", "output", "🎯 Waiting for data...", () => {}, {
                    multiline: true,
                    serialize: false
                });
                
                // Basic styling
                setTimeout(() => {
                    if (this.consoleWidget?.inputEl) {
                        const el = this.consoleWidget.inputEl;
                        el.style.backgroundColor = "#001100";
                        el.style.color = "#00ff00";
                        el.style.fontFamily = "monospace";
                        el.style.fontSize = "10px";
                        el.style.height = "200px";
                        el.style.border = "1px solid #00ff00";
                        el.readOnly = true;
                    }
                }, 100);
                
                this.size = [400, 300];
                return result;
            };
            
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (message?.text && this.consoleWidget) {
                    this.consoleWidget.value = message.text[0];
                }
                return onExecuted?.apply(this, arguments);
            };
        }
    }
});
