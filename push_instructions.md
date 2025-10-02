# Push Instructions for Version Updates (ComfyUI Apex Artist)

To publish a new version to the ComfyUI Registry:

1. Update the version number in all required files:
   - `pyproject.toml` (version = "X.Y.Z")
   - `manifest.json` ("version": "X.Y.Z")
   - `custom_nodes.json` ("version": "X.Y.Z")
   - `comfyui.yaml` (version: "X.Y.Z")

2. Stage all changes:
   ```powershell
   git add .
   ```

3. Commit with a clear message:
   ```powershell
   git commit -m "🚀 Version X.Y.Z - [describe your changes]"
   ```

4. Push to the main branch:
   ```powershell
   git push origin main
   ```

5. GitHub Actions will automatically publish to the ComfyUI Registry.

---
**Troubleshooting:**
- If you see errors about user.name or user.email, set them with:
  ```powershell
  git config --global user.name "Your Name"
  git config --global user.email "your@email.com"
  ```
- If you see JSON errors, check for missing/extra commas or duplicate keys.
