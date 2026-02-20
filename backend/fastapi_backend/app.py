"""
Entry point for Hugging Face Spaces deployment.
This file allows HF Spaces to find the app without needing uvicorn in the startup command.
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
