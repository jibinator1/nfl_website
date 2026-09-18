"""
NFL Analytics Hub - App Launcher
================================
Launches the FastAPI backend on http://localhost:8000 and automatically
opens the interactive dashboard in your default browser.
"""

import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING NFL AI ANALYTICS & MATCHUP HUB")
    print("Zero API Keys Required • 100% Open-Source NFL Data")
    print("=" * 60)
    print("Dashboard will open automatically at: http://localhost:8000")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 60)

    # Launch browser on separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI server
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
