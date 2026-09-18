"""
Vercel Serverless Function Entry Point for NFL Analytics Hub
"""

import sys
import os

# Inject project root to sys.path so backend imports resolve seamlessly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.server import app

# Expose app as WSGI/ASGI handler for Vercel
handler = app
