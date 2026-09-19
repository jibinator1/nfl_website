"""
Vercel Serverless Function Entry Point for NFL Analytics Hub
============================================================
Handles automatic path restoration for Vercel rewrites and x-matched-path headers.
"""

import sys
import os
from urllib.parse import urlparse
from starlette.types import ASGIApp, Scope, Receive, Send

# Inject project root to sys.path so backend imports resolve seamlessly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.server import app


class VercelPathMiddleware:
    """
    Normalizes Vercel serverless request paths.
    If Vercel rewrites /api/(.*) to /api/index.py, this restores the original requested path.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
            
            if matched_path:
                parsed = urlparse(matched_path)
                scope["path"] = parsed.path
            elif scope.get("path", "").startswith("/api/index.py"):
                sub = scope["path"][len("/api/index.py"):]
                scope["path"] = sub if sub.startswith("/") else "/" + sub
            elif scope.get("path", "").startswith("/index.py"):
                sub = scope["path"][len("/index.py"):]
                scope["path"] = sub if sub.startswith("/") else "/" + sub
                
        await self.app(scope, receive, send)


# Wrap FastAPI application with path normalization for Vercel
handler = VercelPathMiddleware(app)
