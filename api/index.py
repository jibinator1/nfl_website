"""
Vercel Serverless Function Entry Point for NFL Analytics Hub
============================================================
Handles automatic path and query restoration for Vercel rewrites and x-matched-path headers.
"""

import sys
import os
from urllib.parse import urlparse
from starlette.types import ASGIApp, Scope, Receive, Send

# Inject project root to sys.path so backend imports resolve seamlessly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.server import app as fastapi_app


class VercelPathMiddleware:
    """
    Normalizes Vercel serverless request paths.
    If Vercel rewrites /api/(.*) to /api/index.py, this restores the original requested path and query.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
            if not matched_path:
                matched_path = headers.get(b"x-vercel-matched-path", b"").decode("utf-8", errors="ignore")
            
            # If x-matched-path contains the real client route (e.g. /api/status or /api/full-schedule?season=2026)
            if matched_path and not matched_path.endswith("index.py"):
                parsed = urlparse(matched_path)
                scope["path"] = parsed.path
                if parsed.query and not scope.get("query_string"):
                    scope["query_string"] = parsed.query.encode("utf-8")
                
        await self.app(scope, receive, send)


# Expose 'app' for Vercel Python ASGI serverless runtime
app = VercelPathMiddleware(fastapi_app)
handler = app
