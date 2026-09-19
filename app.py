"""
Vercel Serverless Entrypoint for NFL Analytics Hub
==================================================
Routes all frontend and API requests through FastAPI.
"""

import os
import sys
from urllib.parse import urlparse
from starlette.types import ASGIApp, Scope, Receive, Send

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.server import app as fastapi_app


class VercelPathMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
            if not matched_path:
                matched_path = headers.get(b"x-vercel-matched-path", b"").decode("utf-8", errors="ignore")
            
            if matched_path and not matched_path.endswith("app.py") and not matched_path.endswith("index.py"):
                parsed = urlparse(matched_path)
                scope["path"] = parsed.path
                if parsed.query and not scope.get("query_string"):
                    scope["query_string"] = parsed.query.encode("utf-8")
                    
        await self.app(scope, receive, send)


app = VercelPathMiddleware(fastapi_app)
handler = app
