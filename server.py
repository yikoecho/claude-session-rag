#!/usr/bin/env python3
"""
server.py — Flask-style HTTP server (实际用 http.server)，路由入口
监听 127.0.0.1:15200
"""

import sys
import os

# 让子模块能 import utils / index / rag
sys.path.insert(0, os.path.dirname(__file__))

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from utils.config import PORT, RECALL_ENABLED, RECALL_MODEL
from index.bm25 import build_bm25_index, bm25_search, reload_bm25_async
from index.vector import vector_search_raw  # noqa: F401 — triggers LanceDB init
from rag.hybrid import hybrid_search
from rag.recall import recall_agent

# 启动 BM25 索引
threading.Thread(target=build_bm25_index, daemon=True).start()

if RECALL_ENABLED:
    print(f"[search_server] Recall Agent 已启用，模型: {RECALL_MODEL}")
else:
    print(f"[search_server] ⚠ Recall Agent 未启用（缺少 OPENROUTER_API_KEY）")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]

        if parsed.path == "/bm25":
            if not query:
                self.send_response(400)
                self.end_headers()
                return
            top_k = int(params.get("top_k", [5])[0])
            try:
                result = bm25_search(query, top_k)
            except Exception as e:
                result = ""
                print(f"[bm25] 搜索出错: {e}")
            body = result.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/hybrid":
            if not query:
                self.send_response(400)
                self.end_headers()
                return
            top_k = int(params.get("top_k", [5])[0])
            try:
                results = hybrid_search(query, top_k)
            except Exception as e:
                results = []
                print(f"[hybrid] 出错: {e}")
            body = json.dumps({"results": results}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/reload_bm25":
            reload_bm25_async()
            body = b"reloading"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/search":
            # Deprecated: use /hybrid instead
            body = b'{"error": "deprecated", "message": "Use /hybrid instead of /search"}'
            self.send_response(410)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/recall":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length).decode("utf-8")

        try:
            data = json.loads(raw)
            prompt = data.get("prompt", "")
            candidates = data.get("candidates", [])
        except (json.JSONDecodeError, AttributeError):
            self.send_response(400)
            self.end_headers()
            return

        filtered = recall_agent(prompt, candidates)

        body = json.dumps({"results": filtered}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[search_server] 监听 127.0.0.1:{PORT}")
    server.serve_forever()
