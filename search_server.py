#!/usr/bin/env python3
"""
search_server.py — 入口文件，实际逻辑在 server.py 及子模块中。
systemd service 运行此文件，内容委托给 server.py。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from server import Handler, PORT
from http.server import HTTPServer

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[search_server] 监听 127.0.0.1:{PORT}")
    server.serve_forever()
