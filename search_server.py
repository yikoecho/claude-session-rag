#!/usr/bin/env python3
"""
search_server.py — entry point. All logic lives in server.py and its submodules.
Run:  python search_server.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from http.server import HTTPServer
from server import Handler, PORT

if __name__ == "__main__":
    # Bind loopback only — no auth; never expose this port.
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[search_server] listening on 127.0.0.1:{PORT}")
    server.serve_forever()
