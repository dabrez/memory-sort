#!/usr/bin/env python3
"""Simple HTTP server for the conversations viewer. Run from this directory."""
import http.server
import os
import sys

PORT = 8765
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

print(f"Conversations viewer → http://localhost:{PORT}")
print("Press Ctrl+C to stop.")
with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
