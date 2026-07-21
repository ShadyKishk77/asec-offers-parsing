import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    os.chdir(DIRECTORY)
    print(f"=" * 70)
    print(f"  ASEC Document Intelligence Platform")
    print(f"  Serving Premium Front-End Landing Page on: http://localhost:{PORT}")
    print(f"=" * 70)
    
    url = f"http://localhost:{PORT}"
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
