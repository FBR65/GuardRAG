#!/usr/bin/env python3
"""
GuardRAG Frontend Server
========================

Ein einfacher HTTP-Server zum Bereitstellen des GuardRAG-Frontends.
Das Frontend kommuniziert direkt mit der FastAPI (Port 8000).

Verwendung:
    python start_frontend.py

Der Server läuft standardmäßig auf Port 8080.
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler


class GuardRAGFrontendHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler für das GuardRAG Frontend."""
    
    def __init__(self, *args, **kwargs):
        # Frontend-Verzeichnis als Document Root setzen
        self.frontend_dir = Path(__file__).parent / "frontend"
        super().__init__(*args, directory=str(self.frontend_dir), **kwargs)
    
    def end_headers(self):
        """Add CORS headers and disable caching."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Customize log messages."""
        print(f"[Frontend] {self.address_string()} - {format % args}")


def check_backend_connection():
    """Prüft, ob das FastAPI-Backend erreichbar ist."""
    import urllib.request
    import urllib.error
    
    try:
        with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
            if response.status == 200:
                return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        pass
    return False


def open_browser_delayed(url, delay=2):
    """Öffnet den Browser nach einer kurzen Verzögerung."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"[Frontend] Browser geöffnet: {url}")
    except Exception as e:
        print(f"[Frontend] Browser konnte nicht geöffnet werden: {e}")


def main():
    """Hauptfunktion zum Starten des Frontend-Servers."""
    # Konfiguration
    HOST = '127.0.0.1'
    PORT = 8080
    
    # Frontend-Verzeichnis prüfen
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend-Verzeichnis nicht gefunden: {frontend_dir}")
        print("Stellen Sie sicher, dass das Frontend-Verzeichnis existiert.")
        sys.exit(1)
    
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        print(f"❌ index.html nicht gefunden: {index_file}")
        sys.exit(1)
    
    print("🚀 GuardRAG Frontend Server wird gestartet...")
    print(f"📁 Frontend-Verzeichnis: {frontend_dir.absolute()}")
    
    # Backend-Verbindung prüfen
    backend_available = check_backend_connection()
    if backend_available:
        print("✅ FastAPI Backend erreichbar (Port 8000)")
    else:
        print("⚠️  FastAPI Backend nicht erreichbar (Port 8000)")
        print("   Starten Sie das Backend mit: python main.py")
    
    try:
        # HTTP-Server erstellen
        server = HTTPServer((HOST, PORT), GuardRAGFrontendHandler)
        server_url = f"http://{HOST}:{PORT}"
        
        print(f"🌐 Frontend-Server läuft auf: {server_url}")
        print(f"📊 Backend-API erwaret auf: http://localhost:8000")
        print("🛑 Zum Beenden: Strg+C")
        print("-" * 50)
        
        # Browser in separatem Thread öffnen
        browser_thread = threading.Thread(
            target=open_browser_delayed, 
            args=(server_url,),
            daemon=True
        )
        browser_thread.start()
        
        # Server starten
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server wird beendet...")
        server.shutdown()
        print("✅ Frontend-Server beendet")
        
    except OSError as e:
        if e.errno == 10048:  # Windows: Port already in use
            print(f"❌ Port {PORT} ist bereits belegt")
            print(f"Beenden Sie andere Anwendungen auf Port {PORT} oder verwenden Sie einen anderen Port")
        else:
            print(f"❌ Server-Fehler: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
