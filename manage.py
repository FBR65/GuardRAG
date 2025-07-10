#!/usr/bin/env python3
"""
GuardRAG Management Script
CLI-Tool für die Verwaltung des GuardRAG-Systems.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx


class GuardRAGManager:
    """Management interface for GuardRAG system."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def health_check(self):
        """Check system health."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("🟢 System ist gesund")
                print(f"Status: {data.get('status', 'unknown')}")

                if "components" in data:
                    print("\nKomponenten-Status:")
                    for component, status in data["components"].items():
                        if isinstance(status, dict):
                            comp_status = status.get("status", "unknown")
                            icon = "🟢" if comp_status == "healthy" else "🔴"
                            print(f"  {icon} {component}: {comp_status}")
                        else:
                            print(f"  ℹ️  {component}: {status}")

                return True
            else:
                print(f"🔴 Gesundheitsprüfung fehlgeschlagen: {response.status_code}")
                return False

        except Exception as e:
            print(f"🔴 Verbindungsfehler: {e}")
            return False

    async def upload_document(self, file_path: str):
        """Upload a document to the system."""
        file_path = Path(file_path)

        if not file_path.exists():
            print(f"🔴 Datei nicht gefunden: {file_path}")
            return False

        print(f"📤 Hochladen: {file_path}")

        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                response = await self.client.post(
                    f"{self.base_url}/upload-document", files=files
                )

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Dokument erfolgreich verarbeitet")
                print(f"   Seiten: {data.get('pages_processed', 'unknown')}")
                print(f"   Datei-ID: {data.get('file_id', 'unknown')}")
                return True
            else:
                print(f"🔴 Upload fehlgeschlagen: {response.status_code}")
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    error_data = response.json()
                    print(f"   Fehler: {error_data.get('detail', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"🔴 Upload-Fehler: {e}")
            return False

    async def query_documents(self, query: str, max_results: int = 5):
        """Query the RAG system."""
        print(f"🔍 Abfrage: {query}")

        try:
            request_data = {"query": query, "max_results": max_results}

            response = await self.client.post(
                f"{self.base_url}/rag-query", json=request_data
            )

            if response.status_code == 200:
                data = response.json()

                print(f"\n✅ Antwort (Vertrauen: {data.get('confidence', 0):.2f}):")
                print(f"{data.get('answer', 'Keine Antwort')}")

                if data.get("sources"):
                    print(f"\n📄 Quellen ({len(data['sources'])}):")
                    for i, source in enumerate(data["sources"], 1):
                        print(
                            f"  {i}. Seite {source.get('page_number', '?')} "
                            f"(Score: {source.get('score', 0):.3f})"
                        )
                        if source.get("explanation"):
                            print(f"     {source['explanation']}")

                if data.get("warnings"):
                    print(f"\n⚠️  Warnungen:")
                    for warning in data["warnings"]:
                        print(f"  - {warning}")

                print(f"\n⏱️  Verarbeitungszeit: {data.get('processing_time', 0):.2f}s")
                return True

            else:
                print(f"🔴 Abfrage fehlgeschlagen: {response.status_code}")
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    error_data = response.json()
                    print(f"   Fehler: {error_data.get('detail', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"🔴 Abfrage-Fehler: {e}")
            return False

    async def system_status(self):
        """Get detailed system status."""
        try:
            response = await self.client.get(f"{self.base_url}/system-status")

            if response.status_code == 200:
                data = response.json()

                print("📊 System-Status:")
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   Geladene Dokumente: {data.get('documents_loaded', 0)}")

                guardrails = data.get("guardrails_enabled", {})
                print(
                    f"   Input Guardrails: {'✅' if guardrails.get('input') else '❌'}"
                )
                print(
                    f"   Output Guardrails: {'✅' if guardrails.get('output') else '❌'}"
                )

                # Detailed component information
                components = data.get("components", {})
                if "colpali_status" in components:
                    colpali = components["colpali_status"]
                    print(f"\n🔍 COLPALI:")
                    print(f"   Modell: {colpali.get('model_name', 'unknown')}")
                    print(f"   Seiten indiziert: {colpali.get('num_pages', 0)}")
                    print(
                        f"   Speicherverbrauch: {colpali.get('memory_usage_mb', 0):.1f} MB"
                    )

                return True

            else:
                print(f"🔴 Status-Abfrage fehlgeschlagen: {response.status_code}")
                return False

        except Exception as e:
            print(f"🔴 Status-Fehler: {e}")
            return False

    async def clear_documents(self):
        """Clear all loaded documents."""
        try:
            response = await self.client.delete(f"{self.base_url}/clear-documents")

            if response.status_code == 200:
                print("✅ Alle Dokumente erfolgreich gelöscht")
                return True
            else:
                print(f"🔴 Löschen fehlgeschlagen: {response.status_code}")
                return False

        except Exception as e:
            print(f"🔴 Lösch-Fehler: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GuardRAG Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s health                           # Gesundheitsprüfung
  %(prog)s upload document.pdf              # Dokument hochladen  
  %(prog)s query "What is the methodology?" # Abfrage stellen
  %(prog)s status                           # System-Status anzeigen
  %(prog)s clear                            # Alle Dokumente löschen
        """,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="GuardRAG server URL (default: http://localhost:8000)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Verfügbare Befehle")

    # Health check command
    subparsers.add_parser("health", help="System-Gesundheitsprüfung")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Dokument hochladen")
    upload_parser.add_argument("file", help="Pfad zur Datei")

    # Query command
    query_parser = subparsers.add_parser("query", help="RAG-Abfrage stellen")
    query_parser.add_argument("text", help="Abfragetext")
    query_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximale Anzahl Ergebnisse (default: 5)",
    )

    # Status command
    subparsers.add_parser("status", help="Detaillierter System-Status")

    # Clear command
    subparsers.add_parser("clear", help="Alle Dokumente löschen")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = GuardRAGManager(args.url)

    try:
        if args.command == "health":
            success = await manager.health_check()
        elif args.command == "upload":
            success = await manager.upload_document(args.file)
        elif args.command == "query":
            success = await manager.query_documents(args.text, args.max_results)
        elif args.command == "status":
            success = await manager.system_status()
        elif args.command == "clear":
            success = await manager.clear_documents()
        else:
            print(f"🔴 Unbekannter Befehl: {args.command}")
            success = False

        sys.exit(0 if success else 1)

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
