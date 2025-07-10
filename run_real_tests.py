"""
Test Launcher für echte GuardRAG Tests
Prüft Services und führt dann echte End-to-End Tests durch
"""

import os
import sys
import requests
import subprocess
from pathlib import Path


def check_qdrant_running():
    """Prüfe ob Qdrant läuft."""
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        response = requests.get(f"http://{qdrant_host}:{qdrant_port}/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_llm_endpoint():
    """Prüfe ob LLM Endpoint erreichbar ist."""
    try:
        llm_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:11434/v1")
        response = requests.get(f"{llm_endpoint}/models", timeout=10)
        return response.status_code in [
            200,
            404,
        ]  # 404 ist auch ok, bedeutet Server läuft
    except Exception:
        return False


def print_service_status():
    """Zeige Status aller Services."""
    print("=== SERVICE STATUS CHECK ===")

    qdrant_status = check_qdrant_running()
    llm_status = check_llm_endpoint()

    print(f"Qdrant: {'✓ Running' if qdrant_status else '✗ Not available'}")
    print(f"LLM Endpoint: {'✓ Running' if llm_status else '✗ Not available'}")

    if not qdrant_status:
        print("\n⚠ WARNUNG: Qdrant nicht verfügbar!")
        print("  Starten Sie Qdrant mit: docker compose up qdrant -d")

    if not llm_status:
        print("\n⚠ WARNUNG: LLM Endpoint nicht verfügbar!")
        print("  Stellen Sie sicher, dass Ollama läuft und das Modell verfügbar ist")

    return qdrant_status, llm_status


def run_real_tests():
    """Führe die echten Tests aus."""
    print("\n=== RUNNING REAL INTEGRATION TESTS ===")

    # Führe Tests aus
    cmd = ["python", "-m", "pytest", "tests/test_real_integration.py", "-v", "-s"]
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def run_manual_system_check():
    """Führe manuelle Systemprüfung durch."""
    print("\n=== MANUAL SYSTEM CHECK ===")

    try:
        # Import hier um sicherzustellen dass FastAPI app verfügbar ist
        sys.path.append(str(Path(__file__).parent.parent))
        from tests.test_real_integration import run_manual_test

        run_manual_test()
        return True
    except Exception as e:
        print(f"Manual test failed: {e}")
        return False


def main():
    """Hauptfunktion."""
    print("GuardRAG Real System Test Launcher")
    print("=" * 50)

    # 1. Service Status prüfen
    qdrant_ok, llm_ok = print_service_status()

    # 2. Manueller System Check
    print("\n" + "=" * 50)
    manual_ok = run_manual_system_check()

    # 3. Echte Tests ausführen
    print("\n" + "=" * 50)
    if qdrant_ok and llm_ok:
        tests_ok = run_real_tests()

        print("\n=== FINAL RESULTS ===")
        print(f"Services: {'✓' if qdrant_ok and llm_ok else '✗'}")
        print(f"Manual Check: {'✓' if manual_ok else '✗'}")
        print(f"Integration Tests: {'✓' if tests_ok else '✗'}")

        if tests_ok:
            print("\n🎉 ALLE ECHTEN TESTS BESTANDEN!")
            print("Das System funktioniert mit echten Daten!")
        else:
            print("\n❌ TESTS FEHLGESCHLAGEN!")
            print("Prüfen Sie die Logs für Details.")
    else:
        print("\n⚠ Services nicht verfügbar - Tests übersprungen")
        print("Starten Sie die erforderlichen Services und versuchen Sie es erneut.")

        print("\nZum Starten der Services:")
        print("docker compose up qdrant -d")
        print("# Ollama separat starten")


if __name__ == "__main__":
    main()
