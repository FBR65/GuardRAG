#!/usr/bin/env python3
"""
GuardRAG System Diagnostics
Überprüft alle Systemabhängigkeiten und gibt detaillierte Informationen aus
"""

import os
import sys
import socket
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrag.diagnostics")


def check_port(host: str, port: int, service_name: str) -> bool:
    """Check if a port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            if result == 0:
                print(f"✅ {service_name} ({host}:{port}) - ERREICHBAR")
                return True
            else:
                print(f"❌ {service_name} ({host}:{port}) - NICHT ERREICHBAR")
                return False
    except Exception as e:
        print(f"❌ {service_name} ({host}:{port}) - FEHLER: {e}")
        return False


def check_environment_variables():
    """Check important environment variables."""
    print("\n🔧 UMGEBUNGSVARIABLEN:")
    print("=" * 50)

    important_vars = {
        "LLM_ENDPOINT": "http://localhost:11434",
        "LLM_MODEL": "qwen2.5:latest",
        "COLPALI_MODEL": "vidore/colqwen2.5-v0.2",
        "DEVICE": "cuda",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
    }

    for var, default in important_vars.items():
        value = os.getenv(var, default)
        print(f"📝 {var}: {value}")


def check_python_packages():
    """Check if required Python packages are available."""
    print("\n📦 PYTHON PAKETE:")
    print("=" * 50)

    required_packages = [
        "gradio",
        "torch",
        "transformers",
        "qdrant_client",
        "colpali_engine",
        "fastapi",
        "pydantic_ai",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - INSTALLIERT")
        except ImportError:
            print(f"❌ {package} - FEHLT")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n💡 Fehlende Pakete installieren:")
        print(f"pip install {' '.join(missing_packages)}")

    return len(missing_packages) == 0


def check_gpu_availability():
    """Check GPU availability."""
    print("\n🖥️ GPU VERFÜGBARKEIT:")
    print("=" * 50)

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA verfügbar")
            print(f"📊 GPU Anzahl: {gpu_count}")
            print(f"🎯 Aktuelle GPU: {gpu_name}")
            print(
                f"💾 GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB"
            )
        else:
            print(f"⚠️ CUDA nicht verfügbar - CPU wird verwendet")
    except ImportError:
        print(f"❌ PyTorch nicht installiert")
    except Exception as e:
        print(f"❌ GPU-Check Fehler: {e}")


def check_disk_space():
    """Check available disk space."""
    print("\n💾 SPEICHERPLATZ:")
    print("=" * 50)

    try:
        import shutil

        total, used, free = shutil.disk_usage(".")

        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)

        print(f"📁 Gesamt: {total_gb:.1f} GB")
        print(f"📊 Verwendet: {used_gb:.1f} GB")
        print(f"💚 Frei: {free_gb:.1f} GB")

        if free_gb < 5:
            print(f"⚠️ Wenig Speicherplatz verfügbar!")
        else:
            print(f"✅ Ausreichend Speicherplatz")

    except Exception as e:
        print(f"❌ Speicherplatz-Check Fehler: {e}")


def check_docker_services():
    """Check if Docker services are running."""
    print("\n🐳 DOCKER SERVICES:")
    print("=" * 50)

    try:
        # Check if Docker is available
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"✅ Docker installiert: {result.stdout.strip()}")

            # Check running containers
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"📋 Laufende Container:")
                print(result.stdout)
            else:
                print(f"⚠️ Keine laufenden Container gefunden")
        else:
            print(f"❌ Docker nicht verfügbar")

    except FileNotFoundError:
        print(f"❌ Docker nicht installiert")
    except subprocess.TimeoutExpired:
        print(f"⏱️ Docker-Check Timeout")
    except Exception as e:
        print(f"❌ Docker-Check Fehler: {e}")


def check_services():
    """Check all required services."""
    print("\n🌐 SERVICE VERFÜGBARKEIT:")
    print("=" * 50)

    services = [
        ("Qdrant", "localhost", 6333),
        ("Ollama LLM", "localhost", 11434),
    ]

    all_ok = True
    for name, host, port in services:
        if not check_port(host, port, name):
            all_ok = False

    return all_ok


def provide_solutions():
    """Provide solutions for common problems."""
    print("\n💡 LÖSUNGSVORSCHLÄGE:")
    print("=" * 50)
    print("1. Qdrant starten:")
    print("   docker-compose up qdrant")
    print()
    print("2. Ollama starten:")
    print("   ollama serve")
    print()
    print("3. Alle Services starten:")
    print("   docker-compose up -d")
    print()
    print("4. Abhängigkeiten installieren:")
    print("   pip install -e .")
    print()
    print("5. GPU für CPU ersetzen (falls CUDA-Probleme):")
    print("   export DEVICE=cpu")


def main():
    """Run comprehensive system diagnostics."""
    print("🛡️ GuardRAG System Diagnostics")
    print("=" * 50)
    print(f"🐍 Python Version: {sys.version}")
    print(f"📂 Working Directory: {Path.cwd()}")
    print(f"🕒 Timestamp: {__import__('datetime').datetime.now()}")

    # Run all checks
    env_ok = True  # Environment variables are always "ok" as they have defaults
    packages_ok = check_python_packages()
    services_ok = check_services()

    check_environment_variables()
    check_gpu_availability()
    check_disk_space()
    check_docker_services()

    # Summary
    print("\n📋 ZUSAMMENFASSUNG:")
    print("=" * 50)
    if packages_ok and services_ok:
        print("🎉 Alle Systemchecks bestanden! GuardRAG sollte funktionieren.")
    else:
        print("⚠️ Einige Checks fehlgeschlagen. Siehe Lösungsvorschläge unten.")
        provide_solutions()

    return packages_ok and services_ok


if __name__ == "__main__":
    main()
