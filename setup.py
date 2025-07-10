#!/usr/bin/env python3
"""
GuardRAG Setup Script
Automatisiert die Einrichtung der GuardRAG-Umgebung.
"""

import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: str, shell: bool = True) -> bool:
    """Execute a command and return success status."""
    try:
        subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.10+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ ist erforderlich")
        print(f"Aktuelle Version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} erkannt")
    return True


def check_uv():
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        print(f"✅ uv gefunden: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ uv nicht gefunden")
        print("Bitte installieren Sie uv: https://docs.astral.sh/uv/")
        return False


def check_ollama():
    """Check if Ollama is available."""
    try:
        subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        print("✅ Ollama ist verfügbar")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  Ollama nicht gefunden")
        print("Bitte installieren Sie Ollama: https://ollama.com/")
        return False


def setup_environment():
    """Set up the development environment."""
    print("🔧 Einrichtung der Entwicklungsumgebung...")

    # Create virtual environment
    if not run_command("uv venv"):
        return False

    # Install dependencies
    if not run_command("uv pip install -e ."):
        return False

    return True


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ .env Datei aus Vorlage erstellt")
    elif env_file.exists():
        print("✅ .env Datei bereits vorhanden")
    else:
        print("⚠️  .env.example nicht gefunden")


def setup_ollama_models():
    """Download required Ollama models."""
    models = ["qwen2.5:latest", "granite3.3:8b"]

    print("📦 Herunterladen der Ollama-Modelle...")

    for model in models:
        print(f"Herunterladen: {model}")
        if run_command(f"ollama pull {model}"):
            print(f"✅ {model} erfolgreich installiert")
        else:
            print(f"⚠️  Fehler beim Herunterladen von {model}")


def create_directories():
    """Create necessary directories."""
    directories = ["uploads", "logs"]

    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✅ Verzeichnis erstellt: {directory}")


def main():
    """Main setup function."""
    print("🚀 GuardRAG Setup")
    print("=" * 50)

    # Check prerequisites
    print("\n📋 Überprüfung der Voraussetzungen...")
    if not check_python_version():
        sys.exit(1)

    if not check_uv():
        sys.exit(1)

    ollama_available = check_ollama()

    # Setup environment
    print("\n🔧 Einrichtung der Umgebung...")
    if not setup_environment():
        print("❌ Fehler bei der Umgebungseinrichtung")
        sys.exit(1)

    # Create configuration
    print("\n⚙️  Konfiguration...")
    create_env_file()
    create_directories()

    # Setup Ollama models if available
    if ollama_available:
        print("\n📦 Modell-Setup...")
        setup_ollama_models()

    print("\n🎉 Setup abgeschlossen!")
    print("\nNächste Schritte:")
    print("1. Aktivieren Sie die virtuelle Umgebung:")

    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")

    print("2. Bearbeiten Sie die .env Datei nach Bedarf")
    print("3. Starten Sie den Server:")
    print("   python main.py")
    print("\n📖 Dokumentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
