"""
Development Tasks for GuardRAG
Run with: python tasks.py <task_name>
"""

import subprocess
import sys
import os
from pathlib import Path


def run_cmd(cmd, description=""):
    """Run a command with proper error handling."""
    if description:
        print(f"🔄 {description}")

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}")
        sys.exit(1)
    else:
        print(f"✅ Success")


def get_python():
    """Get the correct Python executable."""
    venv_python = Path(".venv/Scripts/python.exe")
    return str(venv_python) if venv_python.exists() else sys.executable


def task_install():
    """Install dependencies."""
    print("📦 Installing dependencies...")
    run_cmd("uv sync", "Syncing dependencies with uv")


def task_install_dev():
    """Install development dependencies."""
    print("📦 Installing development dependencies...")
    run_cmd("uv sync --extra dev", "Installing dev dependencies")


def task_test():
    """Run all tests."""
    python = get_python()
    run_cmd(f"{python} run_tests.py", "Running full test suite")


def task_test_unit():
    """Run unit tests only."""
    python = get_python()
    run_cmd(f"{python} -m pytest tests/test_comprehensive.py -v", "Running unit tests")


def task_test_api():
    """Run API tests only."""
    python = get_python()
    run_cmd(f"{python} -m pytest tests/test_api.py -v", "Running API tests")


def task_test_coverage():
    """Run tests with coverage."""
    python = get_python()
    run_cmd(
        f"{python} -m pytest --cov=src --cov-report=html --cov-report=term",
        "Running tests with coverage",
    )


def task_format():
    """Format code."""
    python = get_python()
    run_cmd(f"{python} -m black src/ tests/", "Formatting code with Black")
    run_cmd(f"{python} -m isort src/ tests/", "Sorting imports with isort")


def task_lint():
    """Lint code."""
    python = get_python()
    run_cmd(f"{python} -m flake8 src/", "Linting with Flake8")
    run_cmd(f"{python} -m mypy src/", "Type checking with MyPy")


def task_clean():
    """Clean up generated files."""
    print("🧹 Cleaning up...")

    # Remove Python cache
    run_cmd(
        "find . -type d -name __pycache__ -exec rm -rf {} +", "Removing Python cache"
    )

    # Remove coverage files
    if Path("htmlcov").exists():
        run_cmd("rm -rf htmlcov", "Removing coverage HTML")
    if Path(".coverage").exists():
        run_cmd("rm .coverage", "Removing coverage data")

    # Remove pytest cache
    if Path(".pytest_cache").exists():
        run_cmd("rm -rf .pytest_cache", "Removing pytest cache")


def task_dev_setup():
    """Set up development environment."""
    print("🚀 Setting up development environment...")
    task_install_dev()
    task_format()
    task_test()
    print("✅ Development environment ready!")


def task_start():
    """Start the GuardRAG server."""
    python = get_python()
    print("🚀 Starting GuardRAG server...")
    run_cmd(f"{python} main.py", "Starting server")


def task_docker_build():
    """Build Docker image."""
    run_cmd("docker-compose build", "Building Docker image")


def task_docker_up():
    """Start all services with Docker."""
    run_cmd("docker-compose up -d", "Starting services with Docker")


def task_docker_down():
    """Stop all Docker services."""
    run_cmd("docker-compose down", "Stopping Docker services")


def task_docker_logs():
    """Show Docker logs."""
    run_cmd("docker-compose logs -f guardrag", "Showing Docker logs")


def task_help():
    """Show available tasks."""
    print("🛠️  Available tasks:")
    print()
    tasks = {
        "install": "Install dependencies",
        "install-dev": "Install development dependencies",
        "test": "Run all tests",
        "test-unit": "Run unit tests only",
        "test-api": "Run API tests only",
        "test-coverage": "Run tests with coverage",
        "format": "Format code with Black and isort",
        "lint": "Lint code with Flake8 and MyPy",
        "clean": "Clean up generated files",
        "dev-setup": "Complete development setup",
        "start": "Start GuardRAG server",
        "docker-build": "Build Docker image",
        "docker-up": "Start services with Docker",
        "docker-down": "Stop Docker services",
        "docker-logs": "Show Docker logs",
        "help": "Show this help message",
    }

    for task, description in tasks.items():
        print(f"  {task:<15} - {description}")

    print()
    print("Usage: python tasks.py <task_name>")


def main():
    """Main task runner."""
    if len(sys.argv) < 2:
        task_help()
        return

    task_name = sys.argv[1].replace("-", "_")
    task_func = f"task_{task_name}"

    if hasattr(sys.modules[__name__], task_func):
        getattr(sys.modules[__name__], task_func)()
    else:
        print(f"❌ Unknown task: {sys.argv[1]}")
        print()
        task_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
