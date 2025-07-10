"""
Test Runner Script for GuardRAG
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd, capture_output=False, text=True, cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED")
            return False

    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

    return True


def main():
    """Main test runner."""
    print("🧪 GuardRAG Test Suite Runner")
    print("=" * 60)

    # Check if we're in a virtual environment
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"✅ Using virtual environment: {python_cmd}")
    else:
        python_cmd = sys.executable
        print(f"⚠️  Using system Python: {python_cmd}")

    # Test commands to run
    tests = [
        {
            "cmd": [python_cmd, "tests/test_setup.py"],
            "description": "Setup Verification",
        },
        {
            "cmd": [python_cmd, "tests/test_dangerous_query.py"],
            "description": "Dangerous Query Detection Tests",
        },
        {
            "cmd": [python_cmd, "tests/test_product_knowledge.py"],
            "description": "Product Knowledge Security Tests",
        },
        {
            "cmd": [python_cmd, "tests/test_output_score_validation.py"],
            "description": "Output Score Validation Tests",
        },
        {
            "cmd": [python_cmd, "tests/test_modern_guardrails.py"],
            "description": "Modern Guardrail Tests",
        },
        {
            "cmd": [python_cmd, "-m", "pytest", "tests/test_comprehensive.py", "-v"],
            "description": "Comprehensive Unit Tests",
        },
        {
            "cmd": [python_cmd, "-m", "pytest", "tests/test_api.py", "-v"],
            "description": "FastAPI Endpoint Tests",
        },
        {
            "cmd": [python_cmd, "-m", "pytest", "tests/test_guardrails.py", "-v"],
            "description": "Legacy Guardrail Tests",
        },
        {
            "cmd": [
                python_cmd,
                "-m",
                "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=term-missing",
            ],
            "description": "All Tests with Coverage",
        },
    ]

    # Optional: lint and format checks
    code_quality_checks = [
        {
            "cmd": [python_cmd, "-m", "black", "--check", "src/"],
            "description": "Code Formatting Check (Black)",
        },
        {
            "cmd": [python_cmd, "-m", "isort", "--check-only", "src/"],
            "description": "Import Sorting Check (isort)",
        },
        {
            "cmd": [python_cmd, "-m", "flake8", "src/"],
            "description": "Code Linting (Flake8)",
        },
    ]

    # Run basic setup test first
    print("\n🔍 Running Setup Verification...")
    if not run_command([python_cmd, "tests/test_setup.py"], "Setup Verification"):
        print("\n❌ Setup verification failed. Please check your environment.")
        return 1

    # Run main tests
    failed_tests = []

    for test in tests:
        if not run_command(test["cmd"], test["description"]):
            failed_tests.append(test["description"])

    # Run code quality checks (optional)
    print("\n🔍 Running Code Quality Checks...")
    for check in code_quality_checks:
        run_command(check["cmd"], check["description"])

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    if failed_tests:
        print(f"❌ {len(failed_tests)} test suite(s) failed:")
        for test in failed_tests:
            print(f"   - {test}")
        print("\n💡 Tip: Run individual test files to debug specific failures")
        return 1
    else:
        print("✅ All tests passed!")
        print("\n🎉 GuardRAG system is ready for deployment!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
