#!/usr/bin/env python
"""Test runner script for the trading platform."""
import subprocess
import sys


def run_linting():
    """Run code linting with ruff."""
    print("🔍 Running linting with ruff...")
    result = subprocess.run(
        ["ruff", "check", ".", "--fix"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Linting passed")
    else:
        print("❌ Linting issues found:")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def run_type_checking():
    """Run type checking with mypy."""
    print("🔍 Running type checking with mypy...")
    result = subprocess.run(
        ["mypy", "backend", "services", "adapters", "api", "infra"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Type checking passed")
    else:
        print("❌ Type checking issues found:")
        print(result.stdout)
        print(result.stderr)

    return result.returncode == 0


def run_unit_tests():
    """Run unit tests."""
    print("🧪 Running unit tests...")
    result = subprocess.run(
        ["pytest", "tests/unit", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def run_integration_tests():
    """Run integration tests."""
    print("🧪 Running integration tests...")
    result = subprocess.run(
        ["pytest", "tests/integration", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def run_all_tests():
    """Run all tests."""
    print("🧪 Running all tests...")
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short", "--cov=.", "--cov-report=term-missing"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"

    print(f"🚀 Starting test suite: {test_type}")
    print("=" * 50)

    success = True

    if test_type in ["all", "lint"]:
        success &= run_linting()
        print()

    if test_type in ["all", "type"]:
        success &= run_type_checking()
        print()

    if test_type == "unit":
        success &= run_unit_tests()
    elif test_type == "integration":
        success &= run_integration_tests()
    elif test_type == "all":
        success &= run_all_tests()

    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
