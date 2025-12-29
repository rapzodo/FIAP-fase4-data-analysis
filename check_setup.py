#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def check_environment():
    print("🔍 Checking Multi-Agent Video Analysis System Configuration")
    print("=" * 70)

    issues = []
    warnings = []

    # Check Python version
    print("\n📍 Python Version:")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 10:
        print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"   ❌ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        issues.append("Python 3.10 or higher is required")

    # Check virtual environment
    print("\n📍 Virtual Environment:")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Running in virtual environment")
    else:
        print("   ⚠️  Not running in virtual environment")
        warnings.append("It's recommended to use a virtual environment")

    # Check required packages
    print("\n📍 Required Packages:")
    required_packages = [
        'crewai', 'opencv-python', 'face_recognition', 'deepface',
        'mediapipe', 'langchain_groq', 'langchain_ollama', 'pydantic'
    ]

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            issues.append(f"Missing package: {package}")

    # Check .env file
    print("\n📍 Configuration:")
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file exists")

        # Read .env and check key settings
        with open(env_file) as f:
            env_content = f.read()

        if "GROQ_API_KEY=your_groq_api_key_here" in env_content:
            print("   ⚠️  GROQ_API_KEY not configured (using default)")
            warnings.append("Set GROQ_API_KEY in .env or use Ollama")
        elif "GROQ_API_KEY=" in env_content and "gsk_" in env_content:
            print("   ✅ GROQ_API_KEY configured")

        if "USE_GROQ=false" in env_content.lower():
            print("   ℹ️  Configured to use Ollama")
        else:
            print("   ℹ️  Configured to use Groq")
    else:
        print("   ❌ .env file not found")
        issues.append("Run 'cp .env.example .env' to create configuration file")

    # Check input files
    print("\n📍 Input Files:")

    video_path = Path("tech-challenge/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")
    if video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Video file found ({size_mb:.1f} MB)")
    else:
        print("   ❌ Video file not found")
        issues.append("Video file missing: tech-challenge/*.mp4")

    pdf_path = Path("tech-challenge/Tech Challenge - IADT - Fase 4.pdf")
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"   ✅ PDF file found ({size_kb:.1f} KB)")
    else:
        print("   ❌ PDF file not found")
        issues.append("PDF file missing: tech-challenge/*.pdf")

    # Check output directory
    print("\n📍 Output Directory:")
    output_dir = Path("output")
    if output_dir.exists():
        print("   ✅ Output directory exists")
    else:
        print("   ⚠️  Output directory not found (will be created)")
        output_dir.mkdir(exist_ok=True)
        print("   ✅ Created output directory")

    # Check Ollama (if not using Groq)
    print("\n📍 LLM Availability:")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("   ✅ Ollama is running locally")
            models = response.json().get('models', [])
            if models:
                print(f"   ℹ️  Available models: {', '.join([m['name'] for m in models[:3]])}")
        else:
            print("   ⚠️  Ollama not responding")
            warnings.append("Ollama not available (will use Groq if configured)")
    except:
        print("   ⚠️  Ollama not running")
        warnings.append("Ollama not available (will use Groq if configured)")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    if not issues and not warnings:
        print("\n✅ All checks passed! You're ready to run the application.")
        print("\n▶️  Run: python main.py")
        return 0

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   • {warning}")

    if issues:
        print(f"\n❌ Found {len(issues)} issue(s) that need to be fixed:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Run './setup.sh' to fix most issues automatically")
        return 1

    print("\n✅ All critical checks passed, but review warnings above.")
    print("\n▶️  Run: python main.py")
    return 0

if __name__ == "__main__":
    sys.exit(check_environment())

