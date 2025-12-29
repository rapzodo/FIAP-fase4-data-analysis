#!/usr/bin/env python3

import os
import sys

print("🔍 Testing LLM Configuration")
print("=" * 70)

try:
    from config.llm_config import llm_config

    print("\n✅ LLM Config loaded successfully")
    print(f"   Using Groq: {llm_config.use_groq}")
    print(f"   Groq API Key: {'✅ Configured' if llm_config.groq_api_key and llm_config.groq_api_key != 'your_groq_api_key_here' else '❌ Not configured'}")
    print(f"   Ollama URL: {llm_config.ollama_base_url}")

    print("\n📝 Testing LLM initialization...")
    llm = llm_config.get_llm()

    print("\n✅ LLM initialized successfully!")
    print(f"   Type: {type(llm).__name__}")

    print("\n🧪 Testing simple prompt...")
    try:
        response = llm.invoke("Say 'Hello from CrewAI!' in one sentence.")
        print(f"\n✅ LLM Response: {response.content}")

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYour LLM configuration is working correctly.")
        print("You can now run: python main.py")

    except Exception as e:
        print(f"\n❌ LLM invocation failed: {e}")
        print("\nIf using Groq: Check your API key")
        print("If using Ollama: Make sure it's running:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull model: ollama pull llama3.2")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ Configuration error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check .env file exists and has proper values")
    print("  2. Run: pip install -r requirements.txt")
    print("  3. Verify GROQ_API_KEY or setup Ollama")
    sys.exit(1)

