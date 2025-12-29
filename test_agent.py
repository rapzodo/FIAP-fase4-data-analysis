#!/usr/bin/env python3

import os
import sys

print("🧪 Quick Agent Test")
print("=" * 70)

try:
    print("\n1️⃣ Loading configuration...")
    from config.llm_config import llm_config
    from config.settings import AGENT_CONFIG
    print("   ✅ Configuration loaded")

    print("\n2️⃣ Testing LLM...")
    llm = llm_config.get_llm()
    print("   ✅ LLM initialized")

    print("\n3️⃣ Creating a simple agent...")
    from crewai import Agent

    agent = Agent(
        role="Test Agent",
        goal="Test if agent creation works",
        backstory="A simple test agent",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    print("   ✅ Agent created successfully")

    print("\n4️⃣ Testing agent with simple task...")
    from crewai import Task, Crew

    task = Task(
        description="Say 'CrewAI agent is working!' in one sentence.",
        agent=agent,
        expected_output="A single sentence confirming the agent works."
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    print("   🚀 Running crew (this may take a few seconds)...")
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("✅ SUCCESS!")
    print("=" * 70)
    print(f"\n📝 Agent Output:\n{result}")
    print("\n" + "=" * 70)
    print("✅ Your multi-agent system is working!")
    print("   You can now run: python main.py")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Troubleshooting:")
    print("   1. Check .env file has GROQ_API_KEY or USE_GROQ=false")
    print("   2. Run: python test_llm.py")
    print("   3. Run: python check_setup.py")
    sys.exit(1)

