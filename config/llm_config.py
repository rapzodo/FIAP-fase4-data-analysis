import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

os.environ["CREWAI_TRACING_ENABLED"] = "false"
groq_key = os.getenv("GROQ_API_KEY", "")
if groq_key and groq_key != "your_groq_api_key_here":
    os.environ["GROQ_API_KEY"] = groq_key

from crewai import LLM

class LLMConfig:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.use_groq = os.getenv("USE_GROQ", "true").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            print("⚠️  GROQ_API_KEY not configured. Using Ollama instead.")
            self.use_groq = False

    def get_llm(self, model_name: Optional[str] = None):
        if self.use_groq and self.groq_api_key:
            try:
                model = model_name or "llama-3.3-70b-versatile"
                print(f"🚀 Using Groq with model: {model}")
                return LLM(
                    model=f"groq/{model}",
                    api_key=self.groq_api_key,
                    temperature=0.1
                )
            except Exception as e:
                print(f"❌ Failed to initialize Groq: {e}")
                print("🔄 Falling back to Ollama...")
                return self._get_ollama_llm(model_name)
        else:
            return self._get_ollama_llm(model_name)

    def _get_ollama_llm(self, model_name: Optional[str] = None):
        model = model_name or "llama3.2"
        print(f"🦙 Using Ollama with model: {model}")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   And model is pulled: ollama pull {model}")
        return LLM(
            model=f"ollama/{model}",
            base_url=self.ollama_base_url
        )

llm_config = LLMConfig()

