import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

os.environ["CREWAI_TRACING_ENABLED"] = "false"
groq_key = os.getenv("GROQ_API_KEY", "")
openai_key = os.getenv("OPENAI_API_KEY", "")
if groq_key and groq_key != "your_groq_api_key_here":
    os.environ["GROQ_API_KEY"] = groq_key

from crewai import LLM

class LLMConfig:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.use_groq = os.getenv("USE_GROQ", "true").lower() == "true"
        self.use_openai = os.getenv("USE_OPENAI", "true").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = 0.0
        self.time_out = 60000

        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            print("⚠️  GROQ_API_KEY OR OPENAI_API_KEY are not configured. Using Ollama instead.")
            self.use_groq = False

    def get_llm(self, model_name: Optional[str] = None) -> LLM:
        if self.use_groq and self.groq_api_key or self.use_openai and self.openai_key:
            model = model_name or "o3-mini"
            try:
                print(f"🚀 Using {"openai" if self.use_openai else "groq"} with model: {model}")
                model = f"{"openai" if self.use_openai else "groq"}/{model}"
                return LLM(
                    model=f"{"openai" if self.use_openai else "groq"}/{model}",
                    api_key= self.openai_key if self.use_openai else self.groq_api_key,
                    temperature=self.temperature,
                    timeout=self.time_out

                )
            except Exception as e:
                print(f"❌ Failed to initialize Groq: {e}")
                print("🔄 Falling back to Ollama...")
                return self._get_ollama_llm(model_name)
        else:
            return self._get_ollama_llm(model_name)

    def _get_ollama_llm(self, model_name: Optional[str] = None):
        default_model = "llama3.1:latest"
        model = model_name or default_model
        print(f"🦙 Using Ollama with model: {model}")
        print(f"   Make sure Ollama is running: ollama serve")
        print(f"   And model is pulled: ollama pull {model}")
        return LLM(
            model=f"ollama/{model}",
            temperature=self.temperature,
            base_url=self.ollama_base_url,
            timeout=self.time_out
        )

def get_llm_config() -> LLMConfig:
    return LLMConfig()

