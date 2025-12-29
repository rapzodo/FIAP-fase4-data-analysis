import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

class LLMConfig:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.use_groq = os.getenv("USE_GROQ", "true").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def get_llm(self, model_name: Optional[str] = None):
        if self.use_groq and self.groq_api_key:
            try:
                return ChatGroq(
                    api_key=self.groq_api_key,
                    model=model_name or "llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_tokens=4096
                )
            except Exception as e:
                print(f"Failed to initialize Groq: {e}. Falling back to Ollama...")
                return self._get_ollama_llm(model_name)
        else:
            return self._get_ollama_llm(model_name)

    def _get_ollama_llm(self, model_name: Optional[str] = None):
        return ChatOllama(
            base_url=self.ollama_base_url,
            model=model_name or "llama3.2",
            temperature=0.1
        )

llm_config = LLMConfig()

