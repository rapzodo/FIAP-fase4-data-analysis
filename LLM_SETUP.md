# LLM Configuration Guide

## ✅ Quick Test

Before running the main application, test your LLM configuration:

```bash
python test_llm.py
```

This will verify that your LLM (Groq or Ollama) is properly configured.

---

## 🚀 Option 1: Use Groq (Recommended - Fast & Easy)

### Step 1: Get API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Create an API key
4. Copy the key (starts with `gsk_`)

### Step 2: Configure

Edit `.env` file:

```bash
GROQ_API_KEY=gsk_your_actual_key_here
USE_GROQ=true
```

### Step 3: Test

```bash
python test_llm.py
```

You should see: `✅ ALL TESTS PASSED!`

---

## 🦙 Option 2: Use Ollama (Local - No API Key)

### Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### Step 2: Start Ollama

```bash
ollama serve
```

Leave this running in a terminal.

### Step 3: Pull Model

In another terminal:

```bash
ollama pull llama3.2
```

### Step 4: Configure

Edit `.env` file:

```bash
USE_GROQ=false
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 5: Test

```bash
python test_llm.py
```

You should see: `✅ ALL TESTS PASSED!`

---

## 🔧 Troubleshooting

### Error: "OpenAI API key not found"

**Solution**: The `.env` file now includes `OPENAI_API_KEY=not-needed` to bypass this.

If you still see this error, manually add to `.env`:

```bash
OPENAI_API_KEY=not-needed
```

### Error: "Groq API rate limit exceeded"

**Solutions**:
1. Wait a minute and try again (Groq has rate limits)
2. Switch to Ollama temporarily:
   ```bash
   # Edit .env
   USE_GROQ=false
   ```

### Error: "Connection refused" (Ollama)

**Solutions**:
1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check if the model is pulled:
   ```bash
   ollama list
   ollama pull llama3.2
   ```

### Error: "Model not found" (Ollama)

**Solution**: Pull the model:
```bash
ollama pull llama3.2
```

### LLM is too slow

**Solutions**:
1. **Use Groq** (much faster than Ollama):
   ```bash
   # Edit .env
   USE_GROQ=true
   GROQ_API_KEY=your_key_here
   ```
2. Use a smaller Ollama model:
   ```bash
   ollama pull llama3.2:1b  # Smaller, faster
   ```

---

## 📊 LLM Comparison

| Feature | Groq | Ollama |
|---------|------|--------|
| **Speed** | ⚡ Very Fast (<2s) | 🐢 Slower (5-30s) |
| **Setup** | API Key required | Local installation |
| **Cost** | Free tier available | Completely free |
| **Internet** | Required | Not required |
| **Quality** | Excellent | Good |
| **Models** | llama-3.3-70b | llama3.2, llama3.1, etc. |

---

## 🎯 Recommended Setup

### For Development (Fast iteration)
```bash
USE_GROQ=true
GROQ_API_KEY=your_key_here
```

### For Production/Offline
```bash
USE_GROQ=false
# Ollama running locally
```

---

## 🔍 Verify Configuration

Run the test script anytime to check your setup:

```bash
python test_llm.py
```

Expected output:
```
🔍 Testing LLM Configuration
======================================================================

✅ LLM Config loaded successfully
   Using Groq: True
   Groq API Key: ✅ Configured

📝 Testing LLM initialization...
🚀 Using Groq with model: llama-3.3-70b-versatile

✅ LLM initialized successfully!

🧪 Testing simple prompt...
✅ LLM Response: Hello from CrewAI!

======================================================================
✅ ALL TESTS PASSED!
======================================================================
```

---

## 💡 Pro Tips

1. **Start with Groq**: It's faster and easier for testing
2. **Keep Ollama as backup**: Good for offline work
3. **Monitor rate limits**: Groq free tier has limits
4. **Test before running main app**: Use `python test_llm.py`
5. **Check .env file**: Make sure no typos in API key

---

## 📝 Current Configuration

Check your current setup:

```bash
cat .env | grep -E "GROQ|OLLAMA|USE_GROQ"
```

Should show:
```
GROQ_API_KEY=gsk_...
OLLAMA_BASE_URL=http://localhost:11434
USE_GROQ=true
```

---

## 🆘 Still Having Issues?

1. **Run the configuration check**:
   ```bash
   python check_setup.py
   ```

2. **Test LLM directly**:
   ```bash
   python test_llm.py
   ```

3. **Check environment variables**:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('GROQ:', os.getenv('GROQ_API_KEY', 'NOT SET')[:20]); print('USE_GROQ:', os.getenv('USE_GROQ'))"
   ```

4. **Reinstall dependencies**:
   ```bash
   pip install --force-reinstall langchain-groq langchain-ollama
   ```

---

**Now you're ready to run the main application!**

```bash
python main.py
```

