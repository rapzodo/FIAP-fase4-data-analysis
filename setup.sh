#!/bin/bash

echo "🚀 Setting up Multi-Agent Video Analysis System"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2 | cut -d "." -f 1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it and add your GROQ_API_KEY"
    echo ""
    echo "⚠️  IMPORTANT: You need to:"
    echo "   1. Get a Groq API key from https://console.groq.com"
    echo "   2. Edit .env file and add your API key"
    echo "   3. Or set USE_GROQ=false to use Ollama (requires Ollama installed)"
else
    echo "✅ .env file already exists"
fi

# Create output directory
mkdir -p output
echo "✅ Output directory created"

echo ""
echo "================================================"
echo "✅ Setup complete!"
echo "================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file and add your GROQ_API_KEY (or use Ollama)"
echo "   2. Activate virtual environment: source .venv/bin/activate"
echo "   3. Run the application: python main.py"
echo ""
echo "📚 For more information, see README.md"

