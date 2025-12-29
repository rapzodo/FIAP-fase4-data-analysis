# Quick Start Guide

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
./setup.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Create .env configuration file

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env .env
```

## 🔑 Configuration

### Get Groq API Key (Recommended)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Create an API key
4. Add it to your `.env` file:

```env
GROQ_API_KEY=gsk_your_api_key_here
USE_GROQ=true
```

### Alternative: Use Ollama (No API Key Required)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the llama3.2 model:
   ```bash
   ollama pull llama3.2
   ```
3. Update your `.env` file:
   ```env
   USE_GROQ=false
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## ▶️ Running the Application

### Start the Analysis

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # On macOS/Linux

# Run the main application
python main.py
```

### What Happens:

The application will:
1. **Parse PDF** - Extract tech challenge requirements
2. **Analyze Video** - Detect faces and emotions (5-10 minutes for full video)
3. **Detect Activities** - Identify human activities and poses
4. **Generate Report** - Create comprehensive summary
5. **Create Demo Script** - Generate demonstration video script

### Output

Results are saved in the `output/` directory:
```
output/
└── analysis_report_YYYYMMDD_HHMMSS.md
```

## ⚙️ Configuration Options

### Adjust Video Processing Speed

Edit `.env` to change frame sampling rate:

```env
FRAME_SAMPLE_RATE=1   # Process every frame (slower, more accurate)
FRAME_SAMPLE_RATE=5   # Process every 5th frame (faster)
FRAME_SAMPLE_RATE=10  # Process every 10th frame (fastest)
```

### Use Different Video/PDF

Edit `.env` to change input files:

```env
VIDEO_PATH=path/to/your/video.mp4
PDF_PATH=path/to/your/document.pdf
```

## 🐛 Troubleshooting

### Issue: "Package requirements not satisfied"

**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Failed to open video"

**Solution**: Check that the video file exists:
```bash
ls -la tech-challenge/*.mp4
```

### Issue: "Groq API rate limit exceeded"

**Solution**: The system will automatically fallback to Ollama. Or manually switch:
```env
USE_GROQ=false
```

### Issue: "DeepFace model download failed"

**Solution**: DeepFace downloads models on first run. Ensure internet connection and try again.

### Issue: "No module named 'cv2'"

**Solution**: Reinstall opencv:
```bash
pip install --force-reinstall opencv-python
```

## 📊 Sample Output

The system provides:

- **Frame Analysis**: Total frames processed
- **Emotion Detection**: Distribution of detected emotions
  ```
  happy: 45, sad: 12, angry: 3, neutral: 89, surprise: 15
  ```
- **Activity Detection**: Timeline of activities
  ```
  standing: 67%, sitting: 25%, moving: 8%
  ```
- **Anomalies**: Detection issues and low-confidence results
- **Insights**: AI-generated analysis and patterns

## 🎯 Testing

### Quick Test (Fast)

Use a higher frame sample rate for quick testing:

```bash
# Edit .env
FRAME_SAMPLE_RATE=30  # Process every 30th frame

# Run
python main.py
```

This will complete in 1-2 minutes for testing purposes.

### Full Analysis (Accurate)

Use frame-by-frame analysis for best results:

```bash
# Edit .env
FRAME_SAMPLE_RATE=1  # Process every frame

# Run
python main.py
```

This will take longer but provides more accurate results.

## 📝 Next Steps

1. Review the generated report in `output/`
2. Adjust configuration based on your needs
3. Try with your own videos and PDFs
4. Customize agent behaviors in `config/settings.py`

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the plan in [plan-multiAgentVideoAnalysis.prompt.md](plan-multiAgentVideoAnalysis.prompt.md)
- Check agent configurations in `config/settings.py`

