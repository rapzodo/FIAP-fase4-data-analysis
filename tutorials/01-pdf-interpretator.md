# Module 1: PDF Interpretator Agent

**Time**: 30 minutes | **Difficulty**: Easy

## 🎯 What You'll Build

An agent that reads PDF documents and extracts:
- Problem statement
- Proposed solution
- Expected deliverables

## 📋 Files to Create

```
tools/pdf_parser_tool.py
agents/pdf_interpretator.py
tests/test_pdf_agent.py
```

## 💻 Implementation

### Step 1: Create PDF Parser Tool

**File**: `tools/pdf_parser_tool.py`

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import PyPDF2
import json
from pathlib import Path


class PDFParserInput(BaseModel):
    """Input for PDF parser tool."""
    pdf_path: str = Field(..., description="Path to the PDF file")


class PDFParserTool(BaseTool):
    name: str = "pdf_parser"
    description: str = "Extracts text content from PDF files"
    args_schema: type[BaseModel] = PDFParserInput
    
    def _run(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        try:
            # Validate file exists
            if not Path(pdf_path).exists():
                return json.dumps({"error": f"File not found: {pdf_path}"})
            
            # Extract text
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    text_parts.append(f"--- Page {i+1} ---\n{text}")
            
            full_text = "\n\n".join(text_parts)
            
            return json.dumps({
                "success": True,
                "pages": len(reader.pages),
                "content": full_text
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
```

**Key points:**
- Uses `BaseTool` from CrewAI
- `_run()` method does the actual work
- Returns JSON for structured output
- Error handling included

### Step 2: Create the Agent

**File**: `agents/pdf_interpretator.py`

```python
from crewai import Agent
from tools.pdf_parser_tool import PDFParserTool
from config.llm_config import llm_config


def create_pdf_interpretator():
    """Create agent that interprets PDF requirements."""
    
    return Agent(
        role="Technical Requirements Analyst",
        goal="Extract and structure requirements from technical documents",
        backstory=(
            "You are an expert at reading technical documents and "
            "extracting key information: problems, solutions, and deliverables. "
            "You organize information clearly and concisely."
        ),
        tools=[PDFParserTool()],
        llm=llm_config.get_llm(),
        verbose=True,
        allow_delegation=False
    )
```

**Why these parameters:**
- `role`: Defines expertise
- `goal`: What success looks like
- `backstory`: Shapes LLM behavior
- `tools`: PDF parser for reading files
- `verbose=True`: See agent thinking

### Step 3: Create Test

**File**: `tests/test_pdf_agent.py`

```python
import os
from crewai import Crew, Task
from agents.pdf_interpretator import create_pdf_interpretator


def test_pdf_interpretator():
    """Test PDF parsing and interpretation."""
    
    # Check if PDF exists
    pdf_path = "tech-challenge/Tech Challenge - IADT - Fase 4.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at: {pdf_path}")
        print("💡 Update the path to your actual PDF file")
        return
    
    print("🔍 Testing PDF Interpretator Agent\n")
    
    # Create agent
    agent = create_pdf_interpretator()
    print("✅ Agent created\n")
    
    # Create task
    task = Task(
        description=f"""
        Read the PDF document at '{pdf_path}' and extract:
        
        1. **Problem Statement**: What challenge needs to be solved?
        2. **Proposed Solution**: What is being built to solve it?
        3. **Expected Deliverables**: What outputs are required?
        
        Format as clear sections with bullet points.
        """,
        agent=agent,
        expected_output="Structured extraction with 3 sections: Problem, Solution, Deliverables"
    )
    print("✅ Task created\n")
    
    # Execute
    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    
    print("🚀 Executing...\n")
    result = crew.kickoff()
    
    print("\n" + "="*70)
    print("📄 EXTRACTED REQUIREMENTS")
    print("="*70)
    print(result)
    print("="*70)
    
    return result


if __name__ == "__main__":
    test_pdf_interpretator()
```

## 🧪 Testing

Run the test:
```bash
python tests/test_pdf_agent.py
```

**Expected flow:**
1. Agent receives task
2. Uses `pdf_parser` tool to read PDF
3. LLM analyzes extracted text
4. Returns structured requirements

**Success looks like:**
```
📄 EXTRACTED REQUIREMENTS
======================================================================
**Problem Statement**
- [Extracted problem from PDF]

**Proposed Solution**
- [Extracted solution details]

**Expected Deliverables**
- [List of deliverables]
======================================================================
```

## 🐛 Troubleshooting

### Error: "File not found"
**Fix**: Update `pdf_path` in test to your actual PDF location

### Error: "Tool not found"
**Fix**: Make sure `tools/__init__.py` exists (can be empty)

### Agent doesn't use tool
**Fix**: Task description should explicitly mention reading the PDF

### Extraction is empty
**Fix**: PDF might be scanned images. Try a text-based PDF first

## ✅ Verification Checklist

- [ ] Tool extracts PDF text successfully
- [ ] Agent creates without errors
- [ ] Test runs and completes
- [ ] Output contains 3 sections
- [ ] Requirements make sense

## 💡 How It Works

```
1. Task tells agent: "Read PDF at X, extract Y"
2. Agent thinks: "I need to read a PDF"
3. Agent calls: pdf_parser_tool(pdf_path="...")
4. Tool returns: JSON with PDF text
5. Agent analyzes: Text → Structured requirements
6. Agent returns: Final formatted output
```

## 🎯 Key Learnings

1. **Tools extend agent capabilities** - Agents can't read files without tools
2. **JSON for structured data** - Makes parsing easier
3. **Clear task descriptions** - Tell agent exactly what you want
4. **Test incrementally** - Tool → Agent → Task → Crew

## 📝 Next Steps

**Module 2**: Build the Facial Recognition Agent that processes videos!

**Challenge yourself:**
- Add error handling for corrupted PDFs
- Extract tables and images
- Support multiple document formats (DOCX, TXT)

---

**Ready?** Move to `02-facial-recognition.md` when this agent works! 🚀

