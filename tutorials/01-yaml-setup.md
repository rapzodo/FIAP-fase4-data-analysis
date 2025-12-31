# Module 1: YAML Configuration Setup

**Time**: 20 minutes | **Difficulty**: Easy

## 🎯 What You'll Build

A clean configuration system using YAML files for:
- **4 Agent definitions** (facial, activity, summarizer, translator)
- Task descriptions and expected outputs
- Separation of config from code
- **Professional unit tests with unittest**

## 📋 Files to Create

```
config/agents.yaml
config/tasks.yaml
config/settings.py
agents/agent_factory.py
tasks/task_factory.py
tests/test_yaml_config.py
```

## 💻 Implementation

### Step 1: Create Agent Configuration

**File**: `config/agents.yaml`

```yaml
# Agent Configurations
# 4 agents for video analysis and translation

facial_recognition:
  role: "Facial Recognition and Emotion Detection Specialist"
  goal: "Detect faces and identify emotions in video footage with high accuracy"
  backstory: |
    You are a computer vision expert specializing in facial recognition 
    and emotion detection. You analyze video frames to identify faces, 
    determine emotional states, and report on patterns and anomalies. 
    You provide clear, data-driven insights about facial expressions.
  tools:
    - facial_recognition
  verbose: true
  allow_delegation: false

activity_detector:
  role: "Human Activity Recognition Specialist"
  goal: "Identify and classify human activities, poses, and gestures in video footage"
  backstory: |
    You are a computer vision expert specializing in human activity 
    recognition and pose estimation. You analyze body movements, gestures, 
    and activities to understand what people are doing in videos. 
    You provide detailed insights about physical activities and movement patterns.
  tools:
    - activity_detector
  verbose: true
  allow_delegation: false

summarizer:
  role: "Video Analysis Report Generator"
  goal: "Aggregate multi-agent analysis results and generate comprehensive summary reports"
  backstory: |
    You are a data analyst specializing in synthesizing complex information 
    from multiple sources. You excel at identifying patterns, calculating statistics, 
    and presenting insights in clear, actionable reports. You transform raw data 
    into executive-ready summaries with key metrics.
  tools: []
  verbose: true
  allow_delegation: false

translator:
  role: "Multilingual Technical Translator"
  goal: "Translate technical reports and analysis into multiple languages while preserving accuracy"
  backstory: |
    You are an expert technical translator fluent in multiple languages. 
    You specialize in translating data analysis reports, maintaining technical 
    accuracy while adapting content for different cultural contexts. You ensure 
    that statistics, insights, and terminology are correctly conveyed across languages.
  tools: []
  verbose: true
  allow_delegation: false
```

**What this defines:**
- **4 agents** (3 for video analysis + 1 translator)
- **Backstories** that shape behavior
- **Tools** each agent can use
- **Settings** like verbosity

### Step 2: Create Task Configuration

**File**: `config/tasks.yaml`

```yaml
# Task Configurations
# 4 tasks matching our 4 agents

analyze_facial_recognition:
  description: |
    Analyze the video at '{video_path}' for faces and emotions.
    
    Use facial_recognition tool with sample_rate={frame_sample_rate}.
    
    Provide:
    1. Total frames analyzed
    2. Faces detected with timestamps
    3. Emotion distribution (happy, sad, angry, neutral, surprise, fear, disgust)
    4. Confidence scores
    5. Anomalies detected (no face, low confidence)
    
    Format as structured data report with statistics.
  expected_output: "Facial recognition analysis with emotion data and statistics"
  agent: facial_recognition
  context: []

detect_activities:
  description: |
    Analyze the video at '{video_path}' for human activities and poses.
    
    Use activity_detector tool with sample_rate={frame_sample_rate}.
    
    Identify:
    1. Total frames analyzed
    2. Activities detected (standing, sitting, moving)
    3. Hand gestures (raised, lowered)
    4. Activity timeline with timestamps
    5. Anomalies (no pose detected)
    
    Format as structured activity report with timeline.
  expected_output: "Activity detection analysis with pose data and timeline"
  agent: activity_detector
  context: []

generate_summary:
  description: |
    Create a comprehensive summary report aggregating all analyses.
    
    Include these sections:
    
    ## Executive Summary
    - Overview in 3-4 sentences
    - Key findings
    
    ## Processing Statistics
    - Frames analyzed per agent
    - Success rates
    - Processing efficiency
    
    ## Facial Recognition Results
    - Detection rate
    - Emotion distribution with percentages
    - Confidence metrics
    
    ## Activity Detection Results
    - Activity breakdown with percentages
    - Movement patterns
    - Hand gesture analysis
    
    ## Anomaly Report
    - Total anomalies
    - Types and frequencies
    - Impact assessment
    
    ## Key Insights
    - Top 5 findings
    - Patterns and correlations
    - Notable observations
    
    Format in clean markdown with clear headings and bullet points.
  expected_output: "Comprehensive markdown summary report with all sections"
  agent: summarizer
  context:
    - analyze_facial_recognition
    - detect_activities

translate_report:
  description: |
    Translate the video analysis summary report to {target_language}.
    
    Requirements:
    1. Maintain all statistics and numbers exactly as they are
    2. Translate section headings and narrative text
    3. Preserve markdown formatting
    4. Keep technical terms accurate
    5. Adapt cultural context where appropriate
    6. Include a translation note at the top
    
    Target language: {target_language}
    
    Format: Same markdown structure as original, with all content translated.
  expected_output: "Translated report in target language with same structure"
  agent: translator
  context:
    - generate_summary
```

**What this defines:**
- **4 tasks** in execution order
- **Task dependencies** via context
- **Placeholders** for dynamic values (e.g., {video_path}, {target_language})
- **Clear expected outputs**

### Step 3: Update Settings

**File**: `config/settings.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# File paths
VIDEO_PATH = os.getenv("VIDEO_PATH", "tech-challenge/video.mp4")

# Processing settings
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", "10"))

# Translation settings
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "Portuguese")

# Output settings
OUTPUT_DIR = "output"

# YAML configuration paths
AGENTS_CONFIG_PATH = "config/agents.yaml"
TASKS_CONFIG_PATH = "config/tasks.yaml"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### Step 4: Create Agent Factory

**File**: `agents/agent_factory.py`

```python
import yaml
from crewai import Agent
from config.llm_config import llm_config


class AgentFactory:
    """Creates agents from YAML configuration."""
    
    def __init__(self, config_path: str):
        """Load agent configurations from YAML."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_agent(self, agent_name: str, tools: dict = None) -> Agent:
        """
        Create an agent from configuration.
        
        Args:
            agent_name: Name of agent in YAML config
            tools: Dictionary mapping tool names to tool instances
        
        Returns:
            Configured Agent instance
        """
        if agent_name not in self.config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")
        
        agent_config = self.config[agent_name]
        
        # Map tool names to tool instances
        agent_tools = []
        if tools:
            for tool_name in agent_config.get('tools', []):
                if tool_name in tools:
                    agent_tools.append(tools[tool_name])
                else:
                    print(f"⚠️  Warning: Tool '{tool_name}' not found for agent '{agent_name}'")
        
        # Create agent
        agent = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            tools=agent_tools,
            llm=llm_config.get_llm(),
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False)
        )
        
        return agent
    
    def get_all_agents(self):
        """Get list of all agent names in config."""
        return list(self.config.keys())
```

### Step 5: Create Task Factory

**File**: `tasks/task_factory.py`

```python
import yaml
from crewai import Task
from config.settings import VIDEO_PATH, FRAME_SAMPLE_RATE, TARGET_LANGUAGE


class TaskFactory:
    """Creates tasks from YAML configuration."""
    
    def __init__(self, config_path: str):
        """Load task configurations from YAML."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_task(self, task_name: str, agent, context_tasks: dict = None) -> Task:
        """
        Create a task from configuration.
        
        Args:
            task_name: Name of task in YAML config
            agent: Agent instance to assign task to
            context_tasks: Dictionary mapping task names to Task instances
        
        Returns:
            Configured Task instance
        """
        if task_name not in self.config:
            raise ValueError(f"Task '{task_name}' not found in configuration")
        
        task_config = self.config[task_name]
        
        # Replace placeholders in description
        description = task_config['description'].format(
            video_path=VIDEO_PATH,
            frame_sample_rate=FRAME_SAMPLE_RATE,
            target_language=TARGET_LANGUAGE
        )
        
        # Build context from previous tasks
        context = []
        if context_tasks:
            for context_name in task_config.get('context', []):
                if context_name in context_tasks:
                    context.append(context_tasks[context_name])
        
        # Create task
        task = Task(
            description=description,
            expected_output=task_config['expected_output'],
            agent=agent,
            context=context if context else None
        )
        
        return task
    
    def get_all_tasks(self):
        """Get list of all task names in config."""
        return list(self.config.keys())
    
    def get_task_agent_name(self, task_name: str) -> str:
        """Get the agent name for a task."""
        if task_name not in self.config:
            raise ValueError(f"Task '{task_name}' not found")
        return self.config[task_name]['agent']
```

## 🧪 Testing with unittest

### Step 6: Create Proper Unit Tests

**File**: `tests/test_yaml_config.py`

```python
import unittest
import yaml
from pathlib import Path

from agents.agent_factory import AgentFactory
from tasks.task_factory import TaskFactory
from config.settings import AGENTS_CONFIG_PATH, TASKS_CONFIG_PATH


class TestYamlConfiguration(unittest.TestCase):
    """Test YAML configuration loading and validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.agent_factory = AgentFactory(AGENTS_CONFIG_PATH)
        cls.task_factory = TaskFactory(TASKS_CONFIG_PATH)
    
    def test_agents_config_file_exists(self):
        """Test that agents.yaml file exists."""
        self.assertTrue(
            Path(AGENTS_CONFIG_PATH).exists(), 
            f"Config file not found: {AGENTS_CONFIG_PATH}"
        )
    
    def test_tasks_config_file_exists(self):
        """Test that tasks.yaml file exists."""
        self.assertTrue(
            Path(TASKS_CONFIG_PATH).exists(),
            f"Config file not found: {TASKS_CONFIG_PATH}"
        )
    
    def test_agent_count(self):
        """Test correct number of agents loaded."""
        agents = self.agent_factory.get_all_agents()
        self.assertEqual(
            len(agents), 4, 
            f"Expected 4 agents, got {len(agents)}"
        )
    
    def test_task_count(self):
        """Test correct number of tasks loaded."""
        tasks = self.task_factory.get_all_tasks()
        self.assertEqual(
            len(tasks), 4,
            f"Expected 4 tasks, got {len(tasks)}"
        )
    
    def test_agent_names(self):
        """Test agent names are correct."""
        agents = self.agent_factory.get_all_agents()
        expected_agents = [
            'facial_recognition', 
            'activity_detector', 
            'summarizer',
            'translator'
        ]
        
        for expected in expected_agents:
            self.assertIn(
                expected, agents,
                f"Agent '{expected}' not found in configuration"
            )
    
    def test_task_names(self):
        """Test task names are correct."""
        tasks = self.task_factory.get_all_tasks()
        expected_tasks = [
            'analyze_facial_recognition', 
            'detect_activities', 
            'generate_summary',
            'translate_report'
        ]
        
        for expected in expected_tasks:
            self.assertIn(
                expected, tasks,
                f"Task '{expected}' not found in configuration"
            )
    
    def test_agent_required_fields(self):
        """Test all agents have required fields."""
        required_fields = ['role', 'goal', 'backstory', 'tools']
        
        for agent_name in self.agent_factory.get_all_agents():
            agent_config = self.agent_factory.config[agent_name]
            
            for field in required_fields:
                self.assertIn(
                    field, agent_config,
                    f"Agent '{agent_name}' missing field: {field}"
                )
                self.assertIsNotNone(
                    agent_config[field],
                    f"Agent '{agent_name}' field '{field}' is None"
                )
    
    def test_task_required_fields(self):
        """Test all tasks have required fields."""
        required_fields = ['description', 'expected_output', 'agent']
        
        for task_name in self.task_factory.get_all_tasks():
            task_config = self.task_factory.config[task_name]
            
            for field in required_fields:
                self.assertIn(
                    field, task_config,
                    f"Task '{task_name}' missing field: {field}"
                )
                self.assertIsNotNone(
                    task_config[field],
                    f"Task '{task_name}' field '{field}' is None"
                )
    
    def test_task_agent_mapping(self):
        """Test all task agents reference valid agents."""
        agent_names = set(self.agent_factory.get_all_agents())
        
        for task_name in self.task_factory.get_all_tasks():
            agent_name = self.task_factory.get_task_agent_name(task_name)
            self.assertIn(
                agent_name, agent_names,
                f"Task '{task_name}' references unknown agent '{agent_name}'"
            )
    
    def test_yaml_syntax_valid(self):
        """Test YAML files have valid syntax."""
        # Test agents.yaml
        with open(AGENTS_CONFIG_PATH, 'r') as f:
            try:
                agents_data = yaml.safe_load(f)
                self.assertIsInstance(agents_data, dict)
            except yaml.YAMLError as e:
                self.fail(f"Invalid YAML syntax in {AGENTS_CONFIG_PATH}: {e}")
        
        # Test tasks.yaml
        with open(TASKS_CONFIG_PATH, 'r') as f:
            try:
                tasks_data = yaml.safe_load(f)
                self.assertIsInstance(tasks_data, dict)
            except yaml.YAMLError as e:
                self.fail(f"Invalid YAML syntax in {TASKS_CONFIG_PATH}: {e}")
    
    def test_agent_tools_is_list(self):
        """Test agent tools field is a list."""
        for agent_name in self.agent_factory.get_all_agents():
            agent_config = self.agent_factory.config[agent_name]
            tools = agent_config.get('tools', [])
            self.assertIsInstance(
                tools, list,
                f"Agent '{agent_name}' tools must be a list"
            )
    
    def test_task_context_is_list(self):
        """Test task context field is a list when present."""
        for task_name in self.task_factory.get_all_tasks():
            task_config = self.task_factory.config[task_name]
            context = task_config.get('context', [])
            self.assertIsInstance(
                context, list,
                f"Task '{task_name}' context must be a list"
            )
    
    def test_translator_agent_exists(self):
        """Test that translator agent is properly configured."""
        self.assertIn(
            'translator', 
            self.agent_factory.get_all_agents(),
            "Translator agent not found in configuration"
        )
        
        translator_config = self.agent_factory.config['translator']
        self.assertIn('Multilingual', translator_config['role'])
        self.assertIn('Translate', translator_config['goal'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
```

## 🧪 Run Tests

```bash
# Install PyYAML if not already installed
pip install pyyaml

# Run all tests
python -m unittest tests.test_yaml_config -v

# Or run directly
python tests/test_yaml_config.py
```

**Expected output:**
```
test_agent_count ... ok
test_agent_names ... ok
test_agent_required_fields ... ok
test_agent_tools_is_list ... ok
test_agents_config_file_exists ... ok
test_task_agent_mapping ... ok
test_task_context_is_list ... ok
test_task_count ... ok
test_task_names ... ok
test_task_required_fields ... ok
test_tasks_config_file_exists ... ok
test_translator_agent_exists ... ok
test_yaml_syntax_valid ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.012s

OK
```

## 💡 Key Concepts

### Why 4 Agents?

1. **Facial Recognition** - Emotion detection
2. **Activity Detector** - Pose and gestures  
3. **Summarizer** - Aggregate results
4. **Translator** - Multilingual reports

**Translator Agent Benefits:**
- Reach international audiences
- Cultural adaptation
- Maintains technical accuracy
- Same data, multiple languages

### Why unittest?

**Professional Testing:**
- ✅ Standard Python testing framework
- ✅ Clear test structure with assertions
- ✅ Automatic test discovery
- ✅ Detailed failure messages
- ✅ setUp/tearDown for fixtures
- ✅ Integration with CI/CD

**Example Test:**
```python
def test_agent_count(self):
    """Test correct number of agents loaded."""
    agents = self.agent_factory.get_all_agents()
    self.assertEqual(len(agents), 4)  # ✅ Automated check
```

### YAML Configuration

**Benefits:**
- ✅ Change agent prompts without code
- ✅ Version control friendly
- ✅ Easy A/B testing
- ✅ Team collaboration
- ✅ Non-developers can modify

**Structure:**
```yaml
agent_name:
  role: "What the agent is"
  goal: "What it should achieve"
  backstory: |
    Long description...
  tools:
    - tool_name
```

## ✅ Verification Checklist

- [ ] agents.yaml created with 4 agents (including translator)
- [ ] tasks.yaml created with 4 tasks
- [ ] settings.py updated with TARGET_LANGUAGE
- [ ] agent_factory.py creates agents from YAML
- [ ] task_factory.py creates tasks from YAML
- [ ] test_yaml_config.py with 13 unittest cases
- [ ] All tests pass (13/13 OK)
- [ ] You understand unittest assertions

## 🎯 What You've Learned

1. **YAML configuration** - Clean, maintainable config
2. **Factory pattern** - Create objects from config
3. **unittest framework** - Professional Python testing
4. **Test assertions** - Validate code behavior
5. **Test organization** - Test classes and methods
6. **Translator agent** - Multilingual capabilities

## 📝 Next Steps

**Module 2**: Build the Facial Recognition tool with proper unittest!

**Translator Features:**
- Translate reports to any language
- Preserve technical accuracy
- Maintain markdown formatting
- Cultural adaptation

---

**Ready?** Move to `02-facial-recognition.md` to build your first video processing tool! 🚀

