import os
from crewai import Crew, Task, Process
from agents import (
    create_tech_challenge_interpretator_agent,
    create_facial_recognition_agent,
    create_activity_detector_agent,
    create_summarizer_agent,
    create_demo_video_agent
)
from config.settings import VIDEO_PATH, PDF_PATH, FRAME_SAMPLE_RATE, OUTPUT_DIR
from datetime import datetime
import json

def create_tasks(agents):
    interpretator_agent = agents['interpretator']
    facial_agent = agents['facial']
    activity_agent = agents['activity']
    summarizer_agent = agents['summarizer']
    demo_agent = agents['demo']

    task1 = Task(
        description=f"""Analyze the tech challenge PDF document at '{PDF_PATH}'.
        Extract and structure the following information:
        1. The problem statement
        2. The proposed solution with all requirements
        3. What is expected as deliverables
        
        Provide a clear, well-formatted summary that can be used as context by other agents.""",
        agent=interpretator_agent,
        expected_output="A structured document containing the problem, solution requirements, and expected deliverables from the tech challenge PDF."
    )

    task2 = Task(
        description=f"""Analyze the video at '{VIDEO_PATH}' for facial recognition and emotion detection.
        Use the facial_recognition_tool with frame_sample_rate={FRAME_SAMPLE_RATE}.
        
        Process the video and provide:
        1. Total number of frames analyzed
        2. All faces detected with timestamps
        3. Emotions detected for each face
        4. Confidence scores
        5. Any anomalies detected (no face, low confidence, etc.)
        
        Return the complete analysis in JSON format.""",
        agent=facial_agent,
        expected_output="JSON formatted data containing frame counts, detected faces, emotions, confidence scores, and anomalies.",
        context=[task1]
    )

    task3 = Task(
        description=f"""Analyze the video at '{VIDEO_PATH}' for human activity detection.
        Use the activity_detector_tool with frame_sample_rate={FRAME_SAMPLE_RATE}.
        
        Process the video and identify:
        1. Total number of frames analyzed
        2. All activities detected (standing, sitting, walking, hand gestures)
        3. Timeline of activities with timestamps
        4. Activity summary statistics
        5. Any anomalies (no pose detected, unclear movements, etc.)
        
        Return the complete analysis in JSON format.""",
        agent=activity_agent,
        expected_output="JSON formatted data containing frame counts, detected activities, timeline, and anomalies.",
        context=[task1]
    )

    task4 = Task(
        description="""Create a comprehensive summary report aggregating the results from facial recognition and activity detection.
        
        The report should include:
        1. Executive summary of the video analysis
        2. Total frames analyzed by each system
        3. Emotion distribution (how many times each emotion was detected)
        4. Activity distribution (how many times each activity was detected)
        5. Timeline correlation between emotions and activities
        6. Total number of anomalies detected and their types
        7. Key insights and patterns observed
        8. Alignment with the tech challenge requirements
        
        Format the report in Markdown with clear sections and statistics.""",
        agent=summarizer_agent,
        expected_output="A comprehensive Markdown report summarizing all video analysis results with statistics and insights.",
        context=[task1, task2, task3]
    )

    task5 = Task(
        description="""Create a demonstration video script that showcases the multi-agent application.
        
        The script should:
        1. Introduce the tech challenge and requirements
        2. Explain the multi-agent architecture (5 agents and their roles)
        3. Demonstrate each agent's functionality:
           - Tech Challenge Interpretator
           - Facial Recognition Agent
           - Activity Detector Agent
           - Summarizer Agent
           - Demo Video Script Agent (this agent)
        4. Show sample outputs from each agent
        5. Highlight key features and capabilities
        6. Conclude with the value proposition
        
        Format as a video script with scenes, narration, and visual cues.""",
        agent=demo_agent,
        expected_output="A detailed video demonstration script with scenes, narration, and visual descriptions.",
        context=[task1, task4]
    )

    return [task1, task2, task3, task4, task5]

def main():
    print("🚀 Starting Multi-Agent Video Analysis System")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n📋 Creating agents...")
    agents = {
        'interpretator': create_tech_challenge_interpretator_agent(),
        'facial': create_facial_recognition_agent(),
        'activity': create_activity_detector_agent(),
        'summarizer': create_summarizer_agent(),
        'demo': create_demo_video_agent()
    }
    print("✅ All agents created successfully")

    print("\n📝 Creating tasks...")
    tasks = create_tasks(agents)
    print("✅ All tasks created successfully")

    print("\n🤖 Creating crew...")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    print("✅ Crew assembled successfully")

    print("\n🎬 Starting crew execution...")
    print("=" * 70)

    try:
        result = crew.kickoff()

        print("\n" + "=" * 70)
        print("✅ Crew execution completed successfully!")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = os.path.join(OUTPUT_DIR, f"analysis_report_{timestamp}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(result))

        print(f"\n📄 Report saved to: {output_file}")

        print("\n" + "=" * 70)
        print("📊 FINAL RESULTS")
        print("=" * 70)
        print(result)

        return result

    except Exception as e:
        print(f"\n❌ Error during crew execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

