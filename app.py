import os
import tempfile
from pathlib import Path

import streamlit as st

from config.llm_config import LLMConfig
from config.settings import PROJECT_ROOT
from crew import VideoAnalysisSummaryCrew
from models.base_models import BaseInputModel
from tools.activity_detection_tool import MediaPipeModel
from utils import reset_crew_memory

st.set_page_config(
    page_title="Video Analysis AI",
    page_icon="🎥",
    layout="wide"
)

os.environ["USE_OPENAI"] = 'false'
os.environ["USE_GROQ"] = 'false'

st.title("🎥 Video Analysis AI")
st.markdown("Upload a video to analyze emotions and activities using AI agents")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze"
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

with col2:
    st.header("Configuration")

    frame_rate = st.slider(
        "Frame Sample Rate",
        min_value=1,
        max_value=60,
        value=30,
        help="Sample every Nth frame. Higher values = faster processing but may miss details (e.g., 30 = analyze 1 frame every 30 frames)."
    )

    pose_model = st.selectbox(
        "Pose Detection Model",
        options=[model.name for model in MediaPipeModel if model.name != 'HANDS'],
        index=0,
        help="LITE: Fast but less accurate\nFULL: Balanced\nHEAVY: Most accurate but slower"
    )

    use_memory = st.checkbox(
        "Use Crew Memory",
        value=False,
        help="Enables Crew to use analysis memory"
    )

    reset_memory = st.checkbox(
        "Reset Crew Memory",
        value=False,
        help="Clear previous analysis memory before starting"
    )

    if reset_memory:
        memory = st.selectbox(
            label="Memory",
            options=['short', 'long', 'entity', 'knowledge', 'all'],
            index=0
        )

    use_inference = st.checkbox(
        label="Use Inference API",
        help="If this is not checked Ollama container needs to be installed and running in the machine"
    )

    if use_inference:
        inference_api = st.selectbox(
            "Select Inference API",
            options=['OPENAI', 'GROQ'],
            index=0,
        )

        if inference_api == 'OPENAI':
            openai_api_key = st.text_input(
                label="Enter OpenAI api key",
                type="password"
            )
            if openai_api_key:
                os.environ["USE_OPENAI"] = 'true'
                os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            groq_api_key = st.text_input(
                label="Enter Groq api key",
                type="password"
            )
            if groq_api_key:
                os.environ['USE_GROQ'] = 'true'
                os.environ["GROQ_API_KEY"] = groq_api_key

        model_name = st.text_input(
            label="LLM model name"
        )
        if not model_name:
            st.error("Please provide the LLM model name!")

st.divider()

if 'video_cache' not in st.session_state:
    st.session_state.video_cache = {}
if 'last_file_id' not in st.session_state:
    st.session_state.last_file_id = None

if uploaded_file is not None:
    current_file_id = uploaded_file.file_id
    if current_file_id != st.session_state.last_file_id:
        for old_path in st.session_state.video_cache.values():
            if os.path.exists(old_path):
                try:
                    os.unlink(old_path)
                except:
                    pass
        st.session_state.video_cache.clear()
        st.session_state.last_file_id = current_file_id

if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a video file first!")
    else:
        file_id = uploaded_file.file_id

        if file_id in st.session_state.video_cache:
            video_path = st.session_state.video_cache[file_id]
        else:
            tmp_dir = tempfile.mkdtemp()
            video_filename = f"video_{file_id}{Path(uploaded_file.name).suffix}"
            video_path = os.path.join(tmp_dir, video_filename)

            with open(video_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            st.session_state.video_cache[file_id] = video_path

        try:
            with st.status("Analyzing video...", expanded=True) as status:
                st.write("🏃 Starting Activity Detector Agent")
                st.write(f"📹 Video: {uploaded_file.name}")
                st.write(f"⚙️ Frame Sample Rate: {frame_rate}")
                st.write(f"🤖 Model: {pose_model}")
                st.write("⏱️ This may take 2-15 minutes")

                crew = VideoAnalysisSummaryCrew(LLMConfig()).crew()

                if reset_memory:
                    st.write("🧹 Resetting crew memory...")
                    reset_crew_memory(crew, memory)

                if use_memory:
                    crew.memory = use_memory

                inputs = BaseInputModel(
                    video_path=str(Path(video_path)),
                    frame_rate=frame_rate,
                    pose_model=pose_model
                )
                result = crew.kickoff(inputs=inputs.model_dump())

                status.update(label="✅ Analysis Complete!", state="complete")

            st.success("Analysis completed successfully!")

            st.subheader(f"📊 Results - Tokens used : {result.token_usage}")

            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Reports", "Execution Trace", "Raw Output"])

            with tab1:
                st.markdown("### Analysis Summary")
                if hasattr(result, 'raw'):
                    st.write(result.raw)
                else:
                    st.write(result)

            with tab2:
                st.markdown("### Generated Reports")

                reports_dir = PROJECT_ROOT / "reports"
                summary_dir = PROJECT_ROOT / "summary"

                report_files = []
                if reports_dir.exists():
                    report_files.extend(sorted(reports_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True))
                if summary_dir.exists():
                    report_files.extend(sorted(summary_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True))

                if report_files:
                    for report_file in report_files[:5]:
                        with st.expander(f"📄 {report_file.name}"):
                            st.markdown(report_file.read_text())
                else:
                    st.info("No reports generated yet")

            with tab3:
                st.markdown("### Execution Trace")

                if hasattr(result, 'trace') and result.trace:
                    for idx, trace_entry in enumerate(result.trace):
                        with st.expander(f"🔍 Step {idx + 1}: {trace_entry.get('agent', 'Unknown Agent')}",
                                         expanded=(idx < 3)):
                            if 'task' in trace_entry:
                                st.markdown(f"**Task:** {trace_entry['task']}")
                            if 'action' in trace_entry:
                                st.markdown(f"**Action:** {trace_entry['action']}")
                            if 'tool' in trace_entry:
                                st.markdown(f"**Tool Used:** `{trace_entry['tool']}`")
                            if 'tool_input' in trace_entry:
                                st.markdown("**Tool Input:**")
                                st.code(str(trace_entry['tool_input']), language="json")
                            if 'output' in trace_entry:
                                st.markdown("**Output:**")
                                st.text(str(trace_entry['output'])[:1000] + (
                                    "..." if len(str(trace_entry['output'])) > 1000 else ""))
                            if 'thought' in trace_entry:
                                st.markdown("**Thought Process:**")
                                st.info(trace_entry['thought'])
                else:
                    st.info("No trace information available. Make sure tracing is enabled in crew configuration.")

                if hasattr(result, 'tasks_output') and result.tasks_output:
                    st.markdown("### Task Outputs")
                    for idx, task_output in enumerate(result.tasks_output):
                        with st.expander(
                                f"📋 Task {idx + 1}: {task_output.name if hasattr(task_output, 'name') else 'Task'}"):
                            st.markdown(
                                f"**Description:** {task_output.description if hasattr(task_output, 'description') else 'N/A'}")
                            st.markdown("**Output:**")
                            st.write(task_output.raw if hasattr(task_output, 'raw') else str(task_output))

            with tab4:
                st.markdown("### Raw Output")
                st.json(result.json_dict)

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

st.divider()

with st.expander("ℹ️ About"):
    st.markdown("""
    ### Video Analysis AI
    
    This application uses AI agents to analyze videos for:
    - **Emotion Detection**: Identifies facial expressions and emotions
    - **Activity Detection**: Recognizes human poses, gestures, and activities
    
    **Models:**
    - LITE: Fastest processing, suitable for quick analysis
    - FULL: Balanced accuracy and speed
    - HEAVY: Highest accuracy, best for detailed analysis
    
    **Frame Sample Rate:**
    - Lower values (1-10): Analyzes more frames, slower but more detailed
    - Medium values (20-30): Good balance for most videos  
    - Higher values (40-60): Faster processing but may skip details (samples fewer frames)
    """)
