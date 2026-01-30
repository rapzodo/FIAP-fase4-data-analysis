import os
import tempfile
from pathlib import Path

import streamlit as st

from config.settings import PROJECT_ROOT
from crew import VideoAnalysisSummaryCrew
from tools.activity_detection_tool import MediaPipeModel
from utils import reset_crew_memory

st.set_page_config(
    page_title="Video Analysis AI",
    page_icon="🎥",
    layout="wide"
)

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
            options=['short','long','entity','knowledge','all'],
            index = 0
        )

st.divider()

if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload a video file first!")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        try:
            os.environ["VIDEO_PATH"] = video_path
            os.environ["FRAME_SAMPLE_RATE"] = str(frame_rate)
            os.environ["POSE_MODEL"] = pose_model

            with st.status("Analyzing video...", expanded=True) as status:
                st.write("🏃 Starting Activity Detector Agent")
                st.write(f"📹 Video: {uploaded_file.name}")
                st.write(f"⚙️ Frame Sample Rate: {frame_rate}")
                st.write(f"🤖 Model: {pose_model}")
                st.write("⏱️ This may take 2-15 minutes")

                import config.settings as settings
                settings.VIDEO_PATH = Path(video_path)
                settings.FRAME_SAMPLE_RATE = str(frame_rate)
                settings.POSE_MODEL = pose_model

                crew = VideoAnalysisSummaryCrew().crew()

                if reset_memory:
                    st.write("🧹 Resetting crew memory...")
                    reset_crew_memory(crew, memory)

                if use_memory:
                    crew.memory = use_memory

                result = crew.kickoff()

                status.update(label="✅ Analysis Complete!", state="complete")

            st.success("Analysis completed successfully!")

            st.subheader("📊 Results")

            tab1, tab2, tab3 = st.tabs(["Summary", "Reports", "Raw Output"])

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
                st.markdown("### Raw Output")
                st.json(str(result))

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

        finally:
            if os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass

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
