import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import os
import random

# Import our custom modules
from models.adaptive_agent import AdaptiveTutorAgent
from models.student_simulator import StudentSimulator
from utils.data_models import StudentProfile, SessionData, Question, DifficultyLevel, StudentState
from utils.question_bank import QuestionBank
from utils.analytics import AnalyticsDashboard

# Configure Streamlit page
st.set_page_config(
    page_title="AI Adaptive Tutorial Agent Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme styling */
    .main {
        padding-top: 2rem;
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    .stApp {
        background: var(--bg-tertiary);
    }
    
    .stApp > header {
        background: transparent;
    }
    
    /* Subtle dark theme color scheme */
    :root {
        --primary-color: #6366F1;
        --secondary-color: #8B5CF6;
        --accent-color: #EC4899;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --bg-primary: #1E293B;
        --bg-secondary: #334155;
        --bg-tertiary: #0F172A;
        --border-color: #475569;
        --card-bg: #1E293B;
        --sidebar-bg: #1E293B;
    }
    
    /* Header styling - Dark theme */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25), 0 4px 16px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling - Dark theme */
    .css-1d391kg {
        background: var(--sidebar-bg) !important;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 1.5rem 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card styling - Dark theme */
    .info-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .info-card:hover {
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15), 0 4px 16px rgba(0, 0, 0, 0.5);
        transform: translateY(-4px);
        border-color: var(--primary-color);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--success-color), #26d0ce);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Button styling - Dark theme */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25), 0 2px 8px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3), 0 4px 16px rgba(0, 0, 0, 0.5);
        filter: brightness(1.1);
    }
    
    /* Question card styling - Dark theme */
    .question-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3), 0 4px 16px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(20px);
    }
    
    .question-card h3 {
        color: white;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .question-content {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Success/Error styling - Dark theme */
    .stSuccess {
        background: linear-gradient(135deg, var(--success-color), var(--primary-color));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25);
    }
    
    .stError {
        background: linear-gradient(135deg, var(--error-color), #FF6B6B);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.25);
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
    }
    
    /* Tab styling - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
        border-color: var(--primary-color);
    }
    
    /* Metric styling - Dark theme */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-color) 100%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 32px rgba(236, 72, 153, 0.25), 0 4px 16px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"] > label {
        color: white !important;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    /* Input styling - Dark theme */
    .stTextInput > div > div > input {
        background: var(--card-bg) !important;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        color: var(--text-primary) !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        background: var(--bg-secondary) !important;
    }
    
    .stSelectbox > div > div > div {
        background: var(--card-bg) !important;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        color: var(--text-primary) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 4px;
    }
    
    /* Plotly chart containers - Dark theme */
    .plotly-graph-div {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        overflow: hidden;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom animations and effects */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.25); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.4); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .glow-effect {
        animation: glow 3s ease-in-out infinite;
    }
    
    /* Dark theme scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--secondary-color), var(--accent-color));
    }
    
    /* Enhanced data table styling */
    .stDataFrame {
        background: var(--card-bg) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Progress indicators */
    .stProgress .stProgress-bar {
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        border-radius: 4px;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 0 0 12px 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'student_sim' not in st.session_state:
    st.session_state.student_sim = None
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'question_history' not in st.session_state:
    st.session_state.question_history = []
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = AnalyticsDashboard()
if 'asked_questions' not in st.session_state:
    st.session_state.asked_questions = set()
if 'current_topic_index' not in st.session_state:
    st.session_state.current_topic_index = 0

def load_models():
    """Load the pre-trained models"""
    try:
        # Initialize agent
        agent = AdaptiveTutorAgent()
        
        # Load models from data directory
        model_path = Path("data")
        if model_path.exists():
            agent.load_models(str(model_path))
            return agent
        else:
            st.error("Model files not found. Please ensure the trained models are available.")
            return None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def initialize_student_simulator(profile):
    """Initialize the student simulator"""
    return StudentSimulator(profile)

def get_next_unique_question(agent, student_profile, asked_questions_set):
    """Get a unique question that hasn't been asked before"""
    available_topics = agent.question_bank.get_available_topics()
    
    # Try to get a question from each topic rotation
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        # Cycle through topics to ensure variety
        topic_index = st.session_state.current_topic_index % len(available_topics)
        current_topic = available_topics[topic_index]
        st.session_state.current_topic_index += 1
        
        # Get target difficulty for this student and topic
        knowledge = student_profile.knowledge_state.get(current_topic, 0.0)
        
        # Adaptive difficulty based on performance
        recent_performance = get_recent_topic_performance(current_topic, st.session_state.question_history)
        
        if recent_performance > 0.7:  # Doing well, increase difficulty
            target_difficulty = min(1.0, knowledge + 0.2 + random.uniform(0, 0.2))
        elif recent_performance < 0.4:  # Struggling, decrease difficulty
            target_difficulty = max(0.1, knowledge - 0.2 + random.uniform(0, 0.2))
        else:  # Maintain level
            target_difficulty = knowledge + random.uniform(-0.1, 0.1)
        
        target_difficulty = max(0.1, min(1.0, target_difficulty))
        
        # Try different difficulty levels around the target
        for difficulty_offset in [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
            test_difficulty = max(0.1, min(1.0, target_difficulty + difficulty_offset))
            
            # Get questions in this difficulty range
            questions_in_range = agent.question_bank.get_questions_by_difficulty_range(
                current_topic, test_difficulty - 0.1, test_difficulty + 0.1
            )
            
            # Filter out already asked questions
            available_questions = [
                q for q in questions_in_range 
                if q.id not in asked_questions_set
            ]
            
            if available_questions:
                # Return a random question from available ones
                selected_question = random.choice(available_questions)
                return selected_question
        
        attempts += 1
    
    # If we couldn't find any new questions, clear some old ones and try again
    if len(asked_questions_set) > 50:
        # Clear half the asked questions to allow repetition
        questions_to_remove = list(asked_questions_set)[:len(asked_questions_set)//2]
        for q_id in questions_to_remove:
            asked_questions_set.discard(q_id)
        
        # Try one more time with cleared history
        for topic in available_topics:
            questions = agent.question_bank.get_questions_by_topic(topic)
            available = [q for q in questions if q.id not in asked_questions_set]
            if available:
                return random.choice(available)
    
    # Fallback: return any question
    for topic in available_topics:
        questions = agent.question_bank.get_questions_by_topic(topic)
        if questions:
            return random.choice(questions)
    
    return None

def get_recent_topic_performance(topic, question_history):
    """Calculate recent performance for a specific topic"""
    if not question_history:
        return 0.5
    
    # Get recent questions for this topic (last 5)
    topic_questions = [
        q for q in question_history[-10:] 
        if hasattr(q, 'question') and q['question'].topic == topic
    ]
    
    if not topic_questions:
        return 0.5
    
    correct_count = sum(1 for q in topic_questions if q['correct'])
    return correct_count / len(topic_questions)

def create_student_profile(name, persona):
    """Create a student profile based on selected persona"""
    personas = {
        "Fast Learner": {
            "skill_level": np.random.uniform(0.7, 0.9),
            "learning_rate": np.random.uniform(0.15, 0.25),
            "consistency": np.random.uniform(0.8, 0.95)
        },
        "Steady Learner": {
            "skill_level": np.random.uniform(0.4, 0.7),
            "learning_rate": np.random.uniform(0.08, 0.15),
            "consistency": np.random.uniform(0.6, 0.8)
        },
        "Struggling Learner": {
            "skill_level": np.random.uniform(0.1, 0.4),
            "learning_rate": np.random.uniform(0.03, 0.08),
            "consistency": np.random.uniform(0.3, 0.6)
        },
        "Inconsistent Learner": {
            "skill_level": np.random.uniform(0.3, 0.8),
            "learning_rate": np.random.uniform(0.05, 0.20),
            "consistency": np.random.uniform(0.2, 0.5)
        }
    }
    
    profile_data = personas[persona]
    
    return StudentProfile(
        id=str(uuid.uuid4()),
        name=name,
        created_at=datetime.now(),
        last_active=datetime.now(),
        total_sessions=0,
        total_questions=0,
        overall_accuracy=0.0,
        skill_level=profile_data["skill_level"],
        learning_rate=profile_data["learning_rate"],
        knowledge_state={},
        performance_history=[],
        current_state=StudentState.NEW,
        preferences={"consistency": profile_data["consistency"]}
    )

def main():
    # Modern header with gradient background
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>🎓 AI Adaptive Tutorial Agent</h1>
        <p>Experience Personalized Learning with Reinforcement Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration with improved styling
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2 style="margin: 0; color: white;">🎯 Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model loading status
        if st.session_state.agent is None:
            if st.button("🚀 Load AI Models", type="primary"):
                with st.spinner("Loading AI models..."):
                    st.session_state.agent = load_models()
                    if st.session_state.agent:
                        st.success("✅ Models loaded successfully!")
                        st.rerun()
        else:
            st.success("✅ AI Models Ready")
        
        st.divider()
        
        # Student configuration
        st.subheader("👤 Student Simulation")
        
        student_name = st.text_input("Student Name", value="Alex Smith")
        
        persona = st.selectbox(
            "Learning Persona",
            ["Fast Learner", "Steady Learner", "Struggling Learner", "Inconsistent Learner"],
            help="Choose a student archetype to simulate different learning patterns"
        )
        
        if st.button("🎭 Create Student Profile"):
            profile = create_student_profile(student_name, persona)
            st.session_state.student_sim = StudentSimulator(profile)
            st.success(f"✅ Student profile created for {student_name}")
            st.rerun()
        
        if st.session_state.student_sim:
            st.info(f"**Current Student:** {st.session_state.student_sim.profile.name}")
            st.info(f"**Persona:** {persona}")
        
        st.divider()
        
        # Session controls
        st.subheader("📚 Learning Session")
        
        if st.session_state.agent and st.session_state.student_sim:
            if st.session_state.current_session is None:
                if st.button("▶️ Start Learning Session"):
                    session_id = str(uuid.uuid4())
                    st.session_state.current_session = SessionData(
                        session_id=session_id,
                        student_id=st.session_state.student_sim.profile.id,
                        start_time=datetime.now(),
                        end_time=None,
                        questions_attempted=0,
                        correct_answers=0,
                        topics_covered=[],
                        difficulty_progression=[],
                        rewards_earned=0.0,
                        session_state={}
                    )
                    # Clear question tracking for new session
                    st.session_state.asked_questions = set()
                    st.session_state.current_topic_index = 0
                    st.success("🎯 Session started!")
                    st.rerun()
            else:
                if st.button("⏹️ End Session"):
                    st.session_state.current_session.end_time = datetime.now()
                    st.session_state.current_session = None
                    st.success("📊 Session completed!")
                    st.rerun()
        else:
            st.warning("⚠️ Please load models and create student profile first")

    # Main content area
    if st.session_state.agent is None:
        st.info("👈 Please load the AI models from the sidebar to begin the demo.")
        
        # Show model architecture while waiting
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 REINFORCE Neural Network")
            st.write("""
            **Architecture:**
            - Input Layer: 10 features (student state)
            - Hidden Layers: 32 neurons each with ReLU activation
            - Output Layer: 5 difficulty levels with softmax
            - Dropout: 0.1 for regularization
            """)
        
        with col2:
            st.subheader("🎯 Thompson Sampling")
            st.write("""
            **Multi-Armed Bandit:**
            - Beta distributions for each difficulty level
            - Bayesian updating based on student responses
            - Exploration-exploitation balance
            - Contextual bandit with student features
            """)
        
        return
    
    if st.session_state.student_sim is None:
        st.info("👈 Please create a student profile from the sidebar to begin learning.")
        return
    
    # Main demo interface
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Interactive Learning", "📊 Real-time Analytics", "🧠 Model Insights", "📈 Performance Metrics"])
    
    with tab1:
        st.header("Interactive Learning Experience")
        
        if st.session_state.current_session is None:
            st.info("👈 Start a learning session from the sidebar to begin.")
        else:
            # Display current question or generate new one
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Current Question")
                
                if st.button("🎲 Get Next Question", type="primary"):
                    # Get next question from agent with session state tracking
                    question = get_next_unique_question(
                        st.session_state.agent,
                        st.session_state.student_sim.profile,
                        st.session_state.asked_questions
                    )
                    
                    if question:
                        st.session_state.current_question = question
                        st.session_state.asked_questions.add(question.id)
                        st.rerun()
                
                if hasattr(st.session_state, 'current_question'):
                    question = st.session_state.current_question
                    
                    # Enhanced question display with modern card design
                    st.markdown(f"""
                    <div class="question-card animate-fade-in">
                        <h3>📚 {question.topic.replace('_', ' ').title()}</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                            <span><strong>Difficulty:</strong> {question.difficulty:.2f}/1.0</span>
                            <span><strong>Question ID:</strong> {question.id}</span>
                        </div>
                        <div class="question-content">
                            <h4 style="color: white; margin: 0;">{question.content}</h4>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Answer input
                    student_answer = st.text_input("Your Answer:")
                    
                    col_correct, col_incorrect = st.columns(2)
                    
                    with col_correct:
                        if st.button("✅ Correct Answer"):
                            # Process correct answer
                            reward = st.session_state.agent.process_answer(
                                st.session_state.student_sim.profile,
                                question,
                                True
                            )
                            
                            # Update session data
                            st.session_state.current_session.questions_attempted += 1
                            st.session_state.current_session.correct_answers += 1
                            st.session_state.current_session.rewards_earned += reward
                            
                            # Add to history
                            st.session_state.question_history.append({
                                'question': question,
                                'correct': True,
                                'timestamp': datetime.now(),
                                'reward': reward
                            })
                            
                            st.success(f"🎉 Correct! Reward: {reward:.2f}")
                            st.rerun()
                    
                    with col_incorrect:
                        if st.button("❌ Incorrect Answer"):
                            # Process incorrect answer
                            reward = st.session_state.agent.process_answer(
                                st.session_state.student_sim.profile,
                                question,
                                False
                            )
                            
                            # Update session data
                            st.session_state.current_session.questions_attempted += 1
                            st.session_state.current_session.rewards_earned += reward
                            
                            # Add to history
                            st.session_state.question_history.append({
                                'question': question,
                                'correct': False,
                                'timestamp': datetime.now(),
                                'reward': reward
                            })
                            
                            st.error(f"❌ Incorrect. Reward: {reward:.2f}")
                            st.info(f"💡 Hint: {question.hints[0] if question.hints else 'Keep trying!'}")
                            st.rerun()
            
            with col2:
                st.subheader("📊 Session Stats")
                
                # Enhanced session statistics with cards
                
                if st.session_state.current_session:
                    session = st.session_state.current_session
                    
                    # Create beautiful metric cards
                    col_metric1, col_metric2 = st.columns(2)
                    
                    with col_metric1:
                        st.metric("Questions", session.questions_attempted, 
                                 delta=f"+{len(st.session_state.question_history)}" if st.session_state.question_history else None)
                        
                        if session.questions_attempted > 0:
                            accuracy = (session.correct_answers / session.questions_attempted) * 100
                            st.metric("Accuracy", f"{accuracy:.1f}%",
                                     delta=f"{'📈' if accuracy >= 70 else '📉' if accuracy >= 50 else '📊'}")
                    
                    with col_metric2:
                        st.metric("Correct", session.correct_answers,
                                 delta=f"+{session.correct_answers}" if session.correct_answers > 0 else None)
                        st.metric("Rewards", f"{session.rewards_earned:.1f}",
                                 delta=f"{'🎯' if session.rewards_earned > 0 else '💪'}")
                    
                    # Recent performance with enhanced display
                    if st.session_state.question_history:
                        recent_correct = sum(1 for q in st.session_state.question_history[-5:] if q['correct'])
                        recent_total = min(5, len(st.session_state.question_history))
                        
                        st.markdown(f"""
                        <div class="info-card glow-effect">
                            <h4 style="margin-top: 0; color: var(--primary-color);">🔥 Recent Streak</h4>
                            <p style="font-size: 1.2rem; margin: 0; color: var(--text-primary);"><strong>{recent_correct}/{recent_total}</strong> correct in last {recent_total} questions</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recent performance
                if st.session_state.question_history:
                    st.subheader("🕒 Recent Activity")
                    
                    for i, entry in enumerate(st.session_state.question_history[-3:]):
                        with st.expander(f"Question {len(st.session_state.question_history) - 2 + i}"):
                            st.write(f"Topic: {entry['question'].topic}")
                            st.write(f"Difficulty: {entry['question'].difficulty:.2f}")
                            st.write(f"Result: {'✅' if entry['correct'] else '❌'}")
                            st.write(f"Reward: {entry['reward']:.2f}")
    
    with tab2:
        st.header("Real-time Learning Analytics")
        
        if st.session_state.question_history:
            # Performance over time
            df = pd.DataFrame([
                {
                    'question_num': i + 1,
                    'correct': entry['correct'],
                    'difficulty': entry['question'].difficulty,
                    'reward': entry['reward'],
                    'topic': entry['question'].topic,
                    'timestamp': entry['timestamp']
                }
                for i, entry in enumerate(st.session_state.question_history)
            ])
            
            # Performance trend
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Performance Trend")
                
                fig = go.Figure()
                
                # Add accuracy trend (rolling average)
                window_size = min(5, len(df))
                rolling_accuracy = df['correct'].rolling(window=window_size, min_periods=1).mean()
                
                fig.add_trace(go.Scatter(
                    x=df['question_num'],
                    y=rolling_accuracy,
                    mode='lines+markers',
                    name='Accuracy (Rolling Avg)',
                    line=dict(color='#6366F1', width=3),
                    marker=dict(color='#6366F1', size=8)
                ))
                
                fig.update_layout(
                    title="Learning Progress",
                    xaxis_title="Question Number",
                    yaxis_title="Accuracy",
                    plot_bgcolor='rgba(28, 28, 30, 0.8)',
                    paper_bgcolor='rgba(28, 28, 30, 0.8)',
                    font=dict(color='#FFFFFF'),
                    title_font=dict(size=16, color='#6366F1'),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(range=[0, 1], gridcolor='rgba(255, 255, 255, 0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Difficulty Adaptation")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['question_num'],
                    y=df['difficulty'],
                    mode='lines+markers',
                    name='Question Difficulty',
                    line=dict(color='#8B5CF6', width=3),
                    marker=dict(
                        color=['#10B981' if x else '#EF4444' for x in df['correct']],
                        size=12,
                        line=dict(width=2, color='white')
                    )
                ))
                
                fig.update_layout(
                    title="Adaptive Difficulty",
                    xaxis_title="Question Number",
                    yaxis_title="Difficulty Level",
                    plot_bgcolor='rgba(28, 28, 30, 0.8)',
                    paper_bgcolor='rgba(28, 28, 30, 0.8)',
                    font=dict(color='#FFFFFF'),
                    title_font=dict(size=16, color='#8B5CF6'),
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(range=[0, 1], gridcolor='rgba(255, 255, 255, 0.1)')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Topic performance
            st.subheader("📚 Topic Performance")
            
            topic_stats = df.groupby('topic').agg({
                'correct': ['count', 'sum', 'mean'],
                'difficulty': 'mean'
            }).round(3)
            
            topic_stats.columns = ['Questions', 'Correct', 'Accuracy', 'Avg Difficulty']
            
            st.dataframe(topic_stats, use_container_width=True)
            
        else:
            st.info("📝 Start answering questions to see analytics data.")
    
    with tab3:
        st.header("AI Model Decision Making")
        
        if st.session_state.agent and st.session_state.student_sim:
            # Show current model state
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧠 REINFORCE Policy")
                
                # Get current policy distribution
                state_vector = st.session_state.agent.get_state_vector(
                    st.session_state.student_sim.profile
                )
                
                if state_vector is not None:
                    policy_probs = st.session_state.agent.policy_network(
                        torch.FloatTensor(state_vector).unsqueeze(0)
                    ).detach().numpy()[0]
                    
                    difficulty_levels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
                    
                    fig = go.Figure(data=[
                        go.Bar(x=difficulty_levels, y=policy_probs, marker_color='lightblue')
                    ])
                    
                    fig.update_layout(
                        title="Policy Distribution",
                        xaxis_title="Difficulty Level",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Student state representation
                st.subheader("👤 Student State Vector")
                profile = st.session_state.student_sim.profile
                
                state_info = {
                    "Skill Level": f"{profile.skill_level:.3f}",
                    "Learning Rate": f"{profile.learning_rate:.3f}",
                    "Overall Accuracy": f"{profile.overall_accuracy:.3f}",
                    "Total Questions": profile.total_questions,
                    "Current State": profile.current_state.value
                }
                
                for key, value in state_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.subheader("🎯 Thompson Sampling")
                
                # Show bandit arm parameters
                if hasattr(st.session_state.agent, 'thompson_sampler'):
                    sampler = st.session_state.agent.thompson_sampler
                    
                    if hasattr(sampler, 'alpha') and hasattr(sampler, 'beta'):
                        difficulty_levels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
                        
                        # Expected rewards (alpha / (alpha + beta))
                        # Handle both array and dict formats for alpha/beta
                        if isinstance(sampler.alpha, dict):
                            expected_rewards = [
                                sampler.alpha.get(i, 1.0) / (sampler.alpha.get(i, 1.0) + sampler.beta.get(i, 1.0))
                                for i in range(len(difficulty_levels))
                            ]
                        else:
                            expected_rewards = [
                                sampler.alpha[i] / (sampler.alpha[i] + sampler.beta[i])
                                for i in range(min(len(difficulty_levels), len(sampler.alpha)))
                            ]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=difficulty_levels, y=expected_rewards, marker_color='lightcoral')
                        ])
                        
                        fig.update_layout(
                            title="Expected Rewards by Difficulty",
                            xaxis_title="Difficulty Level",
                            yaxis_title="Expected Reward",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show confidence intervals
                        st.subheader("🔍 Bandit Arm Statistics")
                        
                        # Handle both array and dict formats for stats display
                        if isinstance(sampler.alpha, dict):
                            alpha_values = [sampler.alpha.get(i, 1.0) for i in range(len(difficulty_levels))]
                            beta_values = [sampler.beta.get(i, 1.0) for i in range(len(difficulty_levels))]
                            confidence_values = [
                                1 / (sampler.alpha.get(i, 1.0) + sampler.beta.get(i, 1.0))
                                for i in range(len(difficulty_levels))
                            ]
                        else:
                            alpha_values = list(sampler.alpha[:len(difficulty_levels)])
                            beta_values = list(sampler.beta[:len(difficulty_levels)])
                            confidence_values = [
                                1 / (sampler.alpha[i] + sampler.beta[i])
                                for i in range(min(len(difficulty_levels), len(sampler.alpha)))
                            ]
                        
                        stats_df = pd.DataFrame({
                            'Difficulty': difficulty_levels,
                            'Alpha': alpha_values,
                            'Beta': beta_values,
                            'Expected Reward': expected_rewards,
                            'Confidence': confidence_values
                        })
                        
                        st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("🤖 Load models and create student profile to see AI decision making.")
    
    with tab4:
        st.header("Training Performance Metrics")
        
        # Load and display training metrics
        try:
            with open("data/training_metrics.json", "r") as f:
                metrics = json.load(f)
            
            # Training progress
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Training Progress")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Reward Over Episodes', 'Loss Over Episodes'),
                    vertical_spacing=0.1
                )
                
                # Get data with proper handling - use actual keys from the file
                episodes = metrics.get('episode', [])
                avg_rewards = metrics.get('avg_reward', [])
                accuracy = metrics.get('accuracy', [])
                thompson_entropy = metrics.get('thompson_entropy', [])
                policy_entropy = metrics.get('policy_entropy', [])
                
                # Show data summary
                st.info(f"📊 Loaded training data: {len(episodes)} episodes with {len(avg_rewards)} reward measurements")
                
                # Only create plot if we have data
                if episodes and avg_rewards:
                    # Create a 2x2 subplot layout for multiple metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Average Reward', 'Accuracy', 'Thompson Entropy', 'Policy Entropy'),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1
                    )
                    
                    # Average Reward plot
                    fig.add_trace(
                        go.Scatter(
                            x=episodes,
                            y=avg_rewards,
                            mode='lines',
                            name='Avg Reward',
                            line=dict(color='#10B981', width=2),
                            hovertemplate='Episode: %{x}<br>Avg Reward: %{y:.3f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Accuracy plot
                    if accuracy:
                        fig.add_trace(
                            go.Scatter(
                                x=episodes,
                                y=accuracy,
                                mode='lines',
                                name='Accuracy',
                                line=dict(color='#3B82F6', width=2),
                                hovertemplate='Episode: %{x}<br>Accuracy: %{y:.3f}<extra></extra>'
                            ),
                            row=1, col=2
                        )
                    
                    # Thompson Entropy plot
                    if thompson_entropy:
                        fig.add_trace(
                            go.Scatter(
                                x=episodes,
                                y=thompson_entropy,
                                mode='lines',
                                name='Thompson Entropy',
                                line=dict(color='#8B5CF6', width=2),
                                hovertemplate='Episode: %{x}<br>Thompson Entropy: %{y:.3f}<extra></extra>'
                            ),
                            row=2, col=1
                        )
                    
                    # Policy Entropy plot
                    if policy_entropy:
                        fig.add_trace(
                            go.Scatter(
                                x=episodes,
                                y=policy_entropy,
                                mode='lines',
                                name='Policy Entropy',
                                line=dict(color='#EC4899', width=2),
                                hovertemplate='Episode: %{x}<br>Policy Entropy: %{y:.3f}<extra></extra>'
                            ),
                            row=2, col=2
                        )
                
                    # Enhanced layout for dark theme
                    fig.update_layout(
                        height=700, 
                        showlegend=False,
                        plot_bgcolor='rgba(30,41,59,0.3)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=10),
                        title_font=dict(size=14, color='white')
                    )
                    
                    # Update all axes with better styling
                    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title_text="Episode")
                    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No training data found to display. Please check that training_metrics.json contains valid episode and avg_reward data.")
            
            with col2:
                st.subheader("🎯 Model Performance")
                
                # Performance statistics
                if metrics.get('avg_reward'):
                    recent_rewards = metrics['avg_reward'][-100:]  # Last 100 episodes
                    
                    st.metric("Final Average Reward", f"{np.mean(recent_rewards):.3f}")
                    st.metric("Best Reward", f"{np.max(metrics['avg_reward']):.3f}")
                    st.metric("Training Episodes", len(metrics.get('episode', [])))
                
                # Convergence analysis
                st.subheader("📊 Learning Convergence")
                
                if metrics.get('avg_reward'):
                    rewards = np.array(metrics['avg_reward'])
                    window_size = min(100, len(rewards) // 10)
                    
                    if window_size > 1:
                        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(rewards))),
                            y=rewards,
                            mode='lines',
                            name='Raw Reward',
                            line=dict(color='#60A5FA', width=1),  # Light blue
                            opacity=0.4,
                            hovertemplate='Episode: %{x}<br>Raw Reward: %{y:.3f}<extra></extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(window_size-1, len(rewards))),
                            y=moving_avg,
                            mode='lines',
                            name=f'Moving Average ({window_size})',
                            line=dict(color='#1D4ED8', width=3),  # Dark blue
                            hovertemplate='Episode: %{x}<br>Moving Avg: %{y:.3f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Reward Convergence",
                            xaxis_title="Episode",
                            yaxis_title="Reward",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
        except FileNotFoundError:
            st.info("ℹ️ Training metrics file not found. The models are working properly - metrics display is optional.")
        except Exception as e:
            st.error(f"Error loading training metrics: {str(e)}")

if __name__ == "__main__":
    main()
