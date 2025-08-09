# AI Adaptive Tutorial Agent Demo

## Overview

This is an AI-powered adaptive learning system that uses reinforcement learning to personalize educational content delivery. The system combines multiple AI techniques including REINFORCE policy gradient methods and Thompson Sampling to create an intelligent tutoring agent that adapts to individual student learning patterns and performance in real-time.

The application simulates both student behavior and tutor decision-making, providing a comprehensive demonstration of how AI can be used to optimize educational experiences through personalized difficulty adjustment, topic selection, and performance tracking.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive web interface for demonstration and visualization
- **Multi-page Layout**: Wide layout with expandable sidebar for navigation and controls
- **Real-time Visualization**: Plotly-based charts and graphs for performance analytics and training metrics
- **Session Management**: Persistent state management across user interactions

### AI/ML Core Components
- **REINFORCE Policy Network**: Neural network using policy gradient methods for action selection
- **Thompson Sampling**: Multi-armed bandit approach for exploration-exploitation balance
- **Student Simulator**: Behavioral model that simulates realistic student learning patterns, fatigue, and motivation
- **Adaptive Question Selection**: Intelligent content delivery based on student state and performance

### Data Models
- **Student Profiling**: Comprehensive student state tracking including knowledge levels, learning preferences, and historical performance
- **Question Management**: Structured question bank with difficulty levels, topics, hints, and metadata
- **Session Tracking**: Detailed logging of learning sessions with performance metrics and analytics

### Analytics & Monitoring
- **Performance Analytics**: Real-time tracking of student progress, accuracy rates, and response times
- **Learning Curve Analysis**: Visualization of improvement patterns across different topics and difficulty levels
- **Adaptation Events**: Logging and analysis of AI decision-making processes

### Architecture Patterns
- **Modular Design**: Clear separation between models, utilities, and application logic
- **State Management**: Centralized session state handling for consistent user experience
- **Mock Data Integration**: Fallback systems for demonstration when actual trained models are unavailable

## External Dependencies

### Python Libraries
- **PyTorch**: Deep learning framework for neural network implementation and training
- **Streamlit**: Web application framework for interactive demonstrations
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Plotly**: Interactive visualization and charting library

### Data Storage
- **JSON Files**: Configuration data, training metrics, and mock data storage
- **Pickle Files**: Serialized model storage and loading (when available)
- **In-Memory State**: Session data and temporary analytics storage

### AI/ML Dependencies
- **PyTorch Neural Networks**: Custom policy networks for reinforcement learning
- **Statistical Sampling**: Thompson Sampling implementation for multi-armed bandits
- **Performance Tracking**: Built-in analytics and metrics collection systems

## Recent Changes (August 2025)

✅ **Question Repetition Issue Resolved**: Implemented comprehensive session state tracking to prevent question repetition across Streamlit reruns

✅ **Adaptive Difficulty Fixed**: Corrected reward calculation logic so steady learners properly increase difficulty on correct answers

✅ **Project Structure Reorganized**: 
- Moved all attached model files from `attached_assets/` to proper `data/` directory
- Added Jupyter notebooks to `notebooks/` directory  
- Updated model loading paths from `mock_data/` to `data/`
- Created comprehensive README.md with project documentation
- Added proper .gitignore for version control

✅ **Git Repository Ready**: All files properly organized and ready for git push with clean project structure

✅ **Subtle Dark Theme UI**: Completely redesigned with professional dark theme featuring:
- Elegant indigo/violet color scheme (#6366F1, #8B5CF6, #EC4899)
- Subtle glassmorphism effects with backdrop blur
- Gentle glow animations and smooth transitions
- Enhanced dark charts with professional styling
- Clean typography and refined spacing

✅ **Model Loading Issues Completely Fixed** (August 8, 2025):
- Fixed AttributeError with 'estimated_time' by replacing with Question ID display
- Added `weights_only=False` parameter to torch.load() for PyTorch 2.6 compatibility
- Implemented intelligent checkpoint format detection for REINFORCE models
- Added robust Thompson sampler format handling (both dict and array formats)
- Fixed hardcoded "mock_data/" path references, updated all to use "data/"
- Enhanced error handling with informative messages instead of warnings
- Made training metrics optional with graceful degradation

✅ **Dependencies Documentation**: Created comprehensive `dependencies.txt` file with all required packages and installation instructions for both local development and Replit deployment

✅ **Training Metrics Display Fixed** (August 8, 2025):
- Corrected data key mapping: using 'avg_reward' instead of 'reward' to match actual file structure
- Implemented comprehensive 2x2 subplot layout showing all available metrics (avg_reward, accuracy, thompson_entropy, policy_entropy)
- Enhanced plot visibility with proper dark theme colors and styling
- Fixed performance statistics to use correct data keys
- Added data validation and informative error messages

The system now uses the actual trained models provided by the user, loads without any errors, displays all training metrics properly, and includes all original research notebooks and evaluation results with a stunning modern interface.