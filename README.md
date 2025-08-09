# AI Adaptive Tutorial Agent

An interactive Streamlit demonstration showcasing an AI-powered adaptive learning system that uses reinforcement learning to personalize educational content delivery. The system combines multiple AI techniques including REINFORCE policy gradient methods and Thompson Sampling to create an intelligent tutoring agent that adapts to individual student learning patterns and performance in real-time.

## Features

- **🧠 REINFORCE Policy Network**: Neural network using policy gradient methods for adaptive question selection
- **🎯 Thompson Sampling**: Multi-armed bandit approach for exploration-exploitation balance in difficulty selection
- **👤 Student Simulation**: Realistic behavioral models simulating different learning personas (Fast, Steady, Struggling, Inconsistent)
- **📊 Real-time Analytics**: Comprehensive performance tracking and visualization
- **🎓 Adaptive Content**: Dynamic question selection across 6 math topics with 900+ questions
- **📈 Learning Progression**: Intelligent difficulty adaptation based on student performance

## Project Structure

```
├── app.py                      # Main Streamlit application
├── models/                     # AI/ML model implementations
│   ├── adaptive_agent.py       # Main agent orchestrating adaptive tutoring
│   └── student_simulator.py    # Student behavior simulation
├── utils/                      # Utility modules
│   ├── data_models.py          # Data structures and models
│   ├── question_bank.py        # Question management and selection
│   └── analytics.py            # Analytics and visualization
├── data/                       # Model files and data
│   ├── reinforce_model.pth     # Trained REINFORCE policy network
│   ├── thompson_sampler.pkl    # Thompson sampler parameters
│   ├── training_metrics.json   # Training performance metrics
│   └── evaluation_results.pkl  # Model evaluation results
├── notebooks/                  # Jupyter notebooks
│   ├── model-training.ipynb    # Model training and development
│   └── agent-integration.ipynb # Agent integration and testing
├── dependencies.txt            # List of required Python packages
└── replit.md                   # Project documentation and architecture
```

## Quick Start

1. **Install Dependencies** (automatically handled in Replit):
   ```bash
   pip install streamlit torch numpy pandas plotly scikit-learn
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

3. **Use the Demo**:
   - Load AI models from the sidebar
   - Create a student profile with different learning personas
   - Start a learning session
   - Experience adaptive question selection and difficulty adjustment

## Educational Topics

The system covers 6 comprehensive math topics:
- **Algebra Basics**: Fundamental algebraic concepts and operations
- **Linear Equations**: Solving and graphing linear equations
- **Quadratic Equations**: Solving quadratic equations and understanding parabolas
- **Geometry**: Shapes, areas, volumes, and geometric proofs
- **Trigonometry**: Trigonometric functions and their applications
- **Calculus Introduction**: Limits, derivatives, and basic integration

## AI/ML Architecture

### REINFORCE Policy Network
- Input Layer: 10 features representing student state
- Hidden Layers: 32 neurons each with ReLU activation
- Output Layer: 5 difficulty levels with softmax distribution
- Dropout: 0.1 for regularization

### Thompson Sampling
- Beta distributions for each difficulty level
- Bayesian updating based on student responses
- Exploration-exploitation balance optimization
- Contextual bandit with student features

### Student Simulation
- Multiple learning personas with different characteristics
- Adaptive performance based on question difficulty
- Realistic learning progression and fatigue modeling
- Performance consistency variation

## Key Features Implemented

✅ **Question Variety**: Advanced tracking prevents repetition across sessions
✅ **Adaptive Difficulty**: Real-time adjustment based on student performance  
✅ **Topic Rotation**: Intelligent cycling through different subject areas
✅ **Performance Analytics**: Comprehensive tracking and visualization
✅ **Session Management**: Persistent state across interactions
✅ **Multiple Personas**: Support for different learning styles and abilities

## Development

The project uses modern Python development practices:
- **Modular Architecture**: Clean separation of concerns
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and comments
- **Testing**: Simulation-based testing and validation
- **Version Control**: Git-ready project structure

## License

This project is licensed under the MIT License - see the LICENSE file for details.
