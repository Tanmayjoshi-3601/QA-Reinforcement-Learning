import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pickle
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from utils.data_models import StudentProfile, Question, DifficultyLevel, StudentState
from utils.question_bank import QuestionBank

class PolicyNetwork(nn.Module):
    """Neural network for REINFORCE algorithm"""
    
    def __init__(self, state_dim=10, hidden_dim=32, action_dim=5):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs

class ThompsonSampler:
    """Thompson Sampling for multi-armed bandit"""
    
    def __init__(self, n_arms=5):
        self.n_arms = n_arms
        # Initialize Beta distribution parameters
        self.alpha = np.ones(n_arms)  # Success counts + 1
        self.beta = np.ones(n_arms)   # Failure counts + 1
        
    def select_arm(self, context=None):
        """Select an arm using Thompson sampling"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """Update the Beta distribution for the selected arm"""
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += abs(reward)
    
    def get_arm_statistics(self):
        """Get statistics for each arm"""
        expected_rewards = self.alpha / (self.alpha + self.beta)
        confidence = 1 / (self.alpha + self.beta)
        return expected_rewards, confidence

class AdaptiveTutorAgent:
    """Main agent that orchestrates adaptive tutoring using trained models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.policy_network = PolicyNetwork()
        self.thompson_sampler = ThompsonSampler()
        self.question_bank = QuestionBank()
        
        # Agent state
        self.current_topic = None
        self.session_data = {}
        self.asked_questions = set()  # Track asked questions to avoid repetition
        self.performance_history = []
        
        # Model loaded flag
        self.models_loaded = False
        
    def load_models(self, model_path: str):
        """Load pre-trained models from files"""
        try:
            model_dir = Path(model_path)
            
            # Load REINFORCE policy network
            reinforce_path = model_dir / "reinforce_model.pth"
            if reinforce_path.exists():
                checkpoint = torch.load(reinforce_path, map_location=self.device, weights_only=False)
                
                # Handle different checkpoint formats
                if 'policy_state_dict' in checkpoint:
                    # Checkpoint contains multiple components
                    self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"✅ REINFORCE model loaded from checkpoint: {reinforce_path}")
                elif isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
                    # Direct state dict
                    self.policy_network.load_state_dict(checkpoint)
                    print(f"✅ REINFORCE model loaded directly: {reinforce_path}")
                else:
                    print(f"⚠️ Unknown checkpoint format in {reinforce_path}, using untrained model")
                    
                self.policy_network.eval()
            
            # Load Thompson sampler
            thompson_path = model_dir / "thompson_sampler.pkl"
            if thompson_path.exists():
                with open(thompson_path, 'rb') as f:
                    sampler_data = pickle.load(f)
                    
                    # Handle different sampler formats
                    if isinstance(sampler_data, ThompsonSampler):
                        # Direct sampler object
                        self.thompson_sampler = sampler_data
                        print(f"✅ Thompson sampler object loaded from {thompson_path}")
                    elif isinstance(sampler_data, dict):
                        # Dictionary with parameters
                        self.thompson_sampler.alpha = sampler_data.get('alpha', np.ones(5))
                        self.thompson_sampler.beta = sampler_data.get('beta', np.ones(5))
                        print(f"✅ Thompson sampler parameters loaded from {thompson_path}")
                    else:
                        print(f"⚠️ Unknown sampler format in {thompson_path}, using default parameters")
            
            # Load training metrics (optional)
            metrics_path = model_dir / "training_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.training_metrics = json.load(f)
                print(f"✅ Training metrics loaded from {metrics_path}")
            else:
                print(f"ℹ️ Training metrics file not found at {metrics_path}, continuing without metrics")
                self.training_metrics = None
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def get_state_vector(self, student_profile: StudentProfile) -> np.ndarray:
        """Convert student profile to state vector for neural network"""
        # Create a comprehensive state representation
        state_features = [
            student_profile.skill_level,
            student_profile.learning_rate,
            student_profile.overall_accuracy,
            min(student_profile.total_questions / 100.0, 1.0),  # Normalized question count
            len(student_profile.knowledge_state) / 10.0,  # Normalized topic coverage
        ]
        
        # Add topic-specific knowledge (pad or truncate to 5 features)
        topic_knowledge = list(student_profile.knowledge_state.values())
        if len(topic_knowledge) < 5:
            topic_knowledge.extend([0.0] * (5 - len(topic_knowledge)))
        else:
            topic_knowledge = topic_knowledge[:5]
        
        state_features.extend(topic_knowledge)
        
        return np.array(state_features, dtype=np.float32)
    
    def select_difficulty_reinforce(self, student_profile: StudentProfile) -> int:
        """Select difficulty using REINFORCE policy"""
        if not self.models_loaded:
            return np.random.randint(0, 5)  # Random fallback
        
        state_vector = self.get_state_vector(student_profile)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            action_probs = self.policy_network(state_tensor)
            
            # Sample action from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            
            return int(action.item())
    
    def select_difficulty_thompson(self, student_profile: StudentProfile) -> int:
        """Select difficulty using Thompson sampling"""
        # Use student context for contextual bandit
        context = self.get_state_vector(student_profile)
        return self.thompson_sampler.select_arm(context)
    
    def get_next_question(self, student_profile: StudentProfile, 
                         method: str = "reinforce") -> Optional[Question]:
        """Get the next question for the student"""
        # Select topic (simple round-robin for demo)
        available_topics = self.question_bank.get_available_topics()
        if not available_topics:
            return None
        
        # Choose topic based on knowledge gaps
        topic_scores = {}
        for topic in available_topics:
            knowledge = student_profile.knowledge_state.get(topic, 0.0)
            topic_scores[topic] = 1.0 - knowledge  # Prioritize unknown topics
        
        # Select topic with highest knowledge gap
        selected_topic = max(topic_scores.keys(), key=lambda t: topic_scores[t])
        
        # Select difficulty using chosen method
        if method == "reinforce":
            difficulty_level = self.select_difficulty_reinforce(student_profile)
        else:
            difficulty_level = self.select_difficulty_thompson(student_profile)
        
        # Map difficulty level to continuous value
        difficulty_ranges = {
            0: (0.0, 0.2),   # Very Easy
            1: (0.2, 0.4),   # Easy
            2: (0.4, 0.6),   # Medium
            3: (0.6, 0.8),   # Hard
            4: (0.8, 1.0)    # Very Hard
        }
        
        min_diff, max_diff = difficulty_ranges[difficulty_level]
        target_difficulty = (min_diff + max_diff) / 2.0
        
        # Get question from question bank, avoiding already asked questions
        attempts = 0
        max_attempts = 10
        question = None
        
        while attempts < max_attempts:
            candidate_question = self.question_bank.get_question(selected_topic, target_difficulty)
            if candidate_question and candidate_question.id not in self.asked_questions:
                question = candidate_question
                break
            # If we've asked this question, try slightly different difficulty
            target_difficulty += random.uniform(-0.1, 0.1)
            target_difficulty = max(0.0, min(1.0, target_difficulty))
            attempts += 1
        
        # If we couldn't find a new question, clear some history and try again
        if not question:
            if len(self.asked_questions) > 20:  # Clear older questions
                old_questions = list(self.asked_questions)[:10]
                for q_id in old_questions:
                    self.asked_questions.discard(q_id)
            question = self.question_bank.get_question(selected_topic, target_difficulty)
        
        if question:
            self.asked_questions.add(question.id)
            # Store current question info
            self.current_topic = selected_topic
            self.current_difficulty_level = difficulty_level
        
        return question
    
    def process_answer(self, student_profile: StudentProfile, 
                      question: Question, is_correct: bool) -> float:
        """Process student's answer and update models"""
        # Calculate reward based on correctness and adaptive learning principles
        if is_correct:
            # Positive reward for correct answers, scaled by difficulty
            reward = 1.0 + (question.difficulty * 0.5)
        else:
            # Negative reward for incorrect answers, but not too harsh
            reward = -0.3 - (question.difficulty * 0.2)
        
        # Update Thompson sampler with proper reward signaling
        if hasattr(self, 'current_difficulty_level'):
            # For Thompson sampling, we need to signal whether this difficulty level was appropriate
            # If correct on hard question -> good difficulty choice
            # If incorrect on easy question -> bad difficulty choice
            ts_reward = self._calculate_thompson_reward(is_correct, question.difficulty, student_profile)
            self.thompson_sampler.update(self.current_difficulty_level, ts_reward)
        
        # Update student profile
        self._update_student_profile(student_profile, question, is_correct, reward)
        
        # Store performance data
        self.performance_history.append({
            'timestamp': datetime.now(),
            'question_id': question.id,
            'topic': question.topic,
            'difficulty': question.difficulty,
            'correct': is_correct,
            'reward': reward,
            'student_id': student_profile.id
        })
        
        return reward
    
    def _calculate_thompson_reward(self, is_correct: bool, difficulty: float, student_profile: StudentProfile) -> float:
        """Calculate reward for Thompson sampler based on whether difficulty was appropriate"""
        # Get student's estimated ability in this topic
        topic_knowledge = student_profile.knowledge_state.get(self.current_topic, 0.5)
        
        # Ideal difficulty should match student ability
        difficulty_appropriateness = 1.0 - abs(difficulty - topic_knowledge)
        
        if is_correct:
            # Correct answer: reward if difficulty was challenging but fair
            return difficulty_appropriateness * (0.5 + difficulty * 0.5)
        else:
            # Incorrect answer: penalize if difficulty was too hard, smaller penalty if appropriate
            if difficulty > topic_knowledge + 0.3:  # Too hard
                return -1.0
            else:  # Appropriately challenging
                return -0.2
    
    def _update_student_profile(self, student_profile: StudentProfile, 
                               question: Question, is_correct: bool, reward: float):
        """Update student profile based on performance"""
        # Update total questions and accuracy
        student_profile.total_questions += 1
        
        if is_correct:
            student_profile.overall_accuracy = (
                (student_profile.overall_accuracy * (student_profile.total_questions - 1) + 1.0) 
                / student_profile.total_questions
            )
        else:
            student_profile.overall_accuracy = (
                (student_profile.overall_accuracy * (student_profile.total_questions - 1)) 
                / student_profile.total_questions
            )
        
        # Update topic-specific knowledge
        topic = question.topic
        current_knowledge = student_profile.knowledge_state.get(topic, 0.0)
        
        # Learning rate affects knowledge update
        learning_impact = student_profile.learning_rate * (1.0 if is_correct else 0.5)
        difficulty_factor = 1.0 + question.difficulty  # Harder questions teach more
        
        knowledge_update = learning_impact * difficulty_factor * 0.1
        
        if is_correct:
            new_knowledge = min(1.0, current_knowledge + knowledge_update)
        else:
            new_knowledge = max(0.0, current_knowledge - knowledge_update * 0.5)
        
        student_profile.knowledge_state[topic] = new_knowledge
        
        # Update student state based on recent performance
        self._update_student_state(student_profile)
        
        # Update last active timestamp
        student_profile.last_active = datetime.now()
    
    def _update_student_state(self, student_profile: StudentProfile):
        """Update student's learning state based on recent performance"""
        if len(self.performance_history) < 3:
            return
        
        # Analyze recent performance (last 5 questions)
        recent_performance = [
            entry for entry in self.performance_history[-5:]
            if entry['student_id'] == student_profile.id
        ]
        
        if not recent_performance:
            return
        
        recent_accuracy = sum(1 for entry in recent_performance if entry['correct']) / len(recent_performance)
        avg_difficulty = sum(entry['difficulty'] for entry in recent_performance) / len(recent_performance)
        
        # Determine new state
        if recent_accuracy >= 0.8 and avg_difficulty > 0.6:
            student_profile.current_state = StudentState.ADVANCED
        elif recent_accuracy >= 0.7:
            student_profile.current_state = StudentState.MASTERING
        elif recent_accuracy >= 0.4:
            student_profile.current_state = StudentState.LEARNING
        elif recent_accuracy < 0.3:
            student_profile.current_state = StudentState.STRUGGLING
        else:
            student_profile.current_state = StudentState.LEARNING
    
    def get_student_analytics(self, student_profile: StudentProfile) -> Dict:
        """Generate comprehensive analytics for a student"""
        student_history = [
            entry for entry in self.performance_history
            if entry['student_id'] == student_profile.id
        ]
        
        if not student_history:
            return {"message": "No performance data available"}
        
        # Calculate analytics
        total_questions = len(student_history)
        correct_answers = sum(1 for entry in student_history if entry['correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Topic performance
        topic_performance = {}
        for entry in student_history:
            topic = entry['topic']
            if topic not in topic_performance:
                topic_performance[topic] = {'total': 0, 'correct': 0}
            
            topic_performance[topic]['total'] += 1
            if entry['correct']:
                topic_performance[topic]['correct'] += 1
        
        # Calculate topic accuracies
        for topic in topic_performance:
            total = topic_performance[topic]['total']
            correct = topic_performance[topic]['correct']
            topic_performance[topic]['accuracy'] = correct / total if total > 0 else 0
        
        # Difficulty progression
        difficulties = [entry['difficulty'] for entry in student_history]
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 0
        
        # Learning trajectory
        recent_accuracy = accuracy
        if len(student_history) >= 10:
            recent_history = student_history[-10:]
            recent_correct = sum(1 for entry in recent_history if entry['correct'])
            recent_accuracy = recent_correct / len(recent_history)
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'recent_accuracy': recent_accuracy,
            'average_difficulty': avg_difficulty,
            'topic_performance': topic_performance,
            'current_state': student_profile.current_state.value,
            'knowledge_state': student_profile.knowledge_state,
            'skill_level': student_profile.skill_level,
            'learning_rate': student_profile.learning_rate
        }
