import numpy as np
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from utils.data_models import StudentProfile, Question, StudentState

class StudentSimulator:
    """Simulates realistic student behavior and learning patterns"""
    
    def __init__(self, profile: StudentProfile):
        self.profile = profile
        self.fatigue_level = 0.0  # Increases over time, affects performance
        self.motivation_level = 0.8  # Affects engagement
        self.session_start_time = None
        self.questions_in_session = 0
        
        # Learning curves for different topics
        self.topic_difficulty_preferences = {}
        self._initialize_preferences()
    
    def _initialize_preferences(self):
        """Initialize student's preferences and strengths"""
        # Some students are better at certain topics
        topics = ["algebra_basics", "linear_equations", "quadratic_equations", 
                 "geometry", "trigonometry", "calculus_intro"]
        
        for topic in topics:
            # Random preference between 0.1 and 1.0
            preference = np.random.uniform(0.1, 1.0)
            self.topic_difficulty_preferences[topic] = preference
            
            # Initialize knowledge state if not exists
            if topic not in self.profile.knowledge_state:
                # Base knowledge influenced by skill level and topic preference
                base_knowledge = self.profile.skill_level * preference * 0.5
                self.profile.knowledge_state[topic] = max(0.0, min(1.0, base_knowledge))
    
    def start_session(self):
        """Start a new learning session"""
        self.session_start_time = datetime.now()
        self.questions_in_session = 0
        self.fatigue_level = 0.0
        self.motivation_level = min(1.0, self.motivation_level + 0.1)  # Refresh motivation
    
    def answer_question(self, question: Question) -> bool:
        """Simulate student answering a question"""
        self.questions_in_session += 1
        
        # Calculate probability of correct answer
        correct_probability = self._calculate_answer_probability(question)
        
        # Simulate answer
        is_correct = random.random() < correct_probability
        
        # Update internal state
        self._update_internal_state(question, is_correct)
        
        return is_correct
    
    def _calculate_answer_probability(self, question: Question) -> float:
        """Calculate probability of answering correctly based on multiple factors"""
        # Base probability from topic knowledge
        topic_knowledge = self.profile.knowledge_state.get(question.topic, 0.0)
        base_prob = topic_knowledge
        
        # Difficulty adjustment
        difficulty_factor = 1.0 - question.difficulty
        topic_preference = self.topic_difficulty_preferences.get(question.topic, 0.5)
        
        # Skill level influence
        skill_factor = self.profile.skill_level
        
        # Consistency factor (some students are more consistent)
        consistency = self.profile.preferences.get("consistency", 0.7)
        consistency_noise = np.random.normal(0, 1.0 - consistency)
        
        # Fatigue and motivation effects
        fatigue_penalty = self.fatigue_level * 0.3
        motivation_boost = (self.motivation_level - 0.5) * 0.2
        
        # Session length effect (performance degrades over long sessions)
        session_penalty = min(0.2, self.questions_in_session * 0.01)
        
        # Combine all factors
        final_probability = (
            base_prob * 0.4 +                    # 40% from topic knowledge
            difficulty_factor * 0.2 +            # 20% from difficulty
            skill_factor * 0.2 +                 # 20% from general skill
            topic_preference * 0.1 +             # 10% from topic preference
            consistency_noise * 0.1              # 10% consistency variation
        )
        
        # Apply session effects
        final_probability += motivation_boost - fatigue_penalty - session_penalty
        
        # Ensure probability is in valid range
        return max(0.01, min(0.99, final_probability))
    
    def _update_internal_state(self, question: Question, is_correct: bool):
        """Update student's internal learning state"""
        # Update fatigue (increases with each question)
        self.fatigue_level = min(1.0, self.fatigue_level + 0.02)
        
        # Update motivation based on performance
        if is_correct:
            # Correct answers boost motivation, but with diminishing returns
            motivation_boost = 0.05 * (1.0 - self.motivation_level)
            self.motivation_level = min(1.0, self.motivation_level + motivation_boost)
        else:
            # Incorrect answers reduce motivation
            motivation_loss = 0.03 + (question.difficulty * 0.02)
            self.motivation_level = max(0.1, self.motivation_level - motivation_loss)
        
        # Learning occurs (knowledge state updates happen in the agent)
        # But we can update internal topic preferences
        topic = question.topic
        
        if is_correct and question.difficulty > 0.6:
            # Boost preference for topics where student succeeds at hard questions
            current_pref = self.topic_difficulty_preferences.get(topic, 0.5)
            self.topic_difficulty_preferences[topic] = min(1.0, current_pref + 0.01)
        elif not is_correct and question.difficulty < 0.4:
            # Reduce preference for topics where student fails easy questions
            current_pref = self.topic_difficulty_preferences.get(topic, 0.5)
            self.topic_difficulty_preferences[topic] = max(0.1, current_pref - 0.01)
    
    def get_learning_state(self) -> Dict:
        """Get current learning state for analytics"""
        session_duration = 0
        if self.session_start_time:
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        
        return {
            'fatigue_level': self.fatigue_level,
            'motivation_level': self.motivation_level,
            'session_duration_minutes': session_duration,
            'questions_in_session': self.questions_in_session,
            'topic_preferences': self.topic_difficulty_preferences,
            'estimated_performance': self._estimate_current_performance()
        }
    
    def _estimate_current_performance(self) -> float:
        """Estimate current performance based on state"""
        base_performance = self.profile.skill_level
        
        # Apply current state modifiers
        motivation_effect = (self.motivation_level - 0.5) * 0.3
        fatigue_effect = -self.fatigue_level * 0.2
        
        estimated = base_performance + motivation_effect + fatigue_effect
        return max(0.0, min(1.0, estimated))
    
    def take_break(self, duration_minutes: float = 5.0):
        """Simulate taking a break to reduce fatigue"""
        # Reduce fatigue based on break duration
        fatigue_reduction = min(self.fatigue_level, duration_minutes * 0.1)
        self.fatigue_level -= fatigue_reduction
        
        # Small motivation boost from break
        self.motivation_level = min(1.0, self.motivation_level + 0.05)
    
    def end_session(self):
        """End the current learning session"""
        if self.session_start_time:
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
            
            # Update profile based on session
            self.profile.total_sessions += 1
            
            # Reset session state
            self.session_start_time = None
            self.questions_in_session = 0
            self.fatigue_level = 0.0
            
            return {
                'session_duration': session_duration,
                'questions_attempted': self.questions_in_session,
                'final_motivation': self.motivation_level,
                'final_fatigue': self.fatigue_level
            }
        
        return None
    
    def get_personalized_feedback(self, recent_performance: List[Dict]) -> Dict:
        """Generate personalized feedback based on recent performance"""
        if not recent_performance:
            return {"message": "Keep learning! You're just getting started."}
        
        # Analyze recent performance
        recent_accuracy = sum(1 for p in recent_performance if p.get('correct', False)) / len(recent_performance)
        avg_difficulty = sum(p.get('difficulty', 0.5) for p in recent_performance) / len(recent_performance)
        
        feedback = {}
        
        # Performance feedback
        if recent_accuracy >= 0.8:
            feedback['performance'] = "Excellent work! You're mastering these concepts."
            feedback['suggestion'] = "Ready for more challenging questions?"
        elif recent_accuracy >= 0.6:
            feedback['performance'] = "Good progress! You're learning steadily."
            feedback['suggestion'] = "Keep practicing to build confidence."
        elif recent_accuracy >= 0.4:
            feedback['performance'] = "You're working hard! Learning takes time."
            feedback['suggestion'] = "Consider reviewing fundamentals or taking a short break."
        else:
            feedback['performance'] = "Don't get discouraged! Everyone learns at their own pace."
            feedback['suggestion'] = "Let's try some easier questions to build momentum."
        
        # Difficulty feedback
        if avg_difficulty > 0.7 and recent_accuracy < 0.5:
            feedback['difficulty'] = "The questions might be too challenging right now."
        elif avg_difficulty < 0.3 and recent_accuracy > 0.8:
            feedback['difficulty'] = "You're ready for more challenging material!"
        
        # Motivation feedback
        if self.motivation_level < 0.4:
            feedback['motivation'] = "Take a break when you need it. Learning should be enjoyable!"
        elif self.motivation_level > 0.8:
            feedback['motivation'] = "Your enthusiasm is great! Keep up the positive attitude."
        
        # Fatigue feedback
        if self.fatigue_level > 0.6:
            feedback['fatigue'] = "You've been working hard! Consider taking a short break."
        
        return feedback
    
    def simulate_learning_over_time(self, days: int = 30) -> List[Dict]:
        """Simulate learning progress over multiple days"""
        progress = []
        current_date = datetime.now()
        
        for day in range(days):
            # Simulate daily learning session
            daily_accuracy = max(0.1, min(0.9, 
                self.profile.skill_level + 
                np.random.normal(0, 0.1) +
                (day * self.profile.learning_rate * 0.01)  # Gradual improvement
            ))
            
            # Random number of questions per day (5-15)
            daily_questions = random.randint(5, 15)
            
            progress.append({
                'date': current_date + timedelta(days=day),
                'questions_attempted': daily_questions,
                'accuracy': daily_accuracy,
                'skill_level': min(1.0, self.profile.skill_level + (day * self.profile.learning_rate * 0.005)),
                'topics_studied': random.sample(list(self.topic_difficulty_preferences.keys()), 
                                              random.randint(1, 3))
            })
        
        return progress
