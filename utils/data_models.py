from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

class DifficultyLevel(Enum):
    """Enumeration for question difficulty levels"""
    VERY_EASY = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4

class StudentState(Enum):
    """Enumeration for student learning states"""
    NEW = "new"
    LEARNING = "learning"
    STRUGGLING = "struggling"
    MASTERING = "mastering"
    ADVANCED = "advanced"

@dataclass
class Question:
    """Represents a question/problem in the educational system"""
    id: str
    topic: str
    difficulty: float  # 0.0 to 1.0
    content: str
    answer: Any
    hints: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_difficulty_level(self) -> DifficultyLevel:
        """Convert continuous difficulty to discrete level"""
        if self.difficulty < 0.2:
            return DifficultyLevel.VERY_EASY
        elif self.difficulty < 0.4:
            return DifficultyLevel.EASY
        elif self.difficulty < 0.6:
            return DifficultyLevel.MEDIUM
        elif self.difficulty < 0.8:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.VERY_HARD

@dataclass
class StudentProfile:
    """Complete student profile with learning characteristics and history"""
    id: str
    name: str
    created_at: datetime
    last_active: datetime
    total_sessions: int = 0
    total_questions: int = 0
    overall_accuracy: float = 0.0
    skill_level: float = 0.5  # 0.0 to 1.0
    learning_rate: float = 0.1  # How fast they learn
    knowledge_state: Dict[str, float] = field(default_factory=dict)  # topic -> mastery
    performance_history: List[Dict] = field(default_factory=list)
    current_state: StudentState = StudentState.NEW
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def get_topic_mastery(self, topic: str) -> float:
        """Get mastery level for a specific topic"""
        return self.knowledge_state.get(topic, 0.0)
    
    def update_topic_mastery(self, topic: str, mastery: float):
        """Update mastery level for a topic"""
        self.knowledge_state[topic] = max(0.0, min(1.0, mastery))
    
    def get_weak_topics(self, threshold: float = 0.3) -> List[str]:
        """Get topics where student needs improvement"""
        return [topic for topic, mastery in self.knowledge_state.items() 
                if mastery < threshold]
    
    def get_strong_topics(self, threshold: float = 0.7) -> List[str]:
        """Get topics where student is proficient"""
        return [topic for topic, mastery in self.knowledge_state.items() 
                if mastery >= threshold]

@dataclass
class SessionData:
    """Data for a single learning session"""
    session_id: str
    student_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    questions_attempted: int = 0
    correct_answers: int = 0
    topics_covered: List[str] = field(default_factory=list)
    difficulty_progression: List[float] = field(default_factory=list)
    rewards_earned: float = 0.0
    session_state: Dict[str, Any] = field(default_factory=dict)
    
    def get_accuracy(self) -> float:
        """Calculate session accuracy"""
        if self.questions_attempted == 0:
            return 0.0
        return self.correct_answers / self.questions_attempted
    
    def get_duration_minutes(self) -> float:
        """Get session duration in minutes"""
        if self.end_time is None:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        
        duration = end_time - self.start_time
        return duration.total_seconds() / 60.0
    
    def add_question_result(self, topic: str, difficulty: float, correct: bool, reward: float):
        """Add a question result to the session"""
        self.questions_attempted += 1
        if correct:
            self.correct_answers += 1
        
        if topic not in self.topics_covered:
            self.topics_covered.append(topic)
        
        self.difficulty_progression.append(difficulty)
        self.rewards_earned += reward

@dataclass
class QuestionResponse:
    """Represents a student's response to a question"""
    question_id: str
    student_id: str
    response: Any
    is_correct: bool
    response_time: float  # seconds
    timestamp: datetime
    hints_used: int = 0
    attempts: int = 1
    confidence: Optional[float] = None  # Self-reported confidence 0-1
    
@dataclass
class LearningObjective:
    """Represents a learning goal or objective"""
    id: str
    title: str
    description: str
    topics: List[str]  # Related topics
    difficulty_range: tuple  # (min_difficulty, max_difficulty)
    prerequisites: List[str] = field(default_factory=list)  # Prerequisite objective IDs
    estimated_time_minutes: int = 30
    
@dataclass
class AdaptationEvent:
    """Records when and why the system adapted"""
    timestamp: datetime
    student_id: str
    event_type: str  # "difficulty_increase", "difficulty_decrease", "topic_switch", etc.
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    reason: str
    confidence: float  # How confident the system is in this adaptation

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for analysis"""
    student_id: str
    time_period_start: datetime
    time_period_end: datetime
    total_questions: int
    correct_answers: int
    accuracy: float
    average_difficulty: float
    topics_studied: List[str]
    learning_velocity: float  # Rate of improvement
    engagement_score: float  # Derived from session patterns
    adaptations_made: int
    
    def calculate_efficiency(self) -> float:
        """Calculate learning efficiency metric"""
        if self.total_questions == 0:
            return 0.0
        
        # Efficiency considers accuracy, difficulty, and learning velocity
        base_efficiency = self.accuracy * self.average_difficulty
        velocity_bonus = min(1.0, self.learning_velocity * 2.0)
        
        return (base_efficiency + velocity_bonus) / 2.0

# Utility functions for data models
def create_sample_student_profile(name: str, skill_level: float = 0.5) -> StudentProfile:
    """Create a sample student profile for testing"""
    return StudentProfile(
        id=f"student_{name.lower().replace(' ', '_')}",
        name=name,
        created_at=datetime.now(),
        last_active=datetime.now(),
        skill_level=skill_level,
        learning_rate=0.1,
        current_state=StudentState.NEW
    )

def create_sample_question(topic: str, difficulty: float, question_id: Optional[str] = None) -> Question:
    """Create a sample question for testing"""
    if question_id is None:
        question_id = f"{topic}_{int(difficulty*100)}"
    
    return Question(
        id=question_id,
        topic=topic,
        difficulty=difficulty,
        content=f"Sample question about {topic.replace('_', ' ')} at difficulty {difficulty:.1f}",
        answer="sample_answer",
        hints=[f"Think about the fundamentals of {topic.replace('_', ' ')}"],
        explanation=f"This question tests understanding of {topic.replace('_', ' ')} concepts"
    )
