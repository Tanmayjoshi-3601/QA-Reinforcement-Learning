import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from utils.data_models import StudentProfile, Question, SessionData, PerformanceMetrics

class AnalyticsDashboard:
    """Comprehensive analytics for the adaptive learning system"""
    
    def __init__(self):
        self.performance_data = []
        self.session_data = []
        self.adaptation_events = []
    
    def add_performance_record(self, student_id: str, question: Question, 
                             is_correct: bool, response_time: float, 
                             timestamp: Optional[datetime] = None):
        """Add a performance record for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'student_id': student_id,
            'timestamp': timestamp,
            'question_id': question.id,
            'topic': question.topic,
            'difficulty': question.difficulty,
            'correct': is_correct,
            'response_time': response_time
        }
        
        self.performance_data.append(record)
    
    def add_session_record(self, session: SessionData):
        """Add a session record for analysis"""
        self.session_data.append({
            'session_id': session.session_id,
            'student_id': session.student_id,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'duration_minutes': session.get_duration_minutes(),
            'questions_attempted': session.questions_attempted,
            'accuracy': session.get_accuracy(),
            'topics_covered': session.topics_covered,
            'rewards_earned': session.rewards_earned
        })
    
    def generate_student_progress_chart(self, student_id: str) -> go.Figure:
        """Generate a comprehensive progress chart for a student"""
        # Filter data for specific student
        student_data = [d for d in self.performance_data if d['student_id'] == student_id]
        
        if not student_data:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for this student",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(student_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Accuracy Over Time', 'Difficulty Progression',
                'Response Time Trends', 'Topic Performance',
                'Learning Velocity', 'Performance Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Accuracy over time (rolling average)
        window_size = min(10, len(df))
        df['rolling_accuracy'] = df['correct'].rolling(window=window_size, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rolling_accuracy'],
                mode='lines+markers',
                name='Rolling Accuracy',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        # 2. Difficulty progression
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['difficulty'],
                mode='lines+markers',
                name='Question Difficulty',
                line=dict(color='blue'),
                marker=dict(
                    color=['green' if x else 'red' for x in df['correct']],
                    size=6
                )
            ),
            row=1, col=2
        )
        
        # 3. Response time trends
        df['rolling_response_time'] = df['response_time'].rolling(window=window_size, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rolling_response_time'],
                mode='lines+markers',
                name='Avg Response Time',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # 4. Topic performance
        topic_performance = df.groupby('topic')['correct'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=topic_performance['topic'],
                y=topic_performance['correct'],
                name='Topic Accuracy',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        # 5. Learning velocity (improvement rate)
        if len(df) > 5:
            # Calculate trend in accuracy
            recent_period = df.tail(10)
            early_period = df.head(10)
            
            recent_accuracy = recent_period['correct'].mean()
            early_accuracy = early_period['correct'].mean()
            improvement = recent_accuracy - early_accuracy
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=improvement,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Learning Velocity"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-0.5, 0.5]},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-0.5, 0], 'color': 'lightgray'},
                            {'range': [0, 0.5], 'color': 'lightgreen'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.2
                        }
                    }
                ),
                row=3, col=1
            )
        
        # 6. Performance distribution
        fig.add_trace(
            go.Histogram(
                x=df['difficulty'],
                y=df['correct'],
                histfunc="avg",
                name='Accuracy by Difficulty',
                marker_color='purple',
                opacity=0.7
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Learning Analytics Dashboard - Student {student_id}"
        )
        
        return fig
    
    def generate_topic_mastery_heatmap(self, student_profiles: List[StudentProfile]) -> go.Figure:
        """Generate a heatmap showing topic mastery across students"""
        if not student_profiles:
            fig = go.Figure()
            fig.add_annotation(
                text="No student data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Collect all topics
        all_topics = set()
        for profile in student_profiles:
            all_topics.update(profile.knowledge_state.keys())
        
        all_topics = sorted(list(all_topics))
        
        # Create mastery matrix
        mastery_matrix = []
        student_names = []
        
        for profile in student_profiles:
            student_names.append(profile.name)
            mastery_row = []
            for topic in all_topics:
                mastery_row.append(profile.knowledge_state.get(topic, 0.0))
            mastery_matrix.append(mastery_row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=mastery_matrix,
            x=all_topics,
            y=student_names,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Mastery Level")
        ))
        
        fig.update_layout(
            title="Topic Mastery Heatmap",
            xaxis_title="Topics",
            yaxis_title="Students",
            height=max(400, len(student_names) * 40)
        )
        
        return fig
    
    def calculate_learning_metrics(self, student_id: str, 
                                 time_window_days: int = 7) -> Dict:
        """Calculate comprehensive learning metrics for a student"""
        # Filter recent data
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_data = [
            d for d in self.performance_data 
            if d['student_id'] == student_id and d['timestamp'] >= cutoff_date
        ]
        
        if not recent_data:
            return {"error": "No recent data available"}
        
        df = pd.DataFrame(recent_data)
        
        # Basic metrics
        total_questions = len(df)
        correct_answers = df['correct'].sum()
        accuracy = correct_answers / total_questions
        
        # Response time metrics
        avg_response_time = df['response_time'].mean()
        response_time_std = df['response_time'].std()
        
        # Difficulty metrics
        avg_difficulty = df['difficulty'].mean()
        difficulty_std = df['difficulty'].std()
        
        # Learning velocity (trend in accuracy)
        if len(df) >= 5:
            # Split into two halves and compare
            mid_point = len(df) // 2
            early_accuracy = df.iloc[:mid_point]['correct'].mean()
            recent_accuracy = df.iloc[mid_point:]['correct'].mean()
            learning_velocity = recent_accuracy - early_accuracy
        else:
            learning_velocity = 0.0
        
        # Topic performance
        topic_stats = df.groupby('topic').agg({
            'correct': ['count', 'sum', 'mean'],
            'difficulty': 'mean',
            'response_time': 'mean'
        }).round(3)
        
        # Engagement metrics
        daily_questions = df.groupby(df['timestamp'].dt.date).size()
        consistency_score = 1.0 - (daily_questions.std() / daily_questions.mean()) if len(daily_questions) > 1 else 1.0
        
        # Performance patterns
        performance_by_difficulty = df.groupby(pd.cut(df['difficulty'], bins=5))['correct'].mean()
        
        return {
            'time_window_days': time_window_days,
            'total_questions': total_questions,
            'accuracy': accuracy,
            'learning_velocity': learning_velocity,
            'avg_response_time': avg_response_time,
            'response_time_consistency': 1.0 / (1.0 + response_time_std),
            'avg_difficulty': avg_difficulty,
            'difficulty_range': difficulty_std,
            'consistency_score': max(0.0, min(1.0, consistency_score)),
            'topic_performance': topic_stats.to_dict() if not topic_stats.empty else {},
            'performance_by_difficulty': performance_by_difficulty.to_dict(),
            'engagement_level': min(1.0, total_questions / time_window_days / 10.0)  # Normalized to expected 10 questions/day
        }
    
    def generate_class_overview(self, student_profiles: List[StudentProfile]) -> Dict:
        """Generate overview statistics for a class/group of students"""
        if not student_profiles:
            return {"error": "No student data available"}
        
        # Overall statistics
        total_students = len(student_profiles)
        avg_skill_level = np.mean([p.skill_level for p in student_profiles])
        avg_accuracy = np.mean([p.overall_accuracy for p in student_profiles])
        
        # Distribution of learning states
        state_distribution = defaultdict(int)
        for profile in student_profiles:
            state_distribution[profile.current_state.value] += 1
        
        # Topic coverage analysis
        all_topics = set()
        for profile in student_profiles:
            all_topics.update(profile.knowledge_state.keys())
        
        topic_coverage = {}
        for topic in all_topics:
            students_attempted = sum(1 for p in student_profiles if topic in p.knowledge_state)
            avg_mastery = np.mean([
                p.knowledge_state.get(topic, 0.0) 
                for p in student_profiles 
                if topic in p.knowledge_state
            ])
            topic_coverage[topic] = {
                'students_attempted': students_attempted,
                'coverage_percentage': (students_attempted / total_students) * 100,
                'average_mastery': avg_mastery
            }
        
        # Performance distribution
        skill_levels = [p.skill_level for p in student_profiles]
        performance_quartiles = {
            'q1': np.percentile(skill_levels, 25),
            'q2': np.percentile(skill_levels, 50),
            'q3': np.percentile(skill_levels, 75)
        }
        
        # Identify students needing attention
        struggling_students = [
            p.name for p in student_profiles 
            if p.skill_level < performance_quartiles['q1'] or p.overall_accuracy < 0.4
        ]
        
        advanced_students = [
            p.name for p in student_profiles 
            if p.skill_level > performance_quartiles['q3'] and p.overall_accuracy > 0.8
        ]
        
        return {
            'total_students': total_students,
            'average_skill_level': avg_skill_level,
            'average_accuracy': avg_accuracy,
            'state_distribution': dict(state_distribution),
            'topic_coverage': topic_coverage,
            'performance_quartiles': performance_quartiles,
            'struggling_students': struggling_students,
            'advanced_students': advanced_students,
            'total_topics_covered': len(all_topics),
            'class_consistency': 1.0 - (np.std(skill_levels) / np.mean(skill_levels)) if skill_levels else 0
        }
    
    def generate_adaptation_effectiveness_report(self) -> Dict:
        """Analyze how effective the adaptive system is"""
        if not self.performance_data:
            return {"error": "No performance data available"}
        
        df = pd.DataFrame(self.performance_data)
        
        # Overall system performance
        total_questions = len(df)
        overall_accuracy = df['correct'].mean()
        
        # Adaptation patterns
        difficulty_changes = []
        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id].sort_values(by='timestamp')
            if len(student_data) > 1:
                difficulty_diff = student_data['difficulty'].diff().dropna()
                difficulty_changes.extend(difficulty_diff.tolist())
        
        avg_difficulty_change = np.mean(np.abs(difficulty_changes)) if difficulty_changes else 0
        
        # Performance by difficulty level
        difficulty_bins = pd.cut(df['difficulty'], bins=5, labels=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'])
        performance_by_difficulty = df.groupby(difficulty_bins)['correct'].agg(['count', 'mean']).round(3)
        
        # Response time analysis
        avg_response_time = df['response_time'].mean()
        response_time_by_difficulty = df.groupby(difficulty_bins)['response_time'].mean().round(2)
        
        # Learning efficiency (accuracy improvement over time)
        learning_efficiency = {}
        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id].sort_values(by='timestamp')
            if len(student_data) >= 10:
                early_accuracy = student_data.head(5)['correct'].mean()
                recent_accuracy = student_data.tail(5)['correct'].mean()
                learning_efficiency[student_id] = recent_accuracy - early_accuracy
        
        avg_learning_efficiency = np.mean(list(learning_efficiency.values())) if learning_efficiency else 0
        
        return {
            'total_questions_analyzed': total_questions,
            'overall_accuracy': overall_accuracy,
            'average_difficulty_adaptation': avg_difficulty_change,
            'performance_by_difficulty': performance_by_difficulty.to_dict(),
            'average_response_time': avg_response_time,
            'response_time_by_difficulty': response_time_by_difficulty.to_dict(),
            'learning_efficiency': avg_learning_efficiency,
            'students_showing_improvement': len([eff for eff in learning_efficiency.values() if eff > 0]),
            'total_students_analyzed': len(learning_efficiency)
        }
    
    def export_analytics_data(self, filename: str):
        """Export analytics data to CSV for external analysis"""
        if self.performance_data:
            df = pd.DataFrame(self.performance_data)
            df.to_csv(f"{filename}_performance.csv", index=False)
        
        if self.session_data:
            df_sessions = pd.DataFrame(self.session_data)
            df_sessions.to_csv(f"{filename}_sessions.csv", index=False)
