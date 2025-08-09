import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import json

from utils.data_models import Question, DifficultyLevel

class QuestionBank:
    """Manages the question database and provides intelligent question selection"""
    
    def __init__(self):
        self.questions: Dict[str, List[Question]] = defaultdict(list)
        self.topics: List[str] = []
        self.difficulty_distribution: Dict[str, Dict[float, int]] = defaultdict(dict)
        self._initialize_question_bank()
    
    def _initialize_question_bank(self):
        """Initialize the question bank with sample educational content"""
        # Mathematics curriculum topics
        topics_config = {
            "algebra_basics": {
                "name": "Algebra Basics",
                "description": "Fundamental algebraic concepts and operations",
                "prerequisites": []
            },
            "linear_equations": {
                "name": "Linear Equations",
                "description": "Solving and graphing linear equations",
                "prerequisites": ["algebra_basics"]
            },
            "quadratic_equations": {
                "name": "Quadratic Equations",
                "description": "Solving quadratic equations and understanding parabolas",
                "prerequisites": ["algebra_basics", "linear_equations"]
            },
            "geometry": {
                "name": "Geometry",
                "description": "Shapes, areas, volumes, and geometric proofs",
                "prerequisites": []
            },
            "trigonometry": {
                "name": "Trigonometry",
                "description": "Trigonometric functions and their applications",
                "prerequisites": ["geometry", "algebra_basics"]
            },
            "calculus_intro": {
                "name": "Introduction to Calculus",
                "description": "Limits, derivatives, and basic integration",
                "prerequisites": ["algebra_basics", "linear_equations", "quadratic_equations", "trigonometry"]
            }
        }
        
        # Generate questions for each topic
        for topic, config in topics_config.items():
            self._generate_topic_questions(topic, config)
        
        self.topics = list(topics_config.keys())
        print(f"Question bank initialized with {len(self.topics)} topics and {sum(len(questions) for questions in self.questions.values())} total questions")
    
    def _generate_topic_questions(self, topic: str, config: Dict):
        """Generate questions for a specific topic"""
        questions_per_difficulty = 15  # 15 questions per difficulty level for variety
        difficulty_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        question_id = 0
        for difficulty in difficulty_levels:
            # Get appropriate templates for this difficulty level
            appropriate_templates = self._get_templates_for_difficulty(topic, difficulty)
            
            for i in range(questions_per_difficulty):
                template = random.choice(appropriate_templates)
                
                # Generate dynamic values for the question
                dynamic_values = self._generate_dynamic_values(topic, difficulty, i)
                
                question = Question(
                    id=f"{topic}_{question_id:03d}",
                    topic=topic,
                    difficulty=difficulty,
                    content=self._format_question_content(template, difficulty, config, dynamic_values),
                    answer=self._generate_answer(template, dynamic_values),
                    hints=self._format_hints(template["hints"], dynamic_values),
                    explanation=template["explanation"].format(topic_name=config["name"]),
                    metadata={
                        "template_id": template["id"],
                        "prerequisites": config["prerequisites"],
                        "estimated_time": template.get("estimated_time", 120),  # seconds
                        "dynamic_values": dynamic_values
                    }
                )
                
                self.questions[topic].append(question)
                question_id += 1
    
    def _generate_dynamic_values(self, topic: str, difficulty: float, variation: int) -> dict:
        """Generate dynamic values for questions to create variety"""
        values = {}
        
        # Adjust number ranges based on difficulty
        if difficulty < 0.3:
            values['small_num'] = random.randint(1, 10)
            values['med_num'] = random.randint(10, 50)
            values['coeff'] = random.randint(2, 5)
        elif difficulty < 0.7:
            values['small_num'] = random.randint(5, 25)
            values['med_num'] = random.randint(20, 100)
            values['coeff'] = random.randint(3, 12)
        else:
            values['small_num'] = random.randint(10, 50)
            values['med_num'] = random.randint(50, 200)
            values['coeff'] = random.randint(5, 20)
        
        # Generate specific values by topic
        if topic == "algebra_basics":
            values['x_val'] = random.randint(1, 20)
            values['constant'] = random.randint(1, 30)
            values['result'] = values['coeff'] * values['x_val'] + values['constant']
            # Make sure we have the right calculation for different equation types
            values['sum_result'] = values['x_val'] + values['small_num']
            values['product_result'] = values['coeff'] * values['x_val']
            values['simplified_result'] = values['coeff'] + values['small_num']
            
        elif topic == "linear_equations":
            values['slope'] = random.randint(-5, 5)
            values['y_intercept'] = random.randint(-10, 10)
            values['x_point'] = random.randint(-5, 5)
            values['y_point'] = values['slope'] * values['x_point'] + values['y_intercept']
            
        elif topic == "quadratic_equations":
            values['a'] = random.randint(1, 3)
            values['b'] = random.randint(-10, 10)
            values['c'] = random.randint(-10, 10)
            
        elif topic == "geometry":
            values['length'] = random.randint(3, 15)
            values['width'] = random.randint(3, 15)
            values['radius'] = random.randint(2, 12)
            values['area'] = values['length'] * values['width']
            
        elif topic == "trigonometry":
            angles = [30, 45, 60, 90, 120, 135, 150, 180]
            values['angle'] = random.choice(angles)
            values['side_a'] = random.randint(3, 12)
            values['side_b'] = random.randint(3, 12)
            
        elif topic == "calculus_intro":
            values['power'] = random.randint(2, 6)
            values['coefficient'] = random.randint(1, 10)
            
        return values
    
    def _format_question_content(self, template: dict, difficulty: float, config: dict, values: dict) -> str:
        """Format question content with dynamic values"""
        content = template["content"]
        
        # Replace common placeholders
        content = content.replace("{difficulty_adj}", self._get_difficulty_adjective(difficulty))
        content = content.replace("{topic_name}", config["name"])
        
        # Replace dynamic values
        for key, value in values.items():
            content = content.replace(f"{{{key}}}", str(value))
            
        return content
    
    def _generate_answer(self, template: dict, values: dict) -> str:
        """Generate the correct answer based on template and values"""
        answer = template["answer"]
        
        # Replace dynamic values in answer
        for key, value in values.items():
            answer = answer.replace(f"{{{key}}}", str(value))
            
        return answer
    
    def _format_hints(self, hints: list, values: dict) -> list:
        """Format hints with dynamic values"""
        formatted_hints = []
        for hint in hints:
            formatted_hint = hint
            for key, value in values.items():
                formatted_hint = formatted_hint.replace(f"{{{key}}}", str(value))
            formatted_hints.append(formatted_hint)
        return formatted_hints
    
    def _get_templates_for_difficulty(self, topic: str, difficulty: float) -> List[Dict]:
        """Get appropriate question templates for a specific difficulty level"""
        all_templates = self._get_question_templates(topic)
        
        if topic == "algebra_basics":
            if difficulty <= 0.3:
                # Very easy: basic addition/subtraction
                filtered = [t for t in all_templates if t["id"] in ["basic_addition", "basic_subtraction"]]
            elif difficulty <= 0.5:
                # Easy: basic multiplication/division
                filtered = [t for t in all_templates if t["id"] in ["basic_multiplication", "basic_division"]]
            elif difficulty <= 0.7:
                # Medium: two-step equations
                filtered = [t for t in all_templates if t["id"] in ["two_step_equation", "basic_fractions"]]
            elif difficulty <= 0.9:
                # Hard: distribution and combining
                filtered = [t for t in all_templates if t["id"] in ["distribute_and_solve", "combine_like_terms"]]
            else:
                # Very hard: complex multi-step
                filtered = [t for t in all_templates if t["id"] in ["multi_step_complex", "word_problems"]]
        
        elif topic == "linear_equations":
            if difficulty <= 0.4:
                filtered = [t for t in all_templates if t["id"] in ["slope_intercept", "point_slope"]]
            elif difficulty <= 0.7:
                filtered = [t for t in all_templates if t["id"] in ["point_slope"]]
            else:
                filtered = all_templates
        
        elif topic == "quadratic_equations":
            if difficulty <= 0.5:
                filtered = [t for t in all_templates if t["id"] in ["factoring"]]
            elif difficulty <= 0.8:
                filtered = [t for t in all_templates if t["id"] in ["quadratic_formula", "factoring"]]
            else:
                filtered = all_templates
        
        elif topic == "geometry":
            if difficulty <= 0.4:
                filtered = [t for t in all_templates if t["id"] in ["area_calculation"]]
            elif difficulty <= 0.7:
                filtered = [t for t in all_templates if t["id"] in ["pythagorean", "area_calculation"]]
            else:
                filtered = all_templates
        
        elif topic == "trigonometry":
            if difficulty <= 0.5:
                filtered = [t for t in all_templates if t["id"] in ["basic_ratios"]]
            else:
                filtered = all_templates
        
        elif topic == "calculus_intro":
            if difficulty <= 0.6:
                filtered = [t for t in all_templates if t["id"] in ["basic_derivative", "basic_limit"]]
            else:
                filtered = all_templates
        
        else:
            filtered = all_templates
        
        # If no templates match, return all templates as fallback
        return filtered if filtered else all_templates
    
    def _get_question_templates(self, topic: str) -> List[Dict]:
        """Get question templates for a specific topic with difficulty-appropriate content"""
        templates = {
            "algebra_basics": [
                # Very Easy (0.1-0.3)
                {
                    "id": "basic_addition",
                    "content": "Solve this {difficulty_adj} equation: x + {small_num} = {sum_result}",
                    "answer": "{x_val}",
                    "hints": ["Subtract {small_num} from both sides", "x = {sum_result} - {small_num}"],
                    "explanation": "This tests basic addition and subtraction in {topic_name}",
                    "estimated_time": 45
                },
                {
                    "id": "basic_subtraction",
                    "content": "Solve this {difficulty_adj} equation: x - {small_num} = {x_val}",
                    "answer": "{sum_result}",
                    "hints": ["Add {small_num} to both sides", "x = {x_val} + {small_num}"],
                    "explanation": "This tests basic subtraction in {topic_name}",
                    "estimated_time": 45
                },
                # Easy (0.3-0.5)
                {
                    "id": "basic_multiplication",
                    "content": "Solve this {difficulty_adj} equation: {coeff}x = {product_result}",
                    "answer": "{x_val}",
                    "hints": ["Divide both sides by {coeff}", "x = {product_result} ÷ {coeff}"],
                    "explanation": "This tests basic multiplication and division in {topic_name}",
                    "estimated_time": 60
                },
                {
                    "id": "basic_division",
                    "content": "Solve this {difficulty_adj} equation: x/{coeff} = {x_val}",
                    "answer": "{product_result}",
                    "hints": ["Multiply both sides by {coeff}", "x = {x_val} × {coeff}"],
                    "explanation": "This tests basic division in {topic_name}",
                    "estimated_time": 60
                },
                # Medium (0.5-0.7)
                {
                    "id": "two_step_equation",
                    "content": "Solve this {difficulty_adj} equation: {coeff}x + {constant} = {result}",
                    "answer": "{x_val}",
                    "hints": ["First subtract {constant} from both sides", "Then divide by {coeff}"],
                    "explanation": "This tests two-step equations in {topic_name}",
                    "estimated_time": 90
                },
                {
                    "id": "basic_fractions",
                    "content": "Solve this {difficulty_adj} equation: x/3 + {small_num} = {sum_result}",
                    "answer": "{product_result}",
                    "hints": ["First subtract {small_num} from both sides", "Then multiply by 3"],
                    "explanation": "This tests fractions in equations in {topic_name}",
                    "estimated_time": 120
                },
                # Hard (0.7-0.9)
                {
                    "id": "distribute_and_solve",
                    "content": "Solve this {difficulty_adj} equation: {coeff}(x + {small_num}) = {result}",
                    "answer": "{x_val}",
                    "hints": ["First distribute {coeff}", "Then solve the resulting equation"],
                    "explanation": "This tests distribution and solving in {topic_name}",
                    "estimated_time": 150
                },
                {
                    "id": "combine_like_terms",
                    "content": "Simplify this {difficulty_adj} expression: {coeff}x + {small_num}x - {constant}",
                    "answer": "{simplified_result}x - {constant}",
                    "hints": ["Combine the x terms: {coeff}x + {small_num}x", "Keep the constant term separate"],
                    "explanation": "This tests combining like terms in {topic_name}",
                    "estimated_time": 120
                },
                # Very Hard (0.9-1.0)
                {
                    "id": "multi_step_complex",
                    "content": "Solve this {difficulty_adj} equation: {coeff}(x - {small_num}) + {constant} = {med_num}x + {result}",
                    "answer": "{x_val}",
                    "hints": ["First distribute on the left side", "Then move all x terms to one side"],
                    "explanation": "This tests complex multi-step solving in {topic_name}",
                    "estimated_time": 240
                },
                {
                    "id": "word_problems",
                    "content": "A {difficulty_adj} word problem: If a number is increased by {small_num} and then multiplied by {coeff}, the result is {result}. Find the number.",
                    "answer": "{x_val}",
                    "hints": ["Set up the equation: {coeff}(x + {small_num}) = {result}", "Then solve step by step"],
                    "explanation": "This tests word problem translation in {topic_name}",
                    "estimated_time": 300
                }
            ],
            "linear_equations": [
                {
                    "id": "slope_intercept",
                    "content": "Find the slope of this {difficulty_adj} linear equation: y = 3x - 2",
                    "answer": "3",
                    "hints": ["The slope is the coefficient of x", "In y = mx + b form, m is the slope"],
                    "explanation": "This tests understanding of slope-intercept form in {topic_name}",
                    "estimated_time": 60
                },
                {
                    "id": "point_slope",
                    "content": "Write the equation of a {difficulty_adj} line passing through (2, 5) with slope 3",
                    "answer": "y = 3x - 1",
                    "hints": ["Use point-slope form: y - y₁ = m(x - x₁)", "Then convert to slope-intercept form"],
                    "explanation": "This tests point-slope form applications in {topic_name}",
                    "estimated_time": 150
                }
            ],
            "quadratic_equations": [
                {
                    "id": "factoring",
                    "content": "Factor this {difficulty_adj} quadratic: x² - 5x + 6",
                    "answer": "(x - 2)(x - 3)",
                    "hints": ["Look for two numbers that multiply to 6 and add to -5", "Those numbers are -2 and -3"],
                    "explanation": "This tests factoring techniques in {topic_name}",
                    "estimated_time": 180
                },
                {
                    "id": "quadratic_formula",
                    "content": "Solve this {difficulty_adj} quadratic using the quadratic formula: x² + 2x - 3 = 0",
                    "answer": "x = 1 or x = -3",
                    "hints": ["Use x = (-b ± √(b² - 4ac)) / 2a", "Here a=1, b=2, c=-3"],
                    "explanation": "This tests the quadratic formula in {topic_name}",
                    "estimated_time": 200
                }
            ],
            "geometry": [
                {
                    "id": "area_calculation",
                    "content": "Calculate the area of this {difficulty_adj} rectangle with length 8 and width 5",
                    "answer": "40 square units",
                    "hints": ["Area of rectangle = length × width", "Multiply 8 × 5"],
                    "explanation": "This tests basic area calculations in {topic_name}",
                    "estimated_time": 90
                },
                {
                    "id": "pythagorean",
                    "content": "Find the hypotenuse of this {difficulty_adj} right triangle with legs 3 and 4",
                    "answer": "5",
                    "hints": ["Use the Pythagorean theorem: a² + b² = c²", "Calculate √(3² + 4²)"],
                    "explanation": "This tests the Pythagorean theorem in {topic_name}",
                    "estimated_time": 120
                }
            ],
            "trigonometry": [
                {
                    "id": "basic_ratios",
                    "content": "In this {difficulty_adj} problem, what is sin(30°)?",
                    "answer": "1/2 or 0.5",
                    "hints": ["This is a special angle", "Remember the 30-60-90 triangle ratios"],
                    "explanation": "This tests knowledge of special angle ratios in {topic_name}",
                    "estimated_time": 90
                },
                {
                    "id": "unit_circle",
                    "content": "Find cos(π/4) in this {difficulty_adj} unit circle problem",
                    "answer": "√2/2 or approximately 0.707",
                    "hints": ["π/4 radians = 45°", "This is another special angle"],
                    "explanation": "This tests unit circle knowledge in {topic_name}",
                    "estimated_time": 100
                }
            ],
            "calculus_intro": [
                {
                    "id": "basic_derivative",
                    "content": "Find the derivative of this {difficulty_adj} function: f(x) = x³",
                    "answer": "f'(x) = 3x²",
                    "hints": ["Use the power rule: d/dx(xⁿ) = nx^(n-1)", "For x³, multiply by the exponent and reduce the power by 1"],
                    "explanation": "This tests basic derivative rules in {topic_name}",
                    "estimated_time": 120
                },
                {
                    "id": "basic_limit",
                    "content": "Evaluate this {difficulty_adj} limit: lim(x→2) of (x² - 4)/(x - 2)",
                    "answer": "4",
                    "hints": ["Factor the numerator first", "x² - 4 = (x-2)(x+2)"],
                    "explanation": "This tests limit evaluation techniques in {topic_name}",
                    "estimated_time": 180
                }
            ]
        }
        
        return templates.get(topic, [
            {
                "id": "generic",
                "content": "A {difficulty_adj} question about {topic_name}",
                "answer": "Generic answer",
                "hints": ["Think about the key concepts", "Review the fundamentals"],
                "explanation": "This tests understanding of {topic_name}",
                "estimated_time": 120
            }
        ])
    
    def _get_difficulty_adjective(self, difficulty: float) -> str:
        """Get a descriptive adjective for the difficulty level"""
        if difficulty < 0.2:
            return "very easy"
        elif difficulty < 0.4:
            return "easy"
        elif difficulty < 0.6:
            return "moderate"
        elif difficulty < 0.8:
            return "challenging"
        else:
            return "very challenging"
    
    def get_available_topics(self) -> List[str]:
        """Get list of all available topics"""
        return self.topics.copy()
    
    def get_questions_by_topic(self, topic: str) -> List[Question]:
        """Get all questions for a specific topic"""
        return self.questions.get(topic, [])
    
    def get_question(self, topic: str, target_difficulty: float, 
                    tolerance: float = 0.1) -> Optional[Question]:
        """Get a question matching the specified criteria"""
        if topic not in self.questions:
            return None
        
        # Find questions within difficulty tolerance
        candidates = [
            q for q in self.questions[topic]
            if abs(q.difficulty - target_difficulty) <= tolerance
        ]
        
        # If no exact matches, find closest
        if not candidates:
            candidates = sorted(
                self.questions[topic],
                key=lambda q: abs(q.difficulty - target_difficulty)
            )[:3]  # Take top 3 closest matches
        
        return random.choice(candidates) if candidates else None
    
    def get_questions_by_difficulty_range(self, topic: str, min_difficulty: float, 
                                        max_difficulty: float) -> List[Question]:
        """Get questions within a difficulty range"""
        if topic not in self.questions:
            return []
        
        return [
            q for q in self.questions[topic]
            if min_difficulty <= q.difficulty <= max_difficulty
        ]
    
    def get_adaptive_question(self, topic: str, student_performance: float,
                            recent_accuracy: float = 0.5) -> Optional[Question]:
        """Get an adaptively selected question based on student performance"""
        if topic not in self.questions:
            return None
        
        # Adjust target difficulty based on recent performance
        if recent_accuracy > 0.8:
            # Student is doing well, increase difficulty
            target_difficulty = min(1.0, student_performance + 0.2)
        elif recent_accuracy < 0.4:
            # Student is struggling, decrease difficulty
            target_difficulty = max(0.1, student_performance - 0.2)
        else:
            # Maintain current level
            target_difficulty = student_performance
        
        return self.get_question(topic, target_difficulty)
    
    def get_prerequisite_topics(self, topic: str) -> List[str]:
        """Get prerequisite topics for a given topic"""
        if topic not in self.questions or not self.questions[topic]:
            return []
        
        # Get prerequisites from the first question's metadata
        sample_question = self.questions[topic][0]
        return sample_question.metadata.get("prerequisites", [])
    
    def validate_topic_progression(self, completed_topics: List[str], 
                                 target_topic: str) -> bool:
        """Check if student has completed prerequisites for target topic"""
        prerequisites = self.get_prerequisite_topics(target_topic)
        return all(prereq in completed_topics for prereq in prerequisites)
    
    def get_next_recommended_topics(self, completed_topics: List[str]) -> List[str]:
        """Get topics that the student is ready to attempt"""
        recommended = []
        
        for topic in self.topics:
            if topic not in completed_topics:
                if self.validate_topic_progression(completed_topics, topic):
                    recommended.append(topic)
        
        return recommended
    
    def get_question_statistics(self) -> Dict:
        """Get statistical information about the question bank"""
        total_questions = sum(len(questions) for questions in self.questions.values())
        
        topic_stats = {}
        for topic, questions in self.questions.items():
            difficulties = [q.difficulty for q in questions]
            topic_stats[topic] = {
                "count": len(questions),
                "avg_difficulty": np.mean(difficulties) if difficulties else 0,
                "difficulty_range": (min(difficulties), max(difficulties)) if difficulties else (0, 0),
                "difficulty_std": np.std(difficulties) if difficulties else 0
            }
        
        return {
            "total_questions": total_questions,
            "total_topics": len(self.topics),
            "topic_statistics": topic_stats,
            "overall_difficulty_distribution": self._get_difficulty_distribution()
        }
    
    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of questions across difficulty levels"""
        distribution = {
            "very_easy": 0,    # 0.0 - 0.2
            "easy": 0,         # 0.2 - 0.4
            "medium": 0,       # 0.4 - 0.6
            "hard": 0,         # 0.6 - 0.8
            "very_hard": 0     # 0.8 - 1.0
        }
        
        for questions in self.questions.values():
            for question in questions:
                difficulty_level = question.get_difficulty_level()
                if difficulty_level == DifficultyLevel.VERY_EASY:
                    distribution["very_easy"] += 1
                elif difficulty_level == DifficultyLevel.EASY:
                    distribution["easy"] += 1
                elif difficulty_level == DifficultyLevel.MEDIUM:
                    distribution["medium"] += 1
                elif difficulty_level == DifficultyLevel.HARD:
                    distribution["hard"] += 1
                elif difficulty_level == DifficultyLevel.VERY_HARD:
                    distribution["very_hard"] += 1
        
        return distribution
    
    def export_questions(self, filename: str):
        """Export questions to JSON file"""
        export_data = {}
        for topic, questions in self.questions.items():
            export_data[topic] = [
                {
                    "id": q.id,
                    "topic": q.topic,
                    "difficulty": q.difficulty,
                    "content": q.content,
                    "answer": q.answer,
                    "hints": q.hints,
                    "explanation": q.explanation,
                    "metadata": q.metadata
                }
                for q in questions
            ]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_questions(self, filename: str):
        """Import questions from JSON file"""
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        for topic, questions_data in import_data.items():
            self.questions[topic] = []
            for q_data in questions_data:
                question = Question(
                    id=q_data["id"],
                    topic=q_data["topic"],
                    difficulty=q_data["difficulty"],
                    content=q_data["content"],
                    answer=q_data["answer"],
                    hints=q_data.get("hints", []),
                    explanation=q_data.get("explanation", ""),
                    metadata=q_data.get("metadata", {})
                )
                self.questions[topic].append(question)
        
        self.topics = list(self.questions.keys())
