#!/usr/bin/env python3
"""
Realistic Student Response Testing Framework
Generates varied student responses to test grading engine robustness
"""

import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.grading_engine import ResponsibleGradingEngine, GradingResult

@dataclass
class TestScenario:
    """Structure for test scenarios with expected outcomes"""
    scenario_name: str
    student_response: str
    expected_grade_range: Tuple[str, str]  # (min_acceptable, max_acceptable)
    expected_confidence_range: Tuple[float, float]  # (min, max)
    test_purpose: str
    should_flag_for_review: bool = False

class ResponseGenerator:
    """Generate realistic student responses for comprehensive testing"""
    
    def __init__(self):
        self.response_templates = {
            'correct_confident': [
                "The answer is {correct_answer}. {reasoning}",
                "I believe the answer is {correct_answer} because {reasoning}",
                "{correct_answer} is correct. {reasoning}"
            ],
            'correct_uncertain': [
                "I think the answer might be {correct_answer}, but I'm not sure.",
                "The answer is probably {correct_answer}?",
                "I'm guessing {correct_answer} because {weak_reasoning}"
            ],
            'incorrect_confident': [
                "The answer is definitely {wrong_answer}. {incorrect_reasoning}",
                "I'm certain it's {wrong_answer} because {incorrect_reasoning}",
                "{wrong_answer} is the obvious choice. {incorrect_reasoning}"
            ],
            'incorrect_with_partial_understanding': [
                "I think it's {wrong_answer}, but I know that {partial_correct_concept}",
                "The answer is {wrong_answer}. I understand {partial_correct_concept}, but {misconception}",
                "{wrong_answer} because {partial_correct_concept}, although {confusion}"
            ],
            'no_clear_answer': [
                "I don't really understand this question.",
                "This is confusing. I think it has something to do with {vague_concept}?",
                "I'm not sure, maybe {random_guess}?",
                "Can you explain this better?"
            ],
            'off_topic': [
                "This reminds me of when we learned about {unrelated_topic}.",
                "I think this is about {completely_wrong_domain}.",
                "My answer is {random_unrelated_content}."
            ]
        }
        
        # Common misconceptions for different subjects
        self.misconceptions = {
            'geology': [
                "rocks are formed only by volcanoes",
                "all mountains are volcanoes",
                "fossils are only found in igneous rocks",
                "earthquakes only happen near volcanoes"
            ],
            'biology': [
                "evolution means humans came from monkeys",
                "all bacteria are harmful",
                "plants don't need oxygen",
                "bigger animals are always stronger"
            ],
            'physics': [
                "heavier objects fall faster",
                "space has no gravity",
                "electricity and magnetism are unrelated",
                "sound travels faster than light"
            ]
        }
    
    def identify_subject_domain(self, question_text: str, lesson_context: Dict) -> str:
        """Identify the subject domain from question and context"""
        text_to_analyze = (question_text + " " + str(lesson_context)).lower()
        
        if any(word in text_to_analyze for word in ['rock', 'mineral', 'earthquake', 'volcano', 'plate', 'sediment']):
            return 'geology'
        elif any(word in text_to_analyze for word in ['cell', 'organism', 'evolution', 'dna', 'biology', 'species']):
            return 'biology'
        elif any(word in text_to_analyze for word in ['force', 'energy', 'motion', 'physics', 'gravity', 'wave']):
            return 'physics'
        else:
            return 'general'
    
    def extract_key_concepts(self, question_data: Dict) -> List[str]:
        """Extract key concepts from question and lesson context"""
        concepts = []
        
        # Extract from question text
        question_words = question_data['question_text'].lower().split()
        
        # Extract from lesson context
        for topic_content in question_data['lesson_context'].values():
            if topic_content:
                # Simple concept extraction - look for capitalized terms
                words = topic_content.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 3:
                        concepts.append(word.lower())
        
        # Extract from vocabulary
        concepts.extend([term.lower() for term in question_data['vocabulary'].keys()])
        
        return list(set(concepts))[:5]  # Return top 5 unique concepts
    
    def generate_reasoning(self, question_data: Dict, is_correct: bool = True) -> str:
        """Generate reasoning based on lesson content"""
        concepts = self.extract_key_concepts(question_data)
        subject = self.identify_subject_domain(question_data['question_text'], question_data['lesson_context'])
        
        if is_correct and concepts:
            return f"based on what we learned about {random.choice(concepts)}"
        elif not is_correct and subject in self.misconceptions:
            return f"I think {random.choice(self.misconceptions[subject])}"
        else:
            return "this seems right to me"
    
    def generate_test_scenarios(self, question_data: Dict) -> List[TestScenario]:
        """Generate comprehensive test scenarios for a question"""
        scenarios = []
        correct_answer = question_data['correct_answer']
        answer_choices = list(question_data['answer_choices'].keys())
        wrong_answers = [choice for choice in answer_choices if choice != correct_answer]
        
        # Scenario 1: Perfect correct response
        correct_reasoning = self.generate_reasoning(question_data, is_correct=True)
        scenarios.append(TestScenario(
            scenario_name="Perfect Correct Response",
            student_response=f"The answer is {correct_answer}. This is correct {correct_reasoning}.",
            expected_grade_range=(correct_answer, correct_answer),
            expected_confidence_range=(0.8, 1.0),
            test_purpose="Baseline accuracy test - should get full credit with high confidence"
        ))
        
        # Scenario 2: Correct but uncertain
        scenarios.append(TestScenario(
            scenario_name="Correct but Uncertain",
            student_response=f"I think the answer might be {correct_answer}, but I'm not completely sure.",
            expected_grade_range=(correct_answer, correct_answer),
            expected_confidence_range=(0.5, 0.8),
            test_purpose="Test confidence calibration - correct answer but lower confidence"
        ))
        
        # Scenario 3: Incorrect but confident
        if wrong_answers:
            wrong_reasoning = self.generate_reasoning(question_data, is_correct=False)
            scenarios.append(TestScenario(
                scenario_name="Incorrect but Confident",
                student_response=f"The answer is definitely {wrong_answers[0]}. I'm sure because {wrong_reasoning}.",
                expected_grade_range=(wrong_answers[0], wrong_answers[0]),
                expected_confidence_range=(0.6, 0.9),
                test_purpose="Test handling of confident incorrect responses"
            ))
        
        # Scenario 4: Partial understanding
        concepts = self.extract_key_concepts(question_data)
        if wrong_answers and concepts:
            scenarios.append(TestScenario(
                scenario_name="Partial Understanding",
                student_response=f"I think it's {wrong_answers[0]}. I know {random.choice(concepts)} is important, but I'm confused about how it applies here.",
                expected_grade_range=(wrong_answers[0], wrong_answers[0]),
                expected_confidence_range=(0.3, 0.7),
                test_purpose="Test partial credit for showing some understanding",
                should_flag_for_review=True
            ))
        
        # Scenario 5: No clear answer
        scenarios.append(TestScenario(
            scenario_name="Unclear Response",
            student_response="I don't really understand this question. Can you explain it better?",
            expected_grade_range=("unclear", "unclear"),
            expected_confidence_range=(0.0, 0.3),
            test_purpose="Test handling of non-responses",
            should_flag_for_review=True
        ))
        
        # Scenario 6: Off-topic response
        scenarios.append(TestScenario(
            scenario_name="Off-topic Response",
            student_response="This reminds me of my vacation to the beach last summer.",
            expected_grade_range=("off-topic", "off-topic"),
            expected_confidence_range=(0.0, 0.2),
            test_purpose="Test handling of completely irrelevant responses",
            should_flag_for_review=True
        ))
        
        return scenarios

class ComprehensiveTester:
    """Comprehensive testing framework for the grading engine"""
    
    def __init__(self, grading_engine: ResponsibleGradingEngine):
        self.grading_engine = grading_engine
        self.response_generator = ResponseGenerator()
        self.test_results = []
    
    def run_comprehensive_test(self, questions: List[Dict], num_questions_to_test: int = 10) -> Dict:
        """Run comprehensive tests on multiple questions with various response types"""
        
        # Select questions for testing
        test_questions = questions[:num_questions_to_test]
        
        print(f"Running comprehensive tests on {len(test_questions)} questions...")
        print("=" * 60)
        
        all_scenarios = []
        scenario_results = []
        
        for i, question in enumerate(test_questions):
            print(f"\nQuestion {i+1}: {question['question_text'][:50]}...")
            
            # Generate test scenarios for this question
            scenarios = self.response_generator.generate_test_scenarios(question)
            
            for scenario in scenarios:
                print(f"  Testing: {scenario.scenario_name}")
                
                # Grade the response
                result = self.grading_engine.grade_response(question, scenario.student_response)
                
                # Evaluate the result
                evaluation = self.evaluate_scenario_result(scenario, result)
                
                scenario_result = {
                    'question_id': question['question_id'],
                    'scenario': scenario,
                    'grading_result': result,
                    'evaluation': evaluation
                }
                
                scenario_results.append(scenario_result)
                
                print(f"    AI Grade: {result.ai_score}, Confidence: {result.confidence_score:.2f}")
                print(f"    Evaluation: {evaluation['status']}")
        
        # Analyze overall results
        analysis = self.analyze_comprehensive_results(scenario_results)
        
        return {
            'scenario_results': scenario_results,
            'analysis': analysis
        }
    
    def evaluate_scenario_result(self, scenario: TestScenario, result: GradingResult) -> Dict:
        """Evaluate how well the grading result matches scenario expectations"""
        
        evaluation = {
            'status': 'UNKNOWN',
            'grade_match': False,
            'confidence_appropriate': False,
            'flagging_appropriate': False,
            'notes': []
        }
        
        # Check grade accuracy
        expected_min, expected_max = scenario.expected_grade_range
        if expected_min == expected_max:
            evaluation['grade_match'] = result.ai_score == expected_min
        else:
            # For ranges, check if result falls within expected range
            evaluation['grade_match'] = True  # More complex logic needed for ranges
        
        # Check confidence appropriateness
        conf_min, conf_max = scenario.expected_confidence_range
        evaluation['confidence_appropriate'] = conf_min <= result.confidence_score <= conf_max
        
        # Check flagging appropriateness
        low_confidence_flag = result.confidence_score < 0.5
        evaluation['flagging_appropriate'] = scenario.should_flag_for_review == low_confidence_flag
        
        # Determine overall status
        if evaluation['grade_match'] and evaluation['confidence_appropriate']:
            evaluation['status'] = 'PASS'
        elif evaluation['grade_match']:
            evaluation['status'] = 'PARTIAL'
            evaluation['notes'].append('Grade correct but confidence calibration off')
        else:
            evaluation['status'] = 'FAIL'
            evaluation['notes'].append('Incorrect grade assignment')
        
        return evaluation
    
    def analyze_comprehensive_results(self, scenario_results: List[Dict]) -> Dict:
        """Analyze patterns across all test scenarios"""
        
        total_scenarios = len(scenario_results)
        if total_scenarios == 0:
            return {'error': 'No scenarios to analyze'}
        
        # Count outcomes
        passes = sum(1 for r in scenario_results if r['evaluation']['status'] == 'PASS')
        partials = sum(1 for r in scenario_results if r['evaluation']['status'] == 'PARTIAL')
        fails = sum(1 for r in scenario_results if r['evaluation']['status'] == 'FAIL')
        
        # Analyze by scenario type
        scenario_performance = {}
        for result in scenario_results:
            scenario_name = result['scenario'].scenario_name
            if scenario_name not in scenario_performance:
                scenario_performance[scenario_name] = {'total': 0, 'passes': 0}
            
            scenario_performance[scenario_name]['total'] += 1
            if result['evaluation']['status'] == 'PASS':
                scenario_performance[scenario_name]['passes'] += 1
        
        # Calculate success rates by scenario type
        for scenario_name in scenario_performance:
            total = scenario_performance[scenario_name]['total']
            passes = scenario_performance[scenario_name]['passes']
            scenario_performance[scenario_name]['success_rate'] = passes / total if total > 0 else 0
        
        return {
            'overall_metrics': {
                'total_scenarios': total_scenarios,
                'pass_rate': passes / total_scenarios,
                'partial_rate': partials / total_scenarios,
                'fail_rate': fails / total_scenarios
            },
            'scenario_performance': scenario_performance,
            'recommendations': self.generate_recommendations(scenario_performance)
        }
    
    def generate_recommendations(self, scenario_performance: Dict) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for scenario_name, performance in scenario_performance.items():
            success_rate = performance['success_rate']
            
            if success_rate < 0.5:
                recommendations.append(f"LOW PERFORMANCE: {scenario_name} - Consider prompt engineering improvements")
            elif success_rate < 0.8:
                recommendations.append(f"MODERATE PERFORMANCE: {scenario_name} - Review confidence calibration")
        
        if not recommendations:
            recommendations.append("GOOD PERFORMANCE: All scenario types performing adequately")
        
        return recommendations
    
    def save_comprehensive_results(self, results: Dict, output_path: str):
        """Save comprehensive test results"""
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = {
            'analysis': results['analysis'],
            'scenario_details': []
        }
        
        for result in results['scenario_results']:
            serializable_result = {
                'question_id': result['question_id'],
                'scenario_name': result['scenario'].scenario_name,
                'scenario_purpose': result['scenario'].test_purpose,
                'student_response': result['scenario'].student_response,
                'expected_grade': result['scenario'].expected_grade_range,
                'expected_confidence': result['scenario'].expected_confidence_range,
                'ai_grade': result['grading_result'].ai_score,
                'ai_confidence': result['grading_result'].confidence_score,
                'ai_explanation': result['grading_result'].explanation,
                'evaluation_status': result['evaluation']['status'],
                'evaluation_notes': result['evaluation']['notes']
            }
            serializable_results['scenario_details'].append(serializable_result)
        
        # Save results
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive test results saved to {output_file}")

def main():
    """Run comprehensive testing"""
    
    # Configuration
    DATASET_PATH = "data/tqa-samples/development_set.json"
    OUTPUT_PATH = "evaluation/comprehensive_test_results.json"
    NUM_QUESTIONS_TO_TEST = 5
    
    try:
        # Initialize systems
        grading_engine = ResponsibleGradingEngine()
        tester = ComprehensiveTester(grading_engine)
        
        # Load questions
        questions = grading_engine.load_development_dataset(DATASET_PATH)
        
        # Run comprehensive tests
        results = tester.run_comprehensive_test(questions, NUM_QUESTIONS_TO_TEST)
        
        # Display summary
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        analysis = results['analysis']
        overall = analysis['overall_metrics']
        
        print(f"Total Scenarios Tested: {overall['total_scenarios']}")
        print(f"Pass Rate: {overall['pass_rate']:.1%}")
        print(f"Partial Rate: {overall['partial_rate']:.1%}")
        print(f"Fail Rate: {overall['fail_rate']:.1%}")
        
        print(f"\nScenario Performance:")
        for scenario, perf in analysis['scenario_performance'].items():
            print(f"  {scenario}: {perf['success_rate']:.1%} ({perf['passes']}/{perf['total']})")
        
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        # Save results
        tester.save_comprehensive_results(results, OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error during comprehensive testing: {e}")

if __name__ == "__main__":
    main()