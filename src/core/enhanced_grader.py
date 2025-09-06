#!/usr/bin/env python3
"""
Improved Grading Engine with Robust Edge Case Handling
Addresses response parsing, prompt engineering, and confidence calibration issues
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import openai
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class GradingResult:
    """Structure for grading results with enhanced transparency"""
    question_id: str
    student_response: str
    ai_score: str
    correct_answer: str
    confidence_score: float
    explanation: str
    context_used: List[str]
    needs_human_review: bool
    parsing_success: bool
    timestamp: str

class ImprovedGradingEngine:
    """
    Enhanced grading engine with robust edge case handling
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the improved grading engine"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        print(f"API Key loaded: {self.api_key[:10]}..." if self.api_key else "No API key found")
        
        self.client = OpenAI(api_key=self.api_key)
        self.grading_results = []
        self.transparency_log = []
        
        # Define confidence thresholds for human review flagging
        self.review_thresholds = {
            'low_confidence': 0.3,
            'uncertain_response': 0.5,
            'off_topic_detection': 0.2
        }
    
    def load_development_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the extracted TQA development dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_relevant_context(self, question_data: Dict) -> str:
        """Extract the most relevant lesson context for answering the question"""
        question_text = question_data['question_text'].lower()
        lesson_context = question_data['lesson_context']
        vocabulary = question_data['vocabulary']
        
        # Enhanced keyword matching with better relevance scoring
        relevant_sections = []
        
        # Check each topic section for relevance
        for topic_id, content in lesson_context.items():
            if content and self._calculate_relevance_score(question_text, content) > 0.3:
                relevant_sections.append(f"Topic {topic_id}: {content}")
        
        # Add relevant vocabulary with better filtering
        relevant_vocab = []
        for term, definition in vocabulary.items():
            if (term.lower() in question_text or 
                any(word in question_text for word in term.lower().split())):
                relevant_vocab.append(f"{term}: {definition}")
        
        # Combine relevant sections with length limits
        context_parts = []
        if relevant_sections:
            context_parts.extend(relevant_sections[:2])  # Top 2 sections to avoid prompt overflow
        if relevant_vocab:
            context_parts.append("Key terms: " + "; ".join(relevant_vocab[:3]))
        
        return "\n\n".join(context_parts)
    
    def _calculate_relevance_score(self, question_text: str, context: str) -> float:
        """Enhanced relevance scoring with better keyword matching"""
        question_words = set(question_text.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        question_words -= stop_words
        context_words -= stop_words
        
        # Calculate overlap with length normalization
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(context_words))
        return overlap / len(question_words)
    
    def build_robust_prompt(self, question_data: Dict, student_response: str) -> str:
        """Build a robust prompt that handles all edge cases properly"""
        
        question_text = question_data['question_text']
        answer_choices = question_data['answer_choices']
        correct_answer = question_data['correct_answer']
        context_content = self.extract_relevant_context(question_data)
        
        # Build comprehensive prompt with explicit edge case handling
        prompt = f"""You are an expert educational assessment system. Your task is to grade a student response to a multiple-choice question.

QUESTION:
{question_text}

ANSWER CHOICES:
"""
        
        for choice_id, choice_text in answer_choices.items():
            prompt += f"{choice_id}. {choice_text}\n"
        
        prompt += f"""
CORRECT ANSWER: {correct_answer}

LESSON CONTEXT:
{context_content}

STUDENT RESPONSE:
"{student_response}"

GRADING INSTRUCTIONS:
1. Determine if the student response contains a clear answer choice (a, b, c, d, etc.)
2. Evaluate the quality of reasoning if provided
3. Assign confidence based on clarity and correctness of the response

SPECIAL CASES:
- If response is unclear or doesn't contain an answer choice, return "UNCLEAR"
- If response is completely off-topic, return "OFF_TOPIC"
- If response shows partial understanding but wrong answer, note this in explanation
- If response is empty or just says "I don't know", return "NO_RESPONSE"

REQUIRED OUTPUT FORMAT:
ANSWER: [single letter a/b/c/d OR special case code]
CONFIDENCE: [number between 0.0 and 1.0]
EXPLANATION: [2-3 sentences explaining your assessment]
REASONING_QUALITY: [GOOD/PARTIAL/POOR/NONE]

Example responses:
- For correct answer with good reasoning: ANSWER: b, CONFIDENCE: 0.9
- For unclear response: ANSWER: UNCLEAR, CONFIDENCE: 0.1
- For off-topic response: ANSWER: OFF_TOPIC, CONFIDENCE: 0.0

Provide your assessment now:"""
        
        return prompt
    
    def grade_response(self, question_data: Dict, student_response: str) -> GradingResult:
        """Grade a student response with robust error handling"""
        
        # Build robust prompt
        prompt = self.build_robust_prompt(question_data, student_response)
        
        # Log the prompt for transparency
        self.transparency_log.append({
            'question_id': question_data['question_id'],
            'prompt_used': prompt,
            'student_response': student_response,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # Call OpenAI API with error handling
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise educational assessment expert. Follow the output format exactly and handle all edge cases appropriately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            
            # Enhanced response parsing with better error handling
            parsed_result = self._parse_ai_response_robust(ai_response)
            
            # Determine if human review is needed
            needs_review = self._determine_review_need(parsed_result, question_data)
            
            # Create enhanced grading result
            result = GradingResult(
                question_id=question_data['question_id'],
                student_response=student_response,
                ai_score=parsed_result['answer'],
                correct_answer=question_data['correct_answer'],
                confidence_score=parsed_result['confidence'],
                explanation=parsed_result['explanation'],
                context_used=parsed_result['context_used'],
                needs_human_review=needs_review,
                parsing_success=parsed_result['parsing_success'],
                timestamp=datetime.now().isoformat()
            )
            
            self.grading_results.append(result)
            return result
            
        except Exception as e:
            # Enhanced error handling with detailed logging
            error_result = GradingResult(
                question_id=question_data['question_id'],
                student_response=student_response,
                ai_score="SYSTEM_ERROR",
                correct_answer=question_data['correct_answer'],
                confidence_score=0.0,
                explanation=f"System error during grading: {str(e)}",
                context_used=[],
                needs_human_review=True,
                parsing_success=False,
                timestamp=datetime.now().isoformat()
            )
            
            self.grading_results.append(error_result)
            return error_result
    
    def _parse_ai_response_robust(self, ai_response: str) -> Dict:
        """Enhanced response parsing with comprehensive error handling"""
        
        # Initialize default values
        parsed = {
            'answer': 'PARSING_ERROR',
            'confidence': 0.0,
            'explanation': 'Failed to parse AI response',
            'reasoning_quality': 'UNKNOWN',
            'context_used': [],
            'parsing_success': False
        }
        
        try:
            lines = ai_response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Parse answer with multiple formats
                if line.startswith('ANSWER:'):
                    answer_text = line.replace('ANSWER:', '').strip()
                    # Handle various answer formats
                    if answer_text in ['a', 'b', 'c', 'd']:
                        parsed['answer'] = answer_text
                    elif answer_text.upper() in ['UNCLEAR', 'OFF_TOPIC', 'NO_RESPONSE']:
                        parsed['answer'] = answer_text.upper()
                    else:
                        # Try to extract letter from more complex responses
                        letter_match = re.search(r'\b([a-d])\b', answer_text.lower())
                        if letter_match:
                            parsed['answer'] = letter_match.group(1)
                        else:
                            parsed['answer'] = 'UNCLEAR'
                
                # Parse confidence with validation
                elif line.startswith('CONFIDENCE:'):
                    conf_text = line.replace('CONFIDENCE:', '').strip()
                    try:
                        confidence = float(conf_text)
                        # Validate confidence range
                        parsed['confidence'] = max(0.0, min(1.0, confidence))
                    except ValueError:
                        parsed['confidence'] = 0.0
                
                # Parse explanation
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
                    if explanation:
                        parsed['explanation'] = explanation
                    else:
                        parsed['explanation'] = 'No explanation provided'
                
                # Parse reasoning quality
                elif line.startswith('REASONING_QUALITY:'):
                    quality = line.replace('REASONING_QUALITY:', '').strip().upper()
                    if quality in ['GOOD', 'PARTIAL', 'POOR', 'NONE']:
                        parsed['reasoning_quality'] = quality
            
            # Mark as successful if we got at least answer and confidence
            if parsed['answer'] != 'PARSING_ERROR' and parsed['confidence'] >= 0.0:
                parsed['parsing_success'] = True
            
            return parsed
            
        except Exception as e:
            # If parsing completely fails, return error state
            parsed['explanation'] = f"Parsing error: {str(e)}"
            return parsed
    
    def _determine_review_need(self, parsed_result: Dict, question_data: Dict) -> bool:
        """Determine if human review is needed based on multiple factors"""
        
        # Always flag if parsing failed
        if not parsed_result['parsing_success']:
            return True
        
        # Flag special cases that need human attention
        special_cases = ['UNCLEAR', 'OFF_TOPIC', 'NO_RESPONSE', 'SYSTEM_ERROR', 'PARSING_ERROR']
        if parsed_result['answer'] in special_cases:
            return True
        
        # Flag low confidence responses
        if parsed_result['confidence'] < self.review_thresholds['low_confidence']:
            return True
        
        # Flag responses with poor reasoning quality
        if parsed_result.get('reasoning_quality') == 'POOR':
            return True
        
        # Flag incorrect answers with high confidence (potential misconceptions)
        if (parsed_result['answer'] != question_data['correct_answer'] and 
            parsed_result['confidence'] > 0.7):
            return True
        
        return False
    
    def evaluate_grading_accuracy(self) -> Dict[str, float]:
        """Enhanced evaluation with better edge case handling"""
        
        if not self.grading_results:
            return {"error": "No grading results to evaluate"}
        
        # Separate different types of results
        standard_results = []
        edge_case_results = []
        error_results = []
        
        for result in self.grading_results:
            if result.ai_score in ['UNCLEAR', 'OFF_TOPIC', 'NO_RESPONSE']:
                edge_case_results.append(result)
            elif result.ai_score in ['SYSTEM_ERROR', 'PARSING_ERROR']:
                error_results.append(result)
            else:
                standard_results.append(result)
        
        # Calculate accuracy for standard responses
        standard_accuracy = 0.0
        if standard_results:
            correct_standard = sum(1 for r in standard_results 
                                 if r.ai_score == r.correct_answer)
            standard_accuracy = correct_standard / len(standard_results)
        
        # Calculate confidence statistics
        all_confidences = [r.confidence_score for r in self.grading_results]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        # Calculate review flagging rate
        review_flagged = sum(1 for r in self.grading_results if r.needs_human_review)
        review_rate = review_flagged / len(self.grading_results)
        
        return {
            "total_responses": len(self.grading_results),
            "standard_responses": len(standard_results),
            "standard_accuracy": standard_accuracy,
            "edge_cases_detected": len(edge_case_results),
            "system_errors": len(error_results),
            "average_confidence": avg_confidence,
            "human_review_rate": review_rate,
            "parsing_success_rate": sum(1 for r in self.grading_results if r.parsing_success) / len(self.grading_results)
        }
    
    def save_enhanced_results(self, output_path: str):
        """Save enhanced results with detailed analysis"""
        
        results_data = {
            "grading_results": [
                {
                    "question_id": r.question_id,
                    "student_response": r.student_response,
                    "ai_score": r.ai_score,
                    "correct_answer": r.correct_answer,
                    "confidence_score": r.confidence_score,
                    "explanation": r.explanation,
                    "context_used": r.context_used,
                    "needs_human_review": r.needs_human_review,
                    "parsing_success": r.parsing_success,
                    "timestamp": r.timestamp
                }
                for r in self.grading_results
            ],
            "transparency_log": self.transparency_log,
            "evaluation_metrics": self.evaluate_grading_accuracy(),
            "system_configuration": {
                "review_thresholds": self.review_thresholds,
                "model_used": "gpt-4",
                "temperature": 0.1
            }
        }
        
        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced results saved to {output_file}")

def main():
    """Test the improved grading engine"""
    
    # Configuration
    DATASET_PATH = "data/tqa-samples/development_set.json"
    OUTPUT_PATH = "evaluation/improved_grading_results.json"
    
    try:
        # Initialize improved grading engine
        engine = ImprovedGradingEngine()
        
        # Load development dataset
        questions = engine.load_development_dataset(DATASET_PATH)
        
        print(f"Testing improved grading engine on {len(questions)} questions")
        print("=" * 60)
        
        # Test with first 5 questions using various response types
        test_scenarios = [
            ("Perfect response", lambda q: f"The answer is {q['correct_answer']}."),
            ("Uncertain response", lambda q: f"I think the answer might be {q['correct_answer']}, but I'm not sure."),
            ("Wrong confident", lambda q: f"The answer is definitely {list(q['answer_choices'].keys())[0] if list(q['answer_choices'].keys())[0] != q['correct_answer'] else list(q['answer_choices'].keys())[1]}."),
            ("Unclear response", lambda q: "I don't really understand this question."),
            ("Off-topic response", lambda q: "This reminds me of my vacation.")
        ]
        
        for i, question in enumerate(questions[:5]):
            print(f"\nQuestion {i+1}: {question['question_text'][:50]}...")
            
            for scenario_name, response_generator in test_scenarios:
                student_response = response_generator(question)
                result = engine.grade_response(question, student_response)
                
                print(f"  {scenario_name}:")
                print(f"    Response: {student_response[:30]}...")
                print(f"    Grade: {result.ai_score}, Confidence: {result.confidence_score:.2f}")
                print(f"    Review needed: {result.needs_human_review}")
                print(f"    Parsing success: {result.parsing_success}")
        
        # Display summary metrics
        metrics = engine.evaluate_grading_accuracy()
        print(f"\n" + "=" * 60)
        print("IMPROVED GRADING ENGINE SUMMARY")
        print("=" * 60)
        print(f"Total responses processed: {metrics['total_responses']}")
        print(f"Standard response accuracy: {metrics['standard_accuracy']:.1%}")
        print(f"Edge cases properly detected: {metrics['edge_cases_detected']}")
        print(f"System errors: {metrics['system_errors']}")
        print(f"Average confidence: {metrics['average_confidence']:.2f}")
        print(f"Human review flagging rate: {metrics['human_review_rate']:.1%}")
        print(f"Parsing success rate: {metrics['parsing_success_rate']:.1%}")
        
        # Save results
        engine.save_enhanced_results(OUTPUT_PATH)
        
    except Exception as e:
        print(f"Error during improved testing: {e}")

if __name__ == "__main__":
    main()