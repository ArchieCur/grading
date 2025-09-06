#!/usr/bin/env python3
"""
Core Grading Engine - ETH Methodology Implementation
Transparent, bias-aware automated grading system
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI  # Change this import
from datetime import datetime
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()  # Add this line

@dataclass
class GradingResult:
    """Structure for grading results with transparency"""
    question_id: str
    student_response: str
    ai_score: str
    correct_answer: str
    confidence_score: float
    explanation: str
    context_used: List[str]
    timestamp: str

@dataclass
class PromptComponents:
    """ETH-style structured prompt components"""
    question_text: str
    sample_solution: str
    evaluation_instructions: str
    context_content: str
    answer_choices: Dict[str, str]

class ResponsibleGradingEngine:
    """
    Core grading engine implementing ETH methodology with transparency and bias detection
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the grading engine"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required")
    
        print(f"API Key loaded: {self.api_key[:10]}..." if self.api_key else "No API key found")
    
        self.client = OpenAI(api_key=self.api_key)
        self.grading_results = []
        self.transparency_log = []
    
    def load_development_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the extracted TQA development dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_relevant_context(self, question_data: Dict) -> str:
        """
        Extract the most relevant lesson context for answering the question
        This implements information retrieval to focus on pertinent content
        """
        question_text = question_data['question_text'].lower()
        lesson_context = question_data['lesson_context']
        vocabulary = question_data['vocabulary']
        
        # Simple keyword matching to find relevant context sections
        relevant_sections = []
        
        # Check each topic section for relevance
        for topic_id, content in lesson_context.items():
            if content and self._is_context_relevant(question_text, content):
                relevant_sections.append(f"Topic {topic_id}: {content}")
        
        # Add relevant vocabulary
        relevant_vocab = []
        for term, definition in vocabulary.items():
            if term.lower() in question_text:
                relevant_vocab.append(f"{term}: {definition}")
        
        # Combine relevant sections
        context_parts = []
        if relevant_sections:
            context_parts.extend(relevant_sections[:3])  # Limit to top 3 sections
        if relevant_vocab:
            context_parts.append("Vocabulary: " + "; ".join(relevant_vocab))
        
        return "\n\n".join(context_parts)
    
    def _is_context_relevant(self, question_text: str, context: str) -> bool:
        """Simple relevance scoring based on keyword overlap"""
        question_words = set(question_text.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        question_words -= stop_words
        context_words -= stop_words
        
        # Calculate overlap ratio
        overlap = len(question_words.intersection(context_words))
        return overlap >= 2  # Require at least 2 meaningful word matches
    
    def build_structured_prompt(self, question_data: Dict) -> PromptComponents:
        """
        Build ETH-style structured prompt components
        Format: Question + Sample Solution + Evaluation Instructions + Points
        """
        
        # Extract components
        question_text = question_data['question_text']
        answer_choices = question_data['answer_choices']
        correct_answer = question_data['correct_answer']
        context_content = self.extract_relevant_context(question_data)
        
        # Build sample solution
        correct_choice_text = answer_choices.get(correct_answer, correct_answer)
        sample_solution = f"The correct answer is '{correct_answer}: {correct_choice_text}'"
        
        # Add reasoning from context if available
        if context_content:
            sample_solution += f"\n\nReasoning: Based on the lesson content, {correct_choice_text.lower()} because the material explains relevant concepts."
        
        # Build evaluation instructions
        evaluation_instructions = f"""
Evaluate the student's answer using these criteria:
1. Correctness: Does the answer match the correct choice '{correct_answer}'?
2. Understanding: Does any explanation show comprehension of underlying concepts?
3. Context alignment: Is the reasoning consistent with the lesson material?

Scoring:
- Correct answer with sound reasoning: Full credit
- Correct answer with minimal explanation: Partial credit  
- Incorrect answer with good reasoning: Minimal credit
- Incorrect answer with poor reasoning: No credit

Provide the answer choice (a, b, c, etc.) and confidence level (0.0-1.0).
        """
        
        return PromptComponents(
            question_text=question_text,
            sample_solution=sample_solution,
            evaluation_instructions=evaluation_instructions,
            context_content=context_content,
            answer_choices=answer_choices
        )
    
    def format_grading_prompt(self, components: PromptComponents, student_response: str) -> str:
        """Format the complete grading prompt using ETH methodology"""
        
        prompt = f"""You are an educational assessment expert. Grade the following student response using the provided materials.

QUESTION:
{components.question_text}

ANSWER CHOICES:
"""
        
        for choice_id, choice_text in components.answer_choices.items():
            prompt += f"{choice_id}. {choice_text}\n"
        
        prompt += f"""
LESSON CONTEXT:
{components.context_content}

SAMPLE SOLUTION:
{components.sample_solution}

EVALUATION INSTRUCTIONS:
{components.evaluation_instructions}

STUDENT RESPONSE:
{student_response}

GRADING TASK:
Please grade this response and provide:
1. Score (correct answer choice: a, b, c, etc.)
2. Confidence (0.0-1.0)
3. Brief explanation of your reasoning
4. Which parts of the lesson context influenced your decision

Format your response as:
ANSWER: [choice letter]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [your reasoning]
CONTEXT_USED: [relevant context sections]
"""
        
        return prompt
    
    def grade_response(self, question_data: Dict, student_response: str) -> GradingResult:
        """
        Grade a single student response using structured prompting
        """
        
        # Build structured prompt
        components = self.build_structured_prompt(question_data)
        prompt = self.format_grading_prompt(components, student_response)
        
        # Log the prompt for transparency
        self.transparency_log.append({
            'question_id': question_data['question_id'],
            'prompt_used': prompt,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # Replace the old openai.ChatCompletion.create call with:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise educational assessment expert. Follow the grading instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                 temperature=0.1,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response
            ai_score, confidence, explanation, context_used = self._parse_ai_response(ai_response)
            
            # Create grading result
            result = GradingResult(
                question_id=question_data['question_id'],
                student_response=student_response,
                ai_score=ai_score,
                correct_answer=question_data['correct_answer'],
                confidence_score=confidence,
                explanation=explanation,
                context_used=context_used,
                timestamp=datetime.now().isoformat()
            )
            
            self.grading_results.append(result)
            return result
            
        except Exception as e:
            # Handle API errors gracefully
            return GradingResult(
                question_id=question_data['question_id'],
                student_response=student_response,
                ai_score="ERROR",
                correct_answer=question_data['correct_answer'],
                confidence_score=0.0,
                explanation=f"Grading error: {str(e)}",
                context_used=[],
                timestamp=datetime.now().isoformat()
            )
    
    def _parse_ai_response(self, ai_response: str) -> Tuple[str, float, str, List[str]]:
        """Parse the AI's grading response into components"""
        
        lines = ai_response.strip().split('\n')
        
        ai_score = "UNKNOWN"
        confidence = 0.0
        explanation = "No explanation provided"
        context_used = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('ANSWER:'):
                ai_score = line.replace('ANSWER:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    confidence = 0.0
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('CONTEXT_USED:'):
                context_text = line.replace('CONTEXT_USED:', '').strip()
                context_used = [context_text] if context_text else []
        
        return ai_score, confidence, explanation, context_used
    
    def evaluate_grading_accuracy(self) -> Dict[str, float]:
        """
        Evaluate the accuracy of AI grading against known correct answers
        """
        if not self.grading_results:
            return {"error": "No grading results to evaluate"}
        
        correct_count = 0
        total_count = len(self.grading_results)
        confidence_scores = []
        
        for result in self.grading_results:
            if result.ai_score == result.correct_answer:
                correct_count += 1
            confidence_scores.append(result.confidence_score)
        
        accuracy = correct_count / total_count
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "accuracy": accuracy,
            "total_questions": total_count,
            "correct_predictions": correct_count,
            "average_confidence": avg_confidence
        }
    
    def save_results(self, output_path: str):
        """Save grading results and transparency logs"""
        
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
                    "timestamp": r.timestamp
                }
                for r in self.grading_results
            ],
            "transparency_log": self.transparency_log,
            "evaluation_metrics": self.evaluate_grading_accuracy()
        }
        
        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")

def main():
    """Test the grading engine with sample data"""
    
    # Configuration
    DATASET_PATH = "data/tqa-samples/development_set.json"
    OUTPUT_PATH = "evaluation/grading_results.json"
    
    try:
        # Initialize grading engine
        engine = ResponsibleGradingEngine()
        
        # Load development dataset
        questions = engine.load_development_dataset(DATASET_PATH)
        
        print(f"Loaded {len(questions)} questions for testing")
        
        # Test with first 5 questions using their correct answers as "student responses"
        test_questions = questions[:5]
        
        print("Grading test responses...")
        for i, question in enumerate(test_questions):
            # Use correct answer as student response for initial testing
            correct_choice = question['correct_answer']
            student_response = f"The answer is {correct_choice}"
            
            result = engine.grade_response(question, student_response)
            print(f"Question {i+1}: AI={result.ai_score}, Correct={result.correct_answer}, Confidence={result.confidence_score:.2f}")
        
        # Evaluate accuracy
        metrics = engine.evaluate_grading_accuracy()
        print(f"\nGrading Accuracy: {metrics['accuracy']:.2f}")
        print(f"Average Confidence: {metrics['average_confidence']:.2f}")
        
        # Save results
        engine.save_results(OUTPUT_PATH)
        
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        print("Please run the data extraction script first")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()