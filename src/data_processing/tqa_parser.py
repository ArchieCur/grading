#!/usr/bin/env python3
"""
TQA Data Extraction Script
Extracts sample questions and lesson context for grading system development
"""

import json
import random
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Question:
    """Structure for a single TQA question with context"""
    question_id: str
    lesson_id: str
    lesson_name: str
    question_text: str
    answer_choices: Dict[str, str]
    correct_answer: str
    question_type: str
    question_subtype: str
    lesson_context: Dict[str, str]  # topic_id -> content
    vocabulary: Dict[str, str]      # term -> definition

class TQAExtractor:
    """Extract and process TQA dataset for grading development"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.train_data = None
        self.extracted_questions = []
    
    def load_training_data(self):
        """Load the training set JSON file"""
        train_file = self.dataset_path / "train" / "tqa_v1_train.json"
        
        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found at {train_file}")
        
        print(f"Loading training data from {train_file}")
        with open(train_file, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        
        print(f"Loaded {len(self.train_data)} lessons")
    
    def extract_lesson_context(self, lesson: Dict) -> Dict[str, str]:
        """Extract all textual context from a lesson"""
        context = {}
        
        # Extract main topics
        if "topics" in lesson:
            for topic_id, topic_data in lesson["topics"].items():
                if "content" in topic_data and "text" in topic_data["content"]:
                    context[f"topic_{topic_id}"] = topic_data["content"]["text"]
        
        # Extract adjunct topics (vocabulary, summary, etc.)
        if "adjunctTopics" in lesson:
            for section_name, section_data in lesson["adjunctTopics"].items():
                if isinstance(section_data, dict) and "content" in section_data:
                    if "text" in section_data["content"]:
                        context[f"adjunct_{section_name}"] = section_data["content"]["text"]
        
        return context
    
    def extract_vocabulary(self, lesson: Dict) -> Dict[str, str]:
        """Extract vocabulary definitions from lesson"""
        vocab = {}
        
        if "adjunctTopics" in lesson and "Vocabulary" in lesson["adjunctTopics"]:
            vocab_section = lesson["adjunctTopics"]["Vocabulary"]
            if isinstance(vocab_section, dict):
                vocab.update(vocab_section)
        
        return vocab
    
    def process_question(self, question_id: str, question_data: Dict, lesson: Dict) -> Optional[Question]:
        """Process a single question into our standard format"""
        
        # Extract basic question info
        question_text = question_data.get("beingAsked", {}).get("processedText", "")
        if not question_text:
            return None
        
        # Extract answer choices
        answer_choices = {}
        if "answerChoices" in question_data:
            for choice_id, choice_data in question_data["answerChoices"].items():
                choice_text = choice_data.get("processedText", "")
                if choice_text:
                    answer_choices[choice_id] = choice_text
        
        # Extract correct answer
        correct_answer = question_data.get("correctAnswer", {}).get("processedText", "")
        
        # Extract question metadata
        question_type = question_data.get("questionType", "")
        question_subtype = question_data.get("questionSubType", "")
        
        # Extract lesson context
        lesson_context = self.extract_lesson_context(lesson)
        vocabulary = self.extract_vocabulary(lesson)
        
        return Question(
            question_id=question_id,
            lesson_id=lesson.get("globalID", ""),
            lesson_name=lesson.get("lessonName", ""),
            question_text=question_text,
            answer_choices=answer_choices,
            correct_answer=correct_answer,
            question_type=question_type,
            question_subtype=question_subtype,
            lesson_context=lesson_context,
            vocabulary=vocabulary
        )
    
    def extract_text_questions(self, sample_size: int = 50) -> List[Question]:
        """Extract sample of text-only questions"""
        
        if not self.train_data:
            self.load_training_data()
        
        text_questions = []
        
        # Iterate through all lessons
        for lesson in self.train_data:
            if "questions" not in lesson:
                continue
            
            # Extract non-diagram questions only
            if "nonDiagramQuestions" in lesson["questions"]:
                for q_id, q_data in lesson["questions"]["nonDiagramQuestions"].items():
                    question = self.process_question(q_id, q_data, lesson)
                    if question:
                        text_questions.append(question)
        
        print(f"Found {len(text_questions)} total text questions")
        
        # Random sample for development
        if len(text_questions) > sample_size:
            sampled_questions = random.sample(text_questions, sample_size)
        else:
            sampled_questions = text_questions
        
        print(f"Selected {len(sampled_questions)} questions for development")
        return sampled_questions
    
    def save_sample_dataset(self, questions: List[Question], output_path: str):
        """Save extracted questions to JSON file"""
        
        # Convert questions to dictionaries for JSON serialization
        questions_data = []
        for q in questions:
            questions_data.append({
                "question_id": q.question_id,
                "lesson_id": q.lesson_id,
                "lesson_name": q.lesson_name,
                "question_text": q.question_text,
                "answer_choices": q.answer_choices,
                "correct_answer": q.correct_answer,
                "question_type": q.question_type,
                "question_subtype": q.question_subtype,
                "lesson_context": q.lesson_context,
                "vocabulary": q.vocabulary
            })
        
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(questions_data)} questions to {output_file}")
    
    def analyze_question_types(self, questions: List[Question]):
        """Analyze the distribution of question types in our sample"""
        
        type_counts = {}
        subtype_counts = {}
        
        for q in questions:
            # Count question types
            q_type = q.question_type or "unknown"
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            
            # Count question subtypes
            q_subtype = q.question_subtype or "unknown"
            subtype_counts[q_subtype] = subtype_counts.get(q_subtype, 0) + 1
        
        print("\nQuestion Type Distribution:")
        for q_type, count in sorted(type_counts.items()):
            print(f"  {q_type}: {count}")
        
        print(f"\nQuestion Subtype Distribution:")
        for q_subtype, count in sorted(subtype_counts.items()):
            print(f"  {q_subtype}: {count}")

def main():
    """Main extraction workflow"""
    
    # Configuration
    DATASET_PATH = r"E:\Grading\TQA datset\tqa_train_val_test\tqa_train_val_test"  # Update this path
    OUTPUT_PATH = "data/tqa-samples/development_set.json"
    SAMPLE_SIZE = 50
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    try:
        # Initialize extractor
        extractor = TQAExtractor(DATASET_PATH)
        
        # Extract sample questions
        questions = extractor.extract_text_questions(SAMPLE_SIZE)
        
        # Analyze the sample
        extractor.analyze_question_types(questions)
        
        # Save for development
        extractor.save_sample_dataset(questions, OUTPUT_PATH)
        
        print(f"\nExtraction complete! Ready for grading engine development.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update DATASET_PATH to point to your TQA dataset directory")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()