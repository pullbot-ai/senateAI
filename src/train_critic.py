"""
Senate AI - Critic Trainer
Trains the Challenger to distinguish good answers from bad ones.
AI generates examples with increasing difficulty levels.
"""

import json
import yaml
import time
import random
from pathlib import Path
from ai_client import call_ai
import re


def generate_critic_examples(topic, difficulty, num_examples=10):
    """Generate examples for training the critic."""
    
    prompt = f"""Generate {num_examples} question-answer pairs for training an AI critic.

Topic: {topic}
Difficulty: {difficulty}/5

For each question:
- good_answer: accurate and well-reasoned
- flawed_answer: has a specific error
- feedback: explain what's wrong

Return JSON array:
[{{"question":"...","good_answer":"...","flawed_answer":"...","flaw_type":"factual_error|logical_fallacy|incomplete|misleading|oversimplified","feedback":"...","difficulty":{difficulty}}}]"""
    
    response = call_ai(prompt, max_tokens=1500)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples[:num_examples]
        except Exception as e:
            print(f"   Parse error: {e}")
    
    # Fallback: simple local examples
    fallback = []
    for i in range(min(num_examples, 5)):
        fallback.append({
            'question': f'What is a key concept in {topic}?',
            'good_answer': f'{topic} involves fundamental principles and their practical applications.',
            'flawed_answer': f'{topic} is only about memorizing facts.',
            'flaw_type': 'oversimplified',
            'feedback': f'The flawed answer ignores the analytical and applied aspects of {topic}.',
            'difficulty': difficulty
        })
    
    return fallback


def generate_questions_only(topic, difficulty, num_questions=10):
    """Generate questions for the critic to evaluate"""
    
    prompt = f"""Generate {num_questions} questions about '{topic}'.
Difficulty: {difficulty}/5

Return as JSON array of strings: ["question1", "question2", ...]"""
    
    response = call_ai(prompt, max_tokens=500)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
    
    return [f"Explain the key concepts of {topic}."]


def generate_all_critic_data():
    """Generate training data for all topics across all difficulty levels"""
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    topics = config['topics']
    
    output_dir = Path('critic_training')
    output_dir.mkdir(exist_ok=True)
    
    all_examples = []
    all_questions = []
    
    for difficulty in range(1, 6):
        print(f"\n{'='*60}")
        print(f"  DIFFICULTY LEVEL {difficulty}/5")
        print(f"{'='*60}")
        
        level_examples = []
        level_questions = []
        
        sample_topics = random.sample(topics, min(15, len(topics)))
        
        for i, topic in enumerate(sample_topics):
            print(f"\n[{i+1}/{len(sample_topics)}] {topic} (difficulty {difficulty})...")
            
            print("   Generating good/bad answer pairs...")
            examples = generate_critic_examples(topic, difficulty)
            
            if examples:
                level_examples.extend(examples)
                all_examples.extend(examples)
                print(f"   {len(examples)} pairs generated")
            else:
                print(f"   Failed to generate")
            
            print("   Generating test questions...")
            questions = generate_questions_only(topic, difficulty)
            
            if questions:
                for q in questions:
                    level_questions.append({
                        "question": q,
                        "topic": topic,
                        "difficulty": difficulty
                    })
                all_questions.extend(level_questions[-len(questions):])
                print(f"   {len(questions)} questions")
            
            time.sleep(0.5)
        
        level_data = {
            "difficulty": difficulty,
            "examples": level_examples,
            "questions": level_questions,
            "example_count": len(level_examples),
            "question_count": len(level_questions)
        }
        
        with open(output_dir / f"difficulty_{difficulty}.json", 'w') as f:
            json.dump(level_data, f, indent=2)
        
        print(f"\n   Level {difficulty} saved: {len(level_examples)} examples, {len(level_questions)} questions")
    
    combined = {
        "total_examples": len(all_examples),
        "total_questions": len(all_questions),
        "by_difficulty": {},
        "examples": all_examples,
        "questions": all_questions
    }
    
    for ex in all_examples:
        diff = ex.get('difficulty', 1)
        if str(diff) not in combined['by_difficulty']:
            combined['by_difficulty'][str(diff)] = []
        combined['by_difficulty'][str(diff)].append(ex)
    
    with open(output_dir / "all_critic_data.json", 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  CRITIC TRAINING DATA COMPLETE")
    print(f"{'='*60}")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  Difficulty levels: 1-5")
    print(f"  Saved to: {output_dir}/")
    
    for diff in range(1, 6):
        count = len(combined['by_difficulty'].get(str(diff), []))
        print(f"    Level {diff}: {count} examples")


if __name__ == "__main__":
    generate_all_critic_data()
