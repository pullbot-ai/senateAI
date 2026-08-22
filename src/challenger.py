"""
Senate AI - Challenger
Uses critic training data to find flaws in consensus answers.
"""

import json
from pathlib import Path
from ai_client import call_ai


class Challenger:
    """Dedicated challenger that attacks consensus answers"""
    
    def __init__(self, training_data_path='critic_training/all_training_data.json'):
        self.training_data_path = Path(training_data_path)
        self.critic_examples = []
        self.load_training_data()
    
    def load_training_data(self):
        """Load critic training examples for context"""
        if self.training_data_path.exists():
            try:
                with open(self.training_data_path) as f:
                    data = json.load(f)
                    self.critic_examples = data.get('critic_examples', [])
                    print(f"   Challenger: {len(self.critic_examples)} training examples loaded")
            except:
                pass
    
    def review(self, consensus, question):
        """Attack the consensus answer"""
        
        # Include examples of good criticism for context
        example_feedback = []
        for ex in self.critic_examples[:3]:
            if ex.get('feedback'):
                example_feedback.append(ex['feedback'])
        
        context = ""
        if example_feedback:
            context = "Examples of good criticism:\n" + "\n".join(example_feedback[:2]) + "\n\n"
        
        prompt = f"""You are a dedicated Challenger in an AI parliament debate.

Your job is to find flaws in the current consensus answer.

{context}Question: {question}
Current consensus: {consensus}

Find specific problems:
- Contradictions
- Missing information
- Incorrect assumptions
- Exceptions or edge cases
- Unsupported claims
- Calculation errors

Return a concise critique that senators can use to improve their answers."""
        
        response = call_ai(prompt, max_tokens=150)
        
        if response and len(response) > 5:
            return response
        
        # Fallback challenges
        challenges = [
            "Are there unstated assumptions that could be wrong?",
            "Does this answer cover edge cases and exceptions?",
            "Is there empirical evidence supporting this claim?",
            "Could there be alternative explanations not considered?",
            "Is the reasoning chain complete or are there gaps?",
            "Are there contradictions with established facts?",
        ]
        
        import random
        return random.choice(challenges)
