"""
Senate AI - The Parliament Runtime
Real senator inference with trained models.
Uses shared wordbank for tokenization and decoding.
"""

import torch
import yaml
import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from wordbank import get_wordbank
import random
import sys


class Senate:
    """The full Senate AI parliament system"""
    
    def __init__(self):
        with open('config.yaml') as f:
            self.config = yaml.safe_load(f)
        
        with open('senate_bundles/senate_index.json') as f:
            self.index = json.load(f)
        
        self.loaded_bundles = {}
        self.active_senators = {}
        self.session_history = []
        
        self.wordbank = get_wordbank()
        
        print(f"Senate ready: {len(self.index.get('senators', []))} senators")
    
    def _load_senator(self, senator_info):
        """Actually load a senator model from its bundle"""
        senator_id = senator_info['senator_id']
        
        if senator_id in self.active_senators:
            return self.active_senators[senator_id]
        
        bundle_id = senator_info['bundle_id']
        
        if bundle_id not in self.loaded_bundles:
            bundle_path = f"senate_bundles/bundle_{bundle_id:03d}.pt"
            
            if not Path(bundle_path).exists():
                return None
            
            self.loaded_bundles[bundle_id] = torch.load(bundle_path, map_location='cpu', weights_only=False)
        
        bundle = self.loaded_bundles[bundle_id]
        senators = bundle.get('senators', {})
        
        if str(senator_id) in senators:
            data = senators[str(senator_id)]
        elif senator_id in senators:
            data = senators[senator_id]
        else:
            return None
        
        from model_template import Senator
        config = data.get('config', {})
        state_dict = data.get('state_dict', data)
        
        senator = Senator(
            model_id=config.get('model_id', senator_id),
            specialties=config.get('specialties', ['general'])
        )
        
        clean_state = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                clean_state[k] = v
        
        senator.load_state_dict(clean_state, strict=False)
        senator.eval()
        
        self.active_senators[senator_id] = senator
        return senator
    
    def _tokenize(self, text, max_len=32):
        """Tokenize using shared wordbank"""
        tokens = self.wordbank.tokenize(text, max_len=max_len)
        return torch.tensor([tokens])
    
    def _decode(self, token_ids):
        """Decode using shared wordbank"""
        return self.wordbank.decode(token_ids)
    
    def _senator_inference(self, senator, question):
        """Run actual inference on a senator model"""
        input_ids = self._tokenize(question)
        
        with torch.no_grad():
            logits = senator(input_ids)
            last_logits = logits[0, -1, :]
            
            temperature = 0.8
            probs = torch.softmax(last_logits / temperature, dim=-1)
            
            generated = []
            current = input_ids
            
            for _ in range(random.randint(15, 30)):
                logits = senator(current)
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits / temperature, dim=-1)
                
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                cutoff = (cumsum > 0.8).nonzero()[0].item() + 1
                cutoff = max(cutoff, 3)
                
                top_probs = sorted_probs[:cutoff]
                top_indices = sorted_indices[:cutoff]
                top_probs = top_probs / top_probs.sum()
                
                next_token = top_indices[torch.multinomial(top_probs, 1)].item()
                
                if next_token == 2:
                    break
                
                generated.append(next_token)
                current = torch.cat([current, torch.tensor([[next_token]])], dim=1)
                
                if len(generated) >= 30:
                    break
            
            return self._decode(torch.tensor(generated))
    
    def router(self, question):
        """Keyword-based router"""
        question_lower = question.lower()
        
        topic_keywords = {
            'mathematics': ['math', 'calculate', 'number', 'equation', 'formula', 'prime', 'derivative', 'integral', 'algebra', 'geometry'],
            'physics': ['physics', 'force', 'energy', 'motion', 'gravity', 'light', 'speed', 'mass', 'quantum', 'wave'],
            'chemistry': ['chemistry', 'chemical', 'element', 'reaction', 'molecule', 'atom', 'bond', 'acid', 'base'],
            'biology': ['biology', 'cell', 'dna', 'organism', 'species', 'evolution', 'gene', 'protein', 'bacteria'],
            'computer_science': ['computer', 'code', 'algorithm', 'program', 'software', 'data', 'binary', 'python', 'java', 'api'],
            'history': ['history', 'war', 'ancient', 'century', 'revolution', 'empire', 'civilization', 'king', 'queen', 'president', 'founded', 'year'],
            'geography': ['capital', 'country', 'city', 'france', 'paris', 'london', 'continent', 'ocean', 'river', 'mountain', 'border', 'europe', 'asia', 'africa', 'america', 'australia'],
            'philosophy': ['philosophy', 'ethic', 'moral', 'existence', 'meaning', 'consciousness', 'reality', 'truth'],
            'logic': ['logic', 'reason', 'argument', 'valid', 'fallacy', 'premise', 'conclusion', 'deductive', 'inductive'],
            'psychology': ['psychology', 'mind', 'behavior', 'cognitive', 'emotion', 'mental', 'brain', 'personality'],
            'economics': ['economy', 'market', 'money', 'trade', 'supply', 'demand', 'price', 'inflation', 'gdp'],
            'linguistics': ['language', 'grammar', 'word', 'syntax', 'meaning', 'semantic', 'speech', 'phonetic'],
            'astronomy': ['space', 'star', 'planet', 'galaxy', 'universe', 'cosmic', 'orbit', 'nasa', 'mars'],
            'medicine': ['medicine', 'disease', 'treatment', 'symptom', 'diagnosis', 'drug', 'cancer', 'virus'],
            'law': ['law', 'legal', 'right', 'constitution', 'court', 'justice', 'crime', 'attorney', 'judge'],
            'art_history': ['art', 'painting', 'sculpture', 'artist', 'renaissance', 'modern', 'picasso', 'museum'],
            'music_theory': ['music', 'note', 'chord', 'rhythm', 'melody', 'harmony', 'scale', 'beat', 'song'],
            'environmental_science': ['environment', 'climate', 'ecosystem', 'pollution', 'sustainable', 'carbon', 'recycle'],
            'engineering': ['engineer', 'design', 'build', 'structure', 'machine', 'technical', 'circuit', 'mechanical'],
        }
        
        topic_scores = defaultdict(float)
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    topic_scores[topic] += 1
            if topic.replace('_', ' ') in question_lower:
                topic_scores[topic] += 3
        
        if not topic_scores:
            return ['logic', 'philosophy', 'history', 'geography']
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:8]]
    
    def select_senators(self, relevant_topics, max_senators=25):
        """Select senators matching relevant topics"""
        senator_scores = []
        
        for senator_info in self.index['senators']:
            senator_topics = set(senator_info['specialties'])
            relevant_set = set(relevant_topics)
            overlap = senator_topics & relevant_set
            
            if overlap:
                score = len(overlap) / len(relevant_set)
                bonus = sum(1 for t in overlap if t in senator_topics) * 0.1
                senator_scores.append((senator_info, score + bonus))
        
        senator_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        covered_topics = set()
        for senator_info, score in senator_scores:
            if len(selected) >= max_senators:
                break
            new_topics = set(senator_info['specialties']) - covered_topics
            if new_topics or len(selected) < 5:
                selected.append(senator_info)
                covered_topics.update(senator_info['specialties'])
        
        return selected[:max_senators]
    
    def grouper(self, answers):
        """Group similar answers together"""
        groups = []
        used = set()
        
        for i, (senator_id, answer) in enumerate(answers):
            if i in used:
                continue
            
            group = {'senators': [senator_id], 'answer': answer, 'count': 1}
            
            for j, (other_id, other_answer) in enumerate(answers):
                if j <= i or j in used:
                    continue
                similarity = SequenceMatcher(None, answer.lower(), other_answer.lower()).ratio()
                if similarity > 0.3:
                    group['senators'].append(other_id)
                    group['count'] += 1
                    used.add(j)
            
            groups.append(group)
            used.add(i)
        
        groups.sort(key=lambda x: x['count'], reverse=True)
        return groups
    
    def challenger_review(self, consensus, question):
        """Challenge the current consensus"""
        challenges = [
            "Are there unstated assumptions?",
            "Does this cover edge cases?",
            "Is there evidence for this?",
            "Could there be alternative explanations?",
            "Is the reasoning complete?",
        ]
        return random.choice(challenges)
    
    def vote(self, groups):
        """Vote on answer groups"""
        if not groups:
            return "No consensus reached.", 0.0
        
        total_votes = sum(g['count'] for g in groups)
        if total_votes == 0:
            return "No consensus reached.", 0.0
        
        leading = groups[0]
        return leading['answer'], leading['count'] / total_votes
    
    def ask(self, question):
        """Public interface: Ask the Senate a question with real inference"""
        print(f"\n{'='*60}")
        print(f"  SENATE DEBATE")
        print(f"{'='*60}")
        print(f"\nQuestion: {question}")
        sys.stdout.flush()
        
        print("\nRouter: Identifying relevant topics...")
        relevant_topics = self.router(question)
        print(f"Topics: {', '.join(relevant_topics)}")
        sys.stdout.flush()
        
        print("\nSelecting senators...")
        selected = self.select_senators(relevant_topics, max_senators=25)
        print(f"{len(selected)} senators selected")
        for s in selected[:10]:
            print(f"  Senator {s['senator_id']}: {', '.join(s['specialties'][:3])}")
        if len(selected) > 10:
            print(f"  ... and {len(selected)-10} more")
        sys.stdout.flush()
        
        print(f"\n{'─'*60}")
        print("  ROUND 1 - Independent Answers")
        print(f"{'─'*60}")
        sys.stdout.flush()
        
        answers = []
        for i, senator_info in enumerate(selected):
            senator = self._load_senator(senator_info)
            if senator is None:
                continue
            
            print(f"  [{len(answers)+1}/{len(selected)}] Senator {senator_info['senator_id']}...", end=' ')
            sys.stdout.flush()
            
            answer = self._senator_inference(senator, question)
            answers.append((senator_info['senator_id'], answer))
            print(f'"{answer[:80]}"')
            sys.stdout.flush()
        
        if not answers:
            return {'question': question, 'consensus': 'No senators available', 'confidence': 0.0, 'rounds': 0, 'senators_involved': 0, 'topics': relevant_topics}
        
        groups = self.grouper(answers)
        print(f"\n  Groups formed: {len(groups)}")
        for i, g in enumerate(groups[:8]):
            print(f"  Group {i+1}: {g['count']} votes - \"{g['answer'][:60]}...\"")
        if len(groups) > 8:
            print(f"  ... and {len(groups)-8} more groups")
        sys.stdout.flush()
        
        consensus, confidence = self.vote(groups)
        
        print(f"\n{'─'*60}")
        print("  ROUND 2 - Reconsider with Challenge")
        print(f"{'─'*60}")
        sys.stdout.flush()
        
        challenge = self.challenger_review(consensus, question)
        print(f"  Current consensus: \"{consensus[:80]}...\"")
        print(f"  Challenge: {challenge}")
        sys.stdout.flush()
        
        answers2 = []
        for i, senator_info in enumerate(selected):
            senator = self._load_senator(senator_info)
            if senator is None:
                continue
            
            reconsider_prompt = (
                f"Question: {question}\n"
                f"Current answer: {consensus}\n"
                f"Critique: {challenge}\n"
                f"Provide your revised answer:"
            )
            
            print(f"  [{len(answers2)+1}/{len(selected)}] Senator {senator_info['senator_id']}...", end=' ')
            sys.stdout.flush()
            
            answer = self._senator_inference(senator, reconsider_prompt)
            answers2.append((senator_info['senator_id'], answer))
            print(f'"{answer[:80]}"')
            sys.stdout.flush()
        
        if answers2:
            groups2 = self.grouper(answers2)
            consensus2, confidence2 = self.vote(groups2)
            
            if confidence2 >= confidence:
                consensus = consensus2
                confidence = confidence2
                rounds = 2
            else:
                rounds = 1
        else:
            rounds = 1
        
        print(f"\n{'='*60}")
        print(f"  FINAL ANSWER")
        print(f"{'='*60}")
        print(f"\n{consensus}")
        print(f"\nConfidence: {confidence:.1%}")
        print(f"Rounds: {rounds}")
        print(f"Senators involved: {len(answers)}")
        print(f"Topics: {', '.join(relevant_topics)}")
        sys.stdout.flush()
        
        result = {
            'question': question,
            'consensus': consensus,
            'confidence': confidence,
            'rounds': rounds,
            'senators_involved': len(answers),
            'topics': relevant_topics
        }
        
        self.session_history.append(result)
        return result


if __name__ == "__main__":
    senate = Senate()
    
    questions = [
        "What is the capital of France?",
        "Why does ice float?",
        "How do computers work?",
    ]
    
    for q in questions:
        senate.ask(q)
        print("\n" + "="*60)
