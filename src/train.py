"""
Senate AI - Topic-Based Training with AI Grading
Uses shared wordbank for tokenization. Intensive training for small bundles.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import json
import sys
from pathlib import Path
from model_template import SenateBundle
import random
from difflib import SequenceMatcher
from ai_client import call_ai
from wordbank import get_wordbank
import re


class TopicDataset(Dataset):
    """Training data using shared wordbank"""
    
    def __init__(self, texts, wordbank, seq_length=64):
        self.seq_length = seq_length
        self.wordbank = wordbank
        
        self.sequences = []
        for text in texts:
            tokens = wordbank.tokenize(text, max_len=seq_length)
            self.sequences.append(torch.tensor(tokens))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx][:-1]
        y = self.sequences[idx][1:]
        return x, y


def generate_training_data(topic, num_examples=30):
    """Generate fresh training examples using AI"""
    
    prompt = f"""Generate {num_examples} diverse training sentences for a tiny AI specializing in '{topic}'.
Each sentence should teach a key concept about {topic}.
Make them varied: definitions, principles, facts, applications, examples.
Return as JSON array of strings: ["sentence1", "sentence2", ...]"""
    
    response = call_ai(prompt, max_tokens=800)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples[:num_examples]
        except:
            pass
    
    return [f"{topic} is an important field of study involving key principles and concepts"]


def generate_qa_pairs(topic, num_pairs=8):
    """Generate fresh Q&A pairs for grading"""
    
    prompt = f"""Generate {num_pairs} question-answer pairs about '{topic}'.
Questions should test understanding of key concepts.
Answers should be 1-2 sentences, accurate.
Return as JSON array: [{{"question": "...", "answer": "..."}}, ...]"""
    
    response = call_ai(prompt, max_tokens=1000)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                pairs = json.loads(match.group())
                return pairs[:num_pairs]
        except:
            pass
    
    return [{"question": f"What is {topic}?", "answer": f"{topic} is a field of study"}]


def train_senator_on_topics(senator, topics, wordbank, epochs=15, lr=0.0005, batch_size=16):
    """Train a senator on freshly generated topic data using wordbank"""
    
    all_texts = []
    for topic in topics:
        texts = generate_training_data(topic)
        all_texts.extend(texts)
    
    if not all_texts:
        return None
    
    random.shuffle(all_texts)
    dataset = TopicDataset(all_texts, wordbank)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(senator.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    senator.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = senator(batch_x)
            loss = criterion(logits.permute(0, 2, 1), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(senator.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return losses


def grade_senator_with_ai(senator, qa_pairs, wordbank):
    """Grade senator using AI judge"""
    senator.eval()
    scores = []
    
    for qa in qa_pairs:
        question = qa.get('question', '')
        correct_answer = qa.get('answer', '')
        topic = qa.get('topic', '')
        
        # Use wordbank for tokenization
        tokens = wordbank.tokenize(question, max_len=32)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            generated = []
            current = input_ids
            
            for _ in range(15):
                logits = senator(current)
                last_logits = logits[0, -1, :]
                probs = torch.softmax(last_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == 2:
                    break
                generated.append(next_token)
                current = torch.cat([current, torch.tensor([[next_token]])], dim=1)
        
        senator_answer = wordbank.decode(torch.tensor(generated))
        
        grading_prompt = f"""You are grading an AI senator's answer.

Question: {question}
Correct answer: {correct_answer}
Senator's answer: {senator_answer}

Score the senator's answer from 0-100 based on:
- Correctness (60%)
- Relevance (20%)
- Clarity (20%)

Return ONLY a number between 0 and 100."""
        
        ai_score = call_ai(grading_prompt, max_tokens=10)
        
        if ai_score:
            try:
                score = float(''.join(c for c in ai_score if c.isdigit() or c == '.'))
                score = min(100, max(0, score))
            except:
                similarity = SequenceMatcher(None, correct_answer.lower(), senator_answer.lower()).ratio()
                score = similarity * 100
        else:
            similarity = SequenceMatcher(None, correct_answer.lower(), senator_answer.lower()).ratio()
            score = similarity * 100
        
        scores.append({
            'topic': topic,
            'question': question,
            'correct_answer': correct_answer,
            'senator_answer': senator_answer,
            'score': score
        })
    
    return scores


def grade_and_update(senator, topics, wordbank):
    """Grade senator on fresh Q&A pairs and update performance"""
    
    all_qa = []
    for topic in topics:
        qa_pairs = generate_qa_pairs(topic)
        for qa in qa_pairs:
            qa['topic'] = topic
        all_qa.extend(qa_pairs)
    
    if not all_qa:
        return None
    
    scores = grade_senator_with_ai(senator, all_qa, wordbank)
    
    if not scores:
        return None
    
    topic_scores = {}
    for s in scores:
        topic = s['topic']
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(s['score'])
    
    for topic, topic_score_list in topic_scores.items():
        avg_score = sum(topic_score_list) / len(topic_score_list)
        senator.performance[topic] = avg_score / 100
    
    avg_score = sum(s['score'] for s in scores) / len(scores)
    return {'average_score': avg_score, 'scores': scores}


def train_topics(topic_list, bundle_id=None, epochs=15, lr=0.0005):
    """Train all senators matching topics, optionally filtered by bundle"""
    
    with open('senate_bundles/senate_index.json') as f:
        index = json.load(f)
    
    topics = set(topic_list)
    print(f"\n{'='*60}")
    print(f"  TOPIC TRAINING - WORDBANK + AI GRADING")
    print(f"{'='*60}")
    print(f"  Topics: {', '.join(sorted(topics))}")
    if bundle_id is not None:
        print(f"  Bundle: {bundle_id}")
    print(f"  Epochs: {epochs} | LR: {lr}")
    sys.stdout.flush()
    
    wordbank = get_wordbank()
    print(f"  Wordbank: {len(wordbank.word_to_id):,} words")
    sys.stdout.flush()
    
    matching_senators = []
    for senator in index['senators']:
        senator_topics = set(senator['specialties'])
        if senator_topics & topics:
            if bundle_id is None or senator['bundle_id'] == bundle_id:
                matching_senators.append(senator)
    
    print(f"  Matching senators: {len(matching_senators)}")
    sys.stdout.flush()
    
    if not matching_senators:
        print("  No senators match these topics")
        return
    
    bundle_groups = {}
    for senator in matching_senators:
        bid = senator['bundle_id']
        if bid not in bundle_groups:
            bundle_groups[bid] = []
        bundle_groups[bid].append(senator)
    
    print(f"  Across {len(bundle_groups)} bundles")
    sys.stdout.flush()
    
    trained = 0
    graded = 0
    total_score = 0
    skipped = 0
    
    for bid, senators in sorted(bundle_groups.items()):
        bundle_path = f"senate_bundles/bundle_{bid:03d}.pt"
        
        if not Path(bundle_path).exists():
            print(f"  Bundle {bid} not found, skipping {len(senators)} senators")
            skipped += len(senators)
            continue
        
        print(f"\n  Bundle {bid} ({len(senators)} senators)...")
        sys.stdout.flush()
        
        bundle = SenateBundle.load(bundle_path)
        bundle_changed = False
        
        for senator_info in senators:
            senator_id = senator_info['senator_id']
            senator = bundle.get_senator(senator_id)
            
            if senator is None:
                print(f"    Senator {senator_id} not in bundle")
                skipped += 1
                continue
            
            senator_topics = set(senator.specialties)
            relevant = list(senator_topics & topics)
            
            print(f"    Senator {senator_id} [{', '.join(relevant[:3])}]...", end=' ')
            sys.stdout.flush()
            
            losses = train_senator_on_topics(senator, relevant, wordbank, epochs=epochs, lr=lr)
            
            if losses:
                trained += 1
                bundle_changed = True
                print(f"train done", end=' ')
            else:
                print(f"no data")
                skipped += 1
                continue
            
            grading_result = grade_and_update(senator, relevant, wordbank)
            
            if grading_result:
                score = grading_result['average_score']
                total_score += score
                graded += 1
                print(f"| grade: {score:.1f}/100")
            else:
                print(f"| no grading")
            
            sys.stdout.flush()
        
        if bundle_changed:
            size_mb = bundle.save(bundle_path)
            print(f"    Saved ({size_mb:.1f}MB)")
            sys.stdout.flush()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Trained: {trained} senators")
    print(f"  Graded: {graded} senators")
    if graded > 0:
        print(f"  Avg score: {total_score/graded:.1f}/100")
    print(f"  Skipped: {skipped}")
    print(f"  Topics: {', '.join(sorted(topics))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', type=str, required=True, help='Comma-separated topics')
    parser.add_argument('--bundle', type=int, default=None, help='Only train senators in this bundle')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    
    args = parser.parse_args()
    topics = [t.strip() for t in args.topics.split(',')]
    
    train_topics(topics, bundle_id=args.bundle, epochs=args.epochs, lr=args.lr)
