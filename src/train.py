"""
Senate AI - Topic-Based Training with AI Grading
Uses shared wordbank for tokenization. AI-guided parameter selection.
Fresh data every 5 epochs to prevent overfitting.
STRICT AI grading. Empty answers auto-score 0.
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
    """Generate fresh training examples using AI with punctuation"""
    
    prompt = f"""Generate {num_examples} diverse training sentences for a tiny AI specializing in '{topic}'.
Each sentence MUST end with a period, question mark, or exclamation point.
Use commas where appropriate.
Make them varied: definitions, principles, facts, applications, examples.
Return as JSON array of strings: ["sentence1.", "sentence2?", "sentence3!"]"""
    
    response = call_ai(prompt, max_tokens=1500)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group())
                return examples[:num_examples]
        except:
            pass
    
    return [
        f"{topic} is an important field of study involving key principles and concepts.",
        f"The study of {topic} requires analytical thinking and practical application.",
        f"Experts in {topic} apply specialized knowledge to solve complex problems.",
        f"Understanding {topic} involves mastering fundamental concepts and theories.",
        f"{topic} continues to evolve with new research and discoveries.",
    ]


def generate_qa_pairs(topic, num_pairs=8):
    """Generate fresh Q&A pairs for grading with punctuation"""
    
    prompt = f"""Generate {num_pairs} question-answer pairs about '{topic}'.
Questions should end with question marks.
Answers should be 1-2 sentences ending with periods.
Return as JSON array: [{{"question": "...?", "answer": "..."}}, ...]"""
    
    response = call_ai(prompt, max_tokens=1000)
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                pairs = json.loads(match.group())
                return pairs[:num_pairs]
        except:
            pass
    
    return [{"question": f"What is {topic}?", "answer": f"{topic} is a field of study."}]


def select_params_with_ai(senator, topics, epoch, epochs, current_loss):
    """Ask AI which parameter groups to update this epoch"""
    
    param_names = list(dict(senator.named_parameters()).keys())
    
    prompt = f"""You are optimizing a 1.8M parameter AI senator specializing in {', '.join(topics[:3])}.
Epoch {epoch+1}/{epochs}
Current loss: {current_loss}

Parameter groups:
{', '.join(param_names[:30])}

Which 5 parameter groups should we focus on updating this epoch?
Return ONLY as JSON array: ["param1", "param2", "param3", "param4", "param5"]"""
    
    response = call_ai(prompt, max_tokens=100)
    selected = []
    
    if response:
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                selected = json.loads(match.group())
        except:
            pass
    
    if selected:
        matched = []
        for name, param in senator.named_parameters():
            for sel in selected:
                if sel.split('.')[0] in name or name.split('.')[0] in sel:
                    matched.append(name)
                    break
        return matched[:10] if matched else None
    
    return None


def train_senator_on_topics(senator, topics, wordbank, epochs=50, lr=0.0005, batch_size=16, ai_guided=True, refresh_interval=5):
    """Train senator with fresh data every 5 epochs to prevent overfitting."""
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    senator.train()
    
    all_losses = []
    total_epochs_done = 0
    num_rounds = epochs // refresh_interval
    
    for round_num in range(num_rounds):
        all_texts = []
        for topic in topics:
            texts = generate_training_data(topic, num_examples=30)
            all_texts.extend(texts)
        
        random.shuffle(all_texts)
        dataset = TopicDataset(all_texts, wordbank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"    Round {round_num+1}/{num_rounds}: fresh data ({len(all_texts)} examples)")
        sys.stdout.flush()
        
        for epoch in range(refresh_interval):
            # ALWAYS reset all params to trainable first
            for param in senator.parameters():
                param.requires_grad = True
            
            selected_tensors = []
            
            if ai_guided and total_epochs_done >= 2:
                current_loss = all_losses[-1] if all_losses else 8.0
                selected_names = select_params_with_ai(senator, topics, total_epochs_done, epochs, current_loss)
                
                if selected_names:
                    selected_tensors = []
                    for name, param in senator.named_parameters():
                        if name in selected_names:
                            param.requires_grad = True
                            selected_tensors.append(param)
                        else:
                            param.requires_grad = False
            
            # Fallback: if no tensors selected, use all
            if not selected_tensors:
                for param in senator.parameters():
                    param.requires_grad = True
                    selected_tensors.append(param)
            
            optimizer = torch.optim.Adam(selected_tensors, lr=lr)
            
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                logits = senator(batch_x)
                loss = criterion(logits.permute(0, 2, 1), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected_tensors, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            all_losses.append(avg_loss)
            total_epochs_done += 1
            
            if total_epochs_done % 5 == 0 or total_epochs_done == 1:
                selected_count = len(selected_tensors) if selected_tensors else 'all'
                print(f"    Epoch {total_epochs_done}/{epochs} - Loss: {avg_loss:.4f} | Params: {selected_count}")
                sys.stdout.flush()
    
    return all_losses


def grade_senator_with_ai(senator, qa_pairs, wordbank):
    """Grade senator using STRICT AI judge. Empty answers auto-score 0."""
    senator.eval()
    scores = []
    
    for qa in qa_pairs:
        question = qa.get('question', '')
        correct_answer = qa.get('answer', '')
        topic = qa.get('topic', '')
        
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
        
        # Auto-score 0 for empty or "..." answers
        cleaned_answer = senator_answer.replace('[', '').replace(']', '').strip()
        if (not cleaned_answer or cleaned_answer == '...' or cleaned_answer == '.' 
            or cleaned_answer == '..' or len(cleaned_answer) <= 3):
            scores.append({
                'topic': topic,
                'question': question,
                'correct_answer': correct_answer,
                'senator_answer': senator_answer,
                'score': 0
            })
            continue
        
        grading_prompt = f"""You are a STRICT grader evaluating an AI senator's answer.

Question: {question}
Correct answer: {correct_answer}
Senator's answer: {senator_answer}

STRICT SCORING RUBRIC:
- 0-5: Complete gibberish, no relation to question at all
- 5-15: Uses some related words but answer doesn't address the question
- 15-30: Attempts to answer but mostly incorrect or incoherent
- 30-50: Partially correct with significant errors
- 50-70: Mostly correct, minor errors
- 70-85: Correct and clear with good phrasing
- 85-100: Model answer quality, directly answers the question perfectly

IMPORTANT RULES:
- If the answer doesn't directly address the question, MAX score is 15
- Unrelated words even if they form coherent phrases = max 15
- 100 is reserved for perfect model answers
- Be harsh. Better to under-score than over-score.

Return ONLY a number between 0 and 100."""
        
        ai_score = call_ai(grading_prompt, max_tokens=10)
        
        if ai_score:
            try:
                score = float(''.join(c for c in ai_score if c.isdigit() or c == '.'))
                
                # Qwen returns scores on 0-10 scale (e.g., "5.7")
                # Convert to 0-100 if score is 10 or less
                if score <= 10:
                    score = score * 10
                
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


def grade_and_update(senator, topics, wordbank, losses=None):
    """Grade senator and update performance"""
    
    all_qa = []
    for topic in topics:
        qa_pairs = generate_qa_pairs(topic)
        for qa in qa_pairs:
            qa['topic'] = topic
        all_qa.extend(qa_pairs)
    
    if not all_qa:
        return None
    
    scores = grade_senator_with_ai(senator, all_qa, wordbank)
    
    if scores:
        avg_ai_score = sum(s['score'] for s in scores) / len(scores)
        avg_score = avg_ai_score
    else:
        avg_score = 0
    
    for topic in topics:
        senator.performance[topic] = avg_score / 100
    
    return {'average_score': avg_score, 'scores': scores}


def train_topics(topic_list, bundle_id=None, epochs=50, lr=0.0005):
    """Train all senators matching topics, optionally filtered by bundle"""
    
    with open('senate_bundles/senate_index.json') as f:
        index = json.load(f)
    
    topics = set(topic_list)
    print(f"\n{'='*60}")
    print(f"  TOPIC TRAINING - WORDBANK + AI GUIDED + ANTI-OVERFIT")
    print(f"{'='*60}")
    print(f"  Topics: {', '.join(sorted(topics))}")
    if bundle_id is not None:
        print(f"  Bundle: {bundle_id}")
    print(f"  Epochs: {epochs} (fresh data every 5) | LR: {lr}")
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
            
            losses = train_senator_on_topics(
                senator, relevant, wordbank,
                epochs=epochs, lr=lr,
                ai_guided=True, refresh_interval=5
            )
            
            if losses:
                trained += 1
                bundle_changed = True
                print(f"train done", end=' ')
            else:
                print(f"no data")
                skipped += 1
                continue
            
            grading_result = grade_and_update(senator, relevant, wordbank, losses)
            
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--ai-guided', action='store_true', default=True, help='Use AI to select params each epoch')
    
    args = parser.parse_args()
    topics = [t.strip() for t in args.topics.split(',')]
    
    train_topics(topics, bundle_id=args.bundle, epochs=args.epochs, lr=args.lr)
