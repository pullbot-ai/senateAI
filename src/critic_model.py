"""
Senate AI - Critic Neural Network
Trained on critic_training data. Scores and groups answers.
Replaces Puter.js for judging and grouping.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path


class CriticModel(nn.Module):
    """Judges and groups senator answers"""
    
    def __init__(self, vocab_size=8000, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Critic head: scores answer 0-100
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Grouper head: outputs group embedding (32-dim)
        self.group_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, input_ids):
        emb = self.embed(input_ids)
        lstm_out, (hidden, _) = self.lstm(emb)
        features = hidden[-1]
        
        score = self.score_head(features) * 100
        group_embed = self.group_head(features)
        
        return score, group_embed
    
    def judge_answer(self, input_ids):
        """Score a single answer 0-100"""
        self.eval()
        with torch.no_grad():
            score, _ = self.forward(input_ids)
        return score.item()
    
    def get_group_embedding(self, input_ids):
        """Get group embedding for an answer"""
        self.eval()
        with torch.no_grad():
            _, group_embed = self.forward(input_ids)
        return group_embed
    
    def save(self, path='models/critic.pt'):
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.embed.num_embeddings,
            'embed_dim': self.embed.embedding_dim,
            'hidden_dim': self.lstm.hidden_size
        }, path)
    
    @classmethod
    def load(cls, path='models/critic.pt'):
        if not Path(path).exists():
            return None
        
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


def group_answers_with_critic(critic, wordbank, answers):
    """Use trained critic to group answers by semantic similarity"""
    
    if len(answers) <= 1:
        return [{'senators': [answers[0][0]], 'answer': answers[0][1], 'count': 1}]
    
    # Get group embeddings for each answer
    embeddings = []
    for senator_id, answer in answers:
        tokens = wordbank.tokenize(answer, max_len=32)
        input_ids = torch.tensor([tokens])
        embedding = critic.get_group_embedding(input_ids)
        embeddings.append((senator_id, answer, embedding))
    
    # Cluster by cosine similarity
    groups = []
    used = set()
    
    for i, (senator_id, answer, emb_i) in enumerate(embeddings):
        if i in used:
            continue
        
        group = {'senators': [senator_id], 'answer': answer, 'count': 1}
        
        for j, (other_id, other_answer, emb_j) in enumerate(embeddings):
            if j <= i or j in used:
                continue
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb_i, emb_j, dim=0).item()
            
            if similarity > 0.7:
                group['senators'].append(other_id)
                group['count'] += 1
                used.add(j)
        
        groups.append(group)
        used.add(i)
    
    groups.sort(key=lambda x: x['count'], reverse=True)
    return groups


def score_answer_with_critic(critic, wordbank, question, answer):
    """Use trained critic to score an answer"""
    
    combined = f"{question} {answer}"
    tokens = wordbank.tokenize(combined, max_len=64)
    input_ids = torch.tensor([tokens])
    
    return critic.judge_answer(input_ids)
