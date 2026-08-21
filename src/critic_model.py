"""
Senate AI - Critic Neural Network
Trained on critic_training data. Scores and groups answers.
"""

import torch
import torch.nn as nn

class CriticModel(nn.Module):
    """Judges and groups senator answers"""
    
    def __init__(self, vocab_size=8000, embed_dim=64, hidden_dim=128):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Critic head: scores answer 0-100
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Grouper head: outputs group embedding
        self.group_head = nn.Linear(hidden_dim, 32)
    
    def forward(self, input_ids):
        emb = self.embed(input_ids)
        lstm_out, (hidden, _) = self.lstm(emb)
        features = hidden[-1]
        
        score = self.score_head(features) * 100
        group_embed = self.group_head(features)
        
        return score, group_embed
