"""
Senate AI - Wordbank Loader
Loads the shared vocabulary from data/wordbank.json.
All senators use this for consistent tokenization and decoding.
"""

import json
from pathlib import Path


class Wordbank:
    """Shared vocabulary for all senators"""
    
    def __init__(self, wordbank_path='data/wordbank.json'):
        self.wordbank_path = Path(wordbank_path)
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1, '<END>': 2}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<END>'}
        self.definitions = {}
        self.load()
    
    def load(self):
        """Load wordbank from JSON file"""
        if not self.wordbank_path.exists():
            print(f"   Wordbank not found at {self.wordbank_path}, using empty vocab")
            return
        
        try:
            with open(self.wordbank_path) as f:
                bank = json.load(f)
        except Exception as e:
            print(f"   Failed to load wordbank: {e}")
            return
        
        words = bank.get('words', {})
        loaded = 0
        defined = 0
        
        for word, info in words.items():
            if isinstance(info, dict) and 'token_id' in info:
                tid = info['token_id']
                self.word_to_id[word] = tid
                self.id_to_word[tid] = word
                
                if info.get('has_definition') and info.get('definition'):
                    self.definitions[word] = info['definition']
                    defined += 1
                
                loaded += 1
            else:
                import hashlib
                tid = 3 + (int(hashlib.md5(word.encode()).hexdigest(), 16) % 7997)
                self.word_to_id[word] = tid
                self.id_to_word[tid] = word
                loaded += 1
        
        print(f"   Wordbank: {loaded:,} words, {defined:,} defined")
    
    def tokenize(self, text, max_len=64):
        """Convert text to token IDs"""
        words = text.lower().split()[:max_len]
        tokens = []
        
        for w in words:
            w = w.strip('.,!?;:()[]{}"\'')
            if w in self.word_to_id:
                tokens.append(self.word_to_id[w])
            else:
                tokens.append(1)
        
        tokens.append(2)
        
        while len(tokens) < max_len:
            tokens.append(0)
        
        return tokens[:max_len]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        words = []
        
        for tid in token_ids:
            tid = tid.item() if hasattr(tid, 'item') else int(tid)
            
            if tid == 0:
                break
            if tid == 1:
                words.append('?')
            elif tid == 2:
                break
            elif tid in self.id_to_word:
                words.append(self.id_to_word[tid])
            else:
                words.append(f'[{tid}]')
        
        return ' '.join(words) if words else "..."
    
    def get_definition(self, word):
        """Get definition for a word"""
        return self.definitions.get(word.lower(), '')
    
    def get_vocab_size(self):
        """Get number of words in vocab"""
        return max(self.id_to_word.keys()) + 1 if self.id_to_word else 8000


_wordbank = None

def get_wordbank():
    global _wordbank
    if _wordbank is None:
        _wordbank = Wordbank()
    return _wordbank
