"""
Senate AI - Punctuator using HuggingFace model
Downloads the model once, uses locally. No API calls.
"""

import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
CACHE_DIR = Path('models/punctuator')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_punctuator_model = None
_punctuator_tokenizer = None


def load_punctuator():
    """Load or download the punctuation model"""
    global _punctuator_model, _punctuator_tokenizer
    
    if _punctuator_model is not None:
        return _punctuator_model, _punctuator_tokenizer
    
    print("Loading punctuator model...")
    
    try:
        model_path = CACHE_DIR / 'model'
        tokenizer_path = CACHE_DIR / 'tokenizer'
        
        if model_path.exists() and tokenizer_path.exists():
            _punctuator_model = AutoModelForTokenClassification.from_pretrained(model_path)
            _punctuator_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            _punctuator_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
            _punctuator_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            _punctuator_model.save_pretrained(model_path)
            _punctuator_tokenizer.save_pretrained(tokenizer_path)
        
        _punctuator_model.eval()
        print("Punctuator ready.")
        
    except Exception as e:
        print(f"Punctuator load failed: {e}")
        _punctuator_model = None
        _punctuator_tokenizer = None
    
    return _punctuator_model, _punctuator_tokenizer


def punctuate(text):
    """Add punctuation and capitalization to text"""
    
    if not text or len(text) < 2:
        return text
    
    # Clean token IDs first
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text or text == '...':
        return text
    
    # Check for math expressions first
    math_match = re.search(r'\b(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\b', text)
    if math_match:
        try:
            a = float(math_match.group(1))
            b = float(math_match.group(3))
            op = math_match.group(2)
            
            if op == '+': result = a + b
            elif op == '-': result = a - b
            elif op == '*': result = a * b
            elif op == '/': result = a / b if b != 0 else 'undefined'
            else: result = '?'
            
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)
        except:
            pass
    
    model, tokenizer = load_punctuator()
    
    if model is None or tokenizer is None:
        # Fallback to basic rules
        text = text[0].upper() + text[1:] if text else text
        if text[-1] not in '.!?':
            text += '.'
        return text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Reconstruct text with punctuation
        result = ''
        for token, pred in zip(tokens, predictions[0]):
            if token.startswith('##'):
                result += token[2:]
            elif token in ['.', ',', '!', '?', ';', ':']:
                result += token
            elif token in ['[SEP]', '[CLS]', '[PAD]']:
                continue
            else:
                if result and token not in ['.', ',', '!', '?']:
                    result += ' '
                result += token
        
        # Clean up
        result = result.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result if result else text
        
    except Exception as e:
        print(f"Punctuation failed: {e}")
        return text


def punctuate_text(text):
    """Quick helper"""
    return punctuate(text)
