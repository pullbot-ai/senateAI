"""
Senate AI - AI Client
Runs Qwen2.5-0.5B locally in GitHub Actions. No API needed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_model = None
_tokenizer = None


def load_model():
    """Load or get cached model"""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("   Loading Qwen2.5-0.5B locally...")
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        _model.eval()
        print("   Model ready")
    except Exception as e:
        print(f"   Model load failed: {e}")
        _model = None
        _tokenizer = None
    
    return _model, _tokenizer


def call_ai(prompt, max_tokens=500, model=None):
    """Generate text using local Qwen2.5-0.5B"""
    
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return None
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        print(f"   Generation failed: {e}")
        return None
