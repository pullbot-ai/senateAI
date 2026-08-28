"""
Senate AI - AI Client
Uses HuggingFace free inference API. No API key needed.
"""

import requests
import json
import re

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"


def call_ai(prompt, max_tokens=500, model="mistralai/Mistral-7B-Instruct-v0.2"):
    """Call HuggingFace free inference API - no key required"""
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        r = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0].get('generated_text', '')
            elif isinstance(data, dict):
                return data.get('generated_text', '')
            else:
                return str(data)
        else:
            print(f'   HF API {r.status_code}: {r.text[:100]}')
    
    except Exception as e:
        print(f'   HF call failed: {e}')
    
    return None
