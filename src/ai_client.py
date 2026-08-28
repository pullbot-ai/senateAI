"""
Senate AI - AI Client
Tries multiple free AI endpoints. No API key needed.
"""

import requests
import json

ENDPOINTS = [
    {
        "url": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        "headers": {"Content-Type": "application/json"},
        "json_key": "inputs"
    },
    {
        "url": "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
        "headers": {"Content-Type": "application/json"},
        "json_key": "inputs"
    },
    {
        "url": "https://api-inference.huggingface.co/pipeline/text-generation/mistralai/Mistral-7B-Instruct-v0.2",
        "headers": {"Content-Type": "application/json"},
        "json_key": "inputs"
    }
]


def call_ai(prompt, max_tokens=500, model="mistralai/Mistral-7B-Instruct-v0.2"):
    """Try multiple HuggingFace endpoints"""
    
    for endpoint in ENDPOINTS:
        try:
            payload = {
                endpoint["json_key"]: prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            r = requests.post(
                endpoint["url"],
                headers=endpoint["headers"],
                json=payload,
                timeout=15
            )
            
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    return data[0].get('generated_text', '')
                elif isinstance(data, dict):
                    return data.get('generated_text', '')
            
            # Don't print error for every attempt, just try next endpoint
        except:
            pass
    
    print("   All AI endpoints failed")
    return None
