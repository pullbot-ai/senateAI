"""
Senate AI - Punctuator
Adds punctuation to senator outputs using AI.
"""

from ai_client import call_ai
import re

def punctuate(text):
    """Add punctuation and clean up senator output"""
    
    if not text or len(text) < 2:
        return text
    
    prompt = f"""Add proper punctuation to this AI response:
"{text}"

Return the punctuated version only."""
    
    response = call_ai(prompt, max_tokens=len(text) + 50)
    
    if response and len(response) > 2:
        return response.strip()
    
    # Fallback: basic cleanup
    text = text.strip()
    if text and not text[-1] in '.!?':
        text += '.'
    text = text[0].upper() + text[1:] if text else text
    
    return text
