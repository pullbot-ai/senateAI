"""
Senate AI - AI Client
Uses deployed Puter Worker for AI calls.
"""

import requests
import os

PUTER_WORKER_URL = os.environ.get('PUTER_WORKER_URL', 'https://senate-ai-worker.puter.site')

def call_ai(prompt, max_tokens=500, purpose='training', **kwargs):
    """Call Puter worker for AI tasks"""
    
    try:
        if purpose == 'training':
            response = requests.post(
                f'{PUTER_WORKER_URL}/generate-training-data',
                json={
                    'topic': kwargs.get('topic', 'general'),
                    'num_examples': kwargs.get('num_examples', 30)
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('data', '')
        
        elif purpose == 'grading':
            response = requests.post(
                f'{PUTER_WORKER_URL}/grade-answer',
                json={
                    'question': kwargs.get('question', ''),
                    'correct_answer': kwargs.get('correct_answer', ''),
                    'senator_answer': kwargs.get('senator_answer', '')
                },
                timeout=20
            )
            if response.status_code == 200:
                return response.json().get('score', '')
        
        elif purpose == 'params':
            response = requests.post(
                f'{PUTER_WORKER_URL}/select-params',
                json={
                    'topics': kwargs.get('topics', []),
                    'epoch': kwargs.get('epoch', 0),
                    'loss': kwargs.get('loss', 8.0),
                    'param_names': kwargs.get('param_names', [])
                },
                timeout=20
            )
            if response.status_code == 200:
                return response.json().get('params', '')
    
    except Exception as e:
        print(f'   Worker call failed: {e}')
    
    return None
