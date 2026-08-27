"""
Senate AI - AI Edit Senator
AI directly modifies weights of overfit senators to reduce memorization.
"""

import sys
import json
import torch
import re
from pathlib import Path

sys.path.insert(0, 'src')
from model_template import SenateBundle
from ai_client import call_ai
from wordbank import get_wordbank


def ai_direct_edit(senator, overfit_severity, specialties):
    """AI directly modifies weights to fix overfitting"""
    
    param_names = list(dict(senator.named_parameters()).keys())
    
    prompt = f"""Edit a neural network to fix overfitting.

Specialties: {', '.join(specialties[:3])}
Severity: {overfit_severity:.1f} percent

Available params: {', '.join(param_names[:8])}

For each param, say ADD_NOISE or REDUCE and a number from 0.05 to 0.3.

Return JSON array with param, action, and amount fields."""
    
    print('Asking AI to edit weights...')
    
    for attempt in range(3):
        response = call_ai(prompt, max_tokens=150)
        
        if response:
            print(f'AI response received (attempt {attempt+1})')
            try:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    edits = json.loads(match.group())
                    if isinstance(edits, list):
                        for edit in edits:
                            param_hint = edit.get('param', '')
                            action = edit.get('action', edit.get('operation', 'ADD_NOISE'))
                            amount = float(edit.get('amount', edit.get('magnitude', 0.1)))
                            
                            for name, param in senator.named_parameters():
                                if param_hint.lower() in name.lower():
                                    with torch.no_grad():
                                        if 'NOISE' in action.upper():
                                            std = param.data.std()
                                            if std > 0:
                                                param.data += torch.randn_like(param.data) * amount * std
                                            print(f'Added noise to {name}')
                                        elif 'REDUCE' in action.upper():
                                            param.data *= (1 - amount)
                                            print(f'Reduced {name}')
                                    break
                        return True
            except Exception as e:
                print(f'Parse error: {e}')
        else:
            print(f'AI returned None (attempt {attempt+1})')
    
    return False


def edit_senator(index):
    """Edit the senator at the given index"""
    
    overfit_file = Path('overfit_senators.json')
    if not overfit_file.exists():
        print("overfit_senators.json not found")
        sys.exit(0)
    
    with open(overfit_file) as f:
        overfit = json.load(f)
    
    if index >= len(overfit):
        print("No more senators")
        sys.exit(0)
    
    senator_data = overfit[index]
    senator_id = senator_data['senator_id']
    bundle_id = senator_data['bundle_id']
    specialties = senator_data['specialties']
    overfit_score = senator_data['score']
    
    bundle_path = f'senate_bundles/bundle_{bundle_id:03d}.pt'
    bundle = SenateBundle.load(bundle_path)
    senator = bundle.get_senator(senator_id)
    
    if senator is None:
        print(f'Senator {senator_id} not found')
        sys.exit(1)
    
    wordbank = get_wordbank()
    
    print(f'AI editing Senator {senator_id}')
    print(f'Overfit score: {overfit_score:.1f}%')
    
    edited = ai_direct_edit(senator, overfit_score, specialties)
    
    if not edited:
        print('AI edit failed, applying weight clipping fallback...')
        with torch.no_grad():
            for name, param in senator.named_parameters():
                if param.dim() >= 2:
                    mean = param.data.mean()
                    std = param.data.std()
                    if std > 0:
                        mask = param.data.abs() > (mean.abs() + 2 * std)
                        param.data[mask] = mean
                        clipped = mask.sum().item()
                        if clipped > 0:
                            print(f'Clipped {clipped} extreme weights in {name}')
        print('Applied weight clipping')
    
    from train import grade_and_update
    print('Re-grading...')
    result = grade_and_update(senator, specialties, wordbank)
    
    if result:
        new_score = result['average_score']
        print(f'Previous: {overfit_score:.1f}%')
        print(f'After: {new_score:.1f}%')
        
        if new_score > overfit_score:
            print(f'Improvement: +{new_score - overfit_score:.1f}%')
        else:
            print('No improvement')
    
    bundle.save(bundle_path)
    print(f'Saved bundle {bundle_id}')


if __name__ == '__main__':
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    edit_senator(index)
