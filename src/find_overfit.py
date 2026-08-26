"""
Senate AI - Find Overfit Senators
Identifies senators with grades 10-25% that are likely overfitting.
"""

import json
import torch
import sys
from pathlib import Path


def find_overfit(max_senators=10):
    """Find senators that are likely overfitting (trained, grade 10-25%)"""
    
    overfit = []
    
    for bundle_file in sorted(Path('senate_bundles').glob('bundle_*.pt')):
        if bundle_file.stem == 'senate_index':
            continue
        
        try:
            bundle_id = int(bundle_file.stem.split('_')[1])
            bundle = torch.load(bundle_file, map_location='cpu', weights_only=False)
            senators = bundle.get('senators', {})
            
            for senator_id_str, data in senators.items():
                try:
                    senator_id = int(senator_id_str)
                except:
                    senator_id = int(data.get('config', {}).get('model_id', 0))
                
                config = data.get('config', {})
                performance = config.get('performance', {})
                specialties = config.get('specialties', [])
                
                if performance:
                    avg = sum(performance.values()) / len(performance) * 100
                else:
                    avg = 0
                
                is_trained = abs(avg - 50.0) >= 0.01
                
                if is_trained and 10 <= avg <= 25:
                    overfit.append({
                        'senator_id': senator_id,
                        'bundle_id': bundle_id,
                        'specialties': specialties,
                        'score': avg
                    })
        except Exception as e:
            pass
    
    overfit.sort(key=lambda x: x['score'])
    
    top_overfit = overfit[:max_senators]
    
    with open('overfit_senators.json', 'w') as f:
        json.dump(top_overfit, f)
    
    print(f'Found {len(overfit)} overfit senators (grade 10-25%)')
    print(f'Selected {len(top_overfit)} for AI editing:')
    for w in top_overfit[:15]:
        specs = ', '.join(w['specialties'][:2]) if w['specialties'] else 'none'
        print(f'  Senator {w["senator_id"]} (Bundle {w["bundle_id"]}): {w["score"]:.1f}% - {specs}')
    
    return top_overfit


if __name__ == '__main__':
    max_senators = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    find_overfit(max_senators)
