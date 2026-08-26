"""
Senate AI - Get Overfit Bundle
Downloads the bundle for a specific overfit senator index.
"""

import json
import subprocess
import sys
from pathlib import Path


def get_bundle(index):
    """Download the bundle for the overfit senator at the given index"""
    
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
    bundle_id = senator_data['bundle_id']
    bundle_file = f'senate_bundles/bundle_{bundle_id:03d}.pt'
    
    result = subprocess.run(
        ['git', 'sparse-checkout', 'add', bundle_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f'Downloaded {bundle_file}')
    else:
        print(f'Failed: {result.stderr[:200]}')


if __name__ == '__main__':
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    get_bundle(index)
