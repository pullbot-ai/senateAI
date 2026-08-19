"""
Senate AI Backend - Runs on Render
Downloads only router + selected senators. Never the whole repo.
"""

from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import sys
import time
import json
import torch
from pathlib import Path

app = Flask(__name__, static_folder='.', static_url_path='')

PAT = os.environ.get('GITHUB_PAT', '')
REPO = 'pullbot-ai/senateAI'
API_BASE = f'https://api.github.com/repos/{REPO}/contents'

# Cache directory
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

# Downloaded files
ROUTER_PATH = CACHE_DIR / 'router.pt'
INDEX_PATH = CACHE_DIR / 'senate_index.json'
WORDBANK_PATH = CACHE_DIR / 'wordbank.json'
CONFIG_PATH = CACHE_DIR / 'config.yaml'

def download_file(relative_path, output_path):
    """Download a single file from GitHub repo without cloning"""
    url = f'{API_BASE}/{relative_path}'
    headers = {'Authorization': f'Bearer {PAT}', 'Accept': 'application/vnd.github.v3+json'}
    
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 200:
        import base64
        content = base64.b64decode(r.json()['content'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        return True
    return False

def download_bundle(bundle_id):
    """Download one senator bundle"""
    bundle_file = f'senate_bundles/bundle_{bundle_id:03d}.pt'
    bundle_path = CACHE_DIR / bundle_file
    
    if bundle_path.exists():
        return bundle_path
    
    # Try raw.githubusercontent.com for large files
    raw_url = f'https://raw.githubusercontent.com/{REPO}/main/{bundle_file}'
    headers = {'Authorization': f'Bearer {PAT}'}
    
    r = requests.get(raw_url, headers=headers, timeout=60)
    if r.status_code == 200:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(r.content)
        return bundle_path
    
    return None

def ensure_essential_files():
    """Download only essential files"""
    if not ROUTER_PATH.exists():
        print("Downloading router...")
        download_file('models/router.pt', ROUTER_PATH)
    
    if not INDEX_PATH.exists():
        print("Downloading senate index...")
        download_file('senate_bundles/senate_index.json', INDEX_PATH)
    
    if not WORDBANK_PATH.exists():
        print("Downloading wordbank...")
        download_file('data/wordbank.json', WORDBANK_PATH)
    
    if not CONFIG_PATH.exists():
        print("Downloading config...")
        download_file('config.yaml', CONFIG_PATH)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chatbot.html')
def chatbot():
    return send_from_directory('.', 'chatbot.html')

@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory('assets', filename)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'router_downloaded': ROUTER_PATH.exists(),
        'index_downloaded': INDEX_PATH.exists(),
        'wordbank_downloaded': WORDBANK_PATH.exists()
    })

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question required'}), 400
    
    # Ensure essential files are downloaded
    ensure_essential_files()
    
    # Add cache to sys.path
    sys.path.insert(0, str(CACHE_DIR))
    sys.path.insert(0, 'src')
    
    try:
        # Run the Senate debate
        from senate import Senate
        senate = Senate()
        result = senate.ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_status')
def download_status():
    """Check what's been downloaded"""
    return jsonify({
        'router': ROUTER_PATH.exists(),
        'index': INDEX_PATH.exists(),
        'wordbank': WORDBANK_PATH.exists(),
        'config': CONFIG_PATH.exists(),
        'cache_size_mb': sum(f.stat().st_size for f in CACHE_DIR.glob('**/*') if f.is_file()) / (1024*1024)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
