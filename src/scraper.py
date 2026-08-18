"""
Senate AI - Wikipedia Word Extractor + Auto-Definer
Scrapes Wikipedia articles, extracts words, defines new ones.
Builds the shared wordbank for all senators.
"""

import os
import sys
import json
import time
import re
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

WORD_BANK_PATH = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
DEFINITIONS_PATH = os.path.join(REPO_ROOT, 'data', 'definitions.json')


def load_wordbank():
    if os.path.exists(WORD_BANK_PATH):
        try:
            with open(WORD_BANK_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("   Wordbank corrupted, backing up and starting fresh")
            backup = WORD_BANK_PATH + '.corrupted'
            try:
                os.rename(WORD_BANK_PATH, backup)
            except:
                pass
            return {"words": {}, "total_articles": 0, "total_words": 0}
    return {"words": {}, "total_articles": 0, "total_words": 0}


def save_wordbank(bank):
    os.makedirs(os.path.dirname(WORD_BANK_PATH), exist_ok=True)
    with open(WORD_BANK_PATH, 'w') as f:
        json.dump(bank, f, indent=2)


def load_definitions():
    if os.path.exists(DEFINITIONS_PATH):
        try:
            with open(DEFINITIONS_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print("   Definitions corrupted, starting fresh")
            return []
    return []


def save_definitions(defs):
    os.makedirs(os.path.dirname(DEFINITIONS_PATH), exist_ok=True)
    with open(DEFINITIONS_PATH, 'w') as f:
        json.dump(defs, f, indent=2)


def extract_words(text):
    """Extract clean English words from text"""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = []
    for w in text.split():
        w = w.lower().strip()
        if len(w) > 2 and w.isalpha():
            words.append(w)
    return list(set(words))


def scrape_article():
    """Scrape one random Wikipedia article"""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/random/summary",
            timeout=15,
            headers={'User-Agent': 'SenateAI/1.0 (https://github.com/pullbot-ai/senateAI)'}
        )
        if r.status_code == 200:
            data = r.json()
            return data.get('extract', ''), data.get('title', 'unknown')
    except:
        pass
    return "", ""


def lookup_definition(word):
    """Look up definition for a single word"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for entry in data[:1]:
                for meaning in entry.get('meanings', [])[:1]:
                    for d in meaning.get('definitions', [])[:1]:
                        return d.get('definition', '')
    except:
        pass
    return None


def get_token_id(word, vocab_size=8000):
    """Map a word to a token ID using stable hash"""
    import hashlib
    hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
    return 3 + (hash_val % (vocab_size - 3))


def process_article(text, title, bank, definitions, article_num):
    """Process one article: extract words, add new ones, define them"""
    
    all_words = extract_words(text)
    
    new_words = []
    skipped = 0
    
    for word in all_words:
        if word not in bank['words']:
            token_id = get_token_id(word)
            bank['words'][word] = {
                'token_id': token_id,
                'first_seen': title,
                'has_definition': False,
                'definition': ''
            }
            new_words.append(word)
        else:
            skipped += 1
    
    defined_count = 0
    for word in new_words[:20]:
        definition = lookup_definition(word)
        if definition:
            bank['words'][word]['has_definition'] = True
            bank['words'][word]['definition'] = definition
            definitions.append({
                'word': word,
                'token_id': bank['words'][word]['token_id'],
                'definition': definition,
                'first_seen': title
            })
            defined_count += 1
        time.sleep(0.2)
    
    return new_words, skipped, defined_count


def run_mass_scrape(num_articles=30):
    """Scrape articles, extract words, define new ones"""
    print("=" * 50)
    print(f"SENATE AI - WORD SCRAPER")
    print(f"   Target: {num_articles} articles")
    print("=" * 50)
    
    bank = load_wordbank()
    definitions = load_definitions()
    
    starting_words = len(bank['words'])
    total_new = 0
    total_skipped = 0
    total_defined = 0
    errors = 0
    
    for i in range(num_articles):
        text, title = scrape_article()
        
        if not text:
            errors += 1
            continue
        
        new_words, skipped, defined = process_article(text, title, bank, definitions, i)
        
        total_new += len(new_words)
        total_skipped += skipped
        total_defined += defined
        bank['total_articles'] += 1
        
        if (i + 1) % 5 == 0:
            total = len(bank['words'])
            print(f"   {i+1}/{num_articles} | {total:,} words | +{total_new} new | {total_defined} defined | {errors} err")
        
        time.sleep(0.3)
    
    bank['total_words'] = len(bank['words'])
    save_wordbank(bank)
    save_definitions(definitions)
    
    print(f"\nDone!")
    print(f"   Articles: {num_articles}")
    print(f"   Words before: {starting_words:,}")
    print(f"   Words after: {len(bank['words']):,}")
    print(f"   New words added: {total_new:,}")
    print(f"   Words already known: {total_skipped:,}")
    print(f"   New definitions: {total_defined:,}")
    print(f"   Errors: {errors}")
    
    all_words = list(bank['words'].keys())
    if all_words:
        print(f"\n   Latest words: {all_words[-20:]}")


if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_mass_scrape(num)
