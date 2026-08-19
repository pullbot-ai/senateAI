/**
 * Senate AI Backend
 * Runs in GitHub Actions. Handles questions, triggers debates, stores results.
 */

const fs = require('fs');

const REPO = 'pullbot-ai/senateAI';
const WORKFLOW_ID = 'senate_debate.yml';
const MAX_CONCURRENT = 3;

const QUESTIONS_FILE = 'questions.json';
const RESULTS_FILE = 'debate_results.json';

function loadJSON(filepath, defaultData = {}) {
    try {
        if (fs.existsSync(filepath)) {
            return JSON.parse(fs.readFileSync(filepath, 'utf8'));
        }
    } catch (e) {
        console.log(`Error loading ${filepath}: ${e.message}`);
    }
    return defaultData;
}

function saveJSON(filepath, data) {
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
}

async function checkRunningDebates() {
    try {
        const response = await fetch(
            `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_ID}/runs?status=in_progress`,
            {
                headers: {
                    'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`,
                    'Accept': 'application/vnd.github.v3+json'
                }
            }
        );
        
        if (response.ok) {
            const data = await response.json();
            return data.workflow_runs ? data.workflow_runs.length : 0;
        }
    } catch (e) {
        console.log(`Error checking running debates: ${e.message}`);
    }
    return 0;
}

async function triggerDebate(question) {
    try {
        const response = await fetch(
            `https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_ID}/dispatches`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`,
                    'Accept': 'application/vnd.github.v3+json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ref: 'main',
                    inputs: { question }
                })
            }
        );
        
        return response.status === 204;
    } catch (e) {
        console.log(`Error triggering debate: ${e.message}`);
        return false;
    }
}

async function processQuestions() {
    console.log('Processing questions...');
    
    const questions = loadJSON(QUESTIONS_FILE, { questions: [] });
    const running = await checkRunningDebates();
    
    console.log(`Running debates: ${running}/${MAX_CONCURRENT}`);
    
    let changed = false;
    
    for (const q of questions.questions || []) {
        if (q.status === 'pending' && running < MAX_CONCURRENT) {
            console.log(`Triggering: "${q.question.substring(0, 60)}..."`);
            
            const success = await triggerDebate(q.question);
            
            if (success) {
                q.status = 'debating';
                q.started_at = new Date().toISOString();
                running++;
                changed = true;
                console.log('Debate triggered');
            } else {
                q.status = 'failed';
                q.error = 'Failed to trigger workflow';
                changed = true;
            }
        }
    }
    
    if (changed) {
        saveJSON(QUESTIONS_FILE, questions);
    }
}

async function main() {
    console.log('='.repeat(50));
    console.log('Senate AI Backend');
    console.log('='.repeat(50));
    console.log('Time:', new Date().toISOString());
    
    await processQuestions();
    
    console.log('Backend complete.');
    console.log('='.repeat(50));
}

main().catch(e => {
    console.error('Backend error:', e);
    process.exit(1);
});
