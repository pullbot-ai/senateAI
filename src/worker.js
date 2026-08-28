/**
 * Senate AI - Puter Worker
 * Handles AI calls for training, grading, and parameter selection.
 */

export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        
        if (url.pathname === '/generate-training-data') {
            const { topic, num_examples } = await request.json();
            
            const prompt = `Generate ${num_examples} diverse training sentences for a tiny AI specializing in '${topic}'.
Each sentence MUST end with punctuation.
Return as JSON array of strings.`;
            
            const response = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
                prompt: prompt,
                max_tokens: 800
            });
            
            return new Response(JSON.stringify({ data: response.response }), {
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        if (url.pathname === '/grade-answer') {
            const { question, correct_answer, senator_answer } = await request.json();
            
            const prompt = `Grade this answer STRICTLY from 0-100.
Question: ${question}
Correct: ${correct_answer}
Senator: ${senator_answer}
Return ONLY a number.`;
            
            const response = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
                prompt: prompt,
                max_tokens: 10
            });
            
            return new Response(JSON.stringify({ score: response.response }), {
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        if (url.pathname === '/select-params') {
            const { topics, epoch, loss, param_names } = await request.json();
            
            const prompt = `Pick 5 parameter groups to update.
Topic: ${topics.join(', ')}
Epoch: ${epoch}
Loss: ${loss}
Params: ${param_names.join(', ')}
Return JSON array.`;
            
            const response = await env.AI.run('@cf/meta/llama-3.1-8b-instruct', {
                prompt: prompt,
                max_tokens: 100
            });
            
            return new Response(JSON.stringify({ params: response.response }), {
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        return new Response('Not found', { status: 404 });
    }
};
