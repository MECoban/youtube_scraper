# modules/intro_analyzer.py
import os
import openai
import re
from modules.prompt_templates import INTRO_ANALYSIS_PROMPT

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Klasik Intro Analizi (örnek: kelime frekansı) ---
def classic_intro_analysis(intro_transcript):
    words = re.findall(r'\w+', intro_transcript.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    return dict(top_words)

# --- GPT Tabanlı Intro Analizi ---
def gpt_intro_analysis(intro_transcript, max_tokens=400):
    prompt = INTRO_ANALYSIS_PROMPT.format(intro_transcript=intro_transcript)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI hata]: {e}"

# --- Ana Fonksiyon ---
def analyze_intro(intro_transcript):
    """
    Hem klasik hem GPT tabanlı intro analizini döndürür.
    """
    classic = classic_intro_analysis(intro_transcript)
    gpt = gpt_intro_analysis(intro_transcript)
    return {
        "classic": classic,
        "gpt_analysis": gpt
    } 