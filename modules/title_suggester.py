# modules/title_suggester.py
import os
import openai
from modules.prompt_templates import TITLE_SUGGESTION_PROMPT

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Analiz Tabanlı Başlık Önerisi ---
def suggest_titles(summary, topics, sentiment, trends, channel_name="Fatih Çoban", n=3, max_tokens=200):
    prompt = TITLE_SUGGESTION_PROMPT.format(
        channel_name=channel_name,
        summary=summary,
        topics=topics,
        sentiment=sentiment,
        trends=trends
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        # Satır satır başlıkları ayıkla
        content = response.choices[0].message.content.strip()
        titles = [t.strip('-•* ') for t in content.split('\n') if t.strip()]
        return titles[:n]
    except Exception as e:
        return [f"[OpenAI hata]: {e}"]

# --- Brief Tabanlı Başlık Önerisi (Chat Asistanı için) ---
def suggest_titles_from_brief(brief, n=3, max_tokens=200):
    prompt = f"""
Aşağıda bir YouTube videosu için kullanıcıdan alınan brief var. Bu brief'e uygun, değer temelli ve ilgi çekici 3 Türkçe başlık öner:

{brief}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        titles = [t.strip('-•* ') for t in content.split('\n') if t.strip()]
        return titles[:n]
    except Exception as e:
        return [f"[OpenAI hata]: {e}"] 