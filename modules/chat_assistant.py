# modules/chat_assistant.py
from modules.script_generator import generate_script_from_brief
from modules.title_suggester import suggest_titles_from_brief

# --- Chat Asistanı: Brief Toplama Akışı ---
def collect_brief(
    topic=None,
    audience=None,
    duration=None,
    style=None,
    intro_strategy=None,
    example_formula=None
):
    """
    Kullanıcıdan adım adım brief toplar ve yapılandırılmış bir brief stringi döndürür.
    """
    brief_parts = []
    if topic:
        brief_parts.append(f"Konu: {topic}")
    if audience:
        brief_parts.append(f"Hedef kitle: {audience}")
    if duration:
        brief_parts.append(f"Hedef süre: {duration}")
    if style:
        brief_parts.append(f"Stil tercihi: {style}")
    if intro_strategy:
        brief_parts.append(f"Giriş stratejisi: {intro_strategy}")
    if example_formula:
        brief_parts.append(f"Örnek video formülü: {example_formula}")
    return "\n".join(brief_parts)

# --- Chat Asistanı: Script ve Başlık Üretimi ---
def chat_assistant_generate_all(brief):
    """
    Brief'e göre başlık ve script üretir.
    """
    titles = suggest_titles_from_brief(brief, n=3)
    script = generate_script_from_brief(brief)
    return {
        "brief": brief,
        "title_suggestions": titles,
        "script": script
    }

def refine_script_and_titles_with_feedback(brief, current_script, current_titles, feedback, max_tokens=2000):
    """
    Kullanıcı geri bildirimiyle script ve başlıkları GPT-4o ile günceller.
    """
    import os
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
Aşağıda bir YouTube videosu için oluşturulmuş script, başlık önerileri ve kullanıcıdan gelen geri bildirim var. Lütfen bu geri bildirimi dikkate alarak scripti ve başlıkları güncelle. Script ve başlıklar Türkçe olmalı.

---
Kullanıcı Brief'i:
{brief}

Mevcut Script:
{current_script}

Mevcut Başlıklar:
{chr(10).join(current_titles)}

Kullanıcı Geri Bildirimi:
{feedback}

---
Güncellenmiş Script ve 3 yeni başlık önerisi üret. Yanıtın şu formatta olsun:

SCRIPT:
<güncellenmiş script>

TITLES:
- <başlık 1>
- <başlık 2>
- <başlık 3>
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content.strip()
        # Parse response
        script = ""
        titles = []
        if "SCRIPT:" in content and "TITLES:" in content:
            script_part = content.split("SCRIPT:",1)[1].split("TITLES:",1)[0].strip()
            titles_part = content.split("TITLES:",1)[1].strip()
            script = script_part
            titles = [t.strip('-•* ') for t in titles_part.split('\n') if t.strip()]
        else:
            script = content
            titles = []
        return {"script": script, "title_suggestions": titles}
    except Exception as e:
        return {"script": f"[OpenAI hata]: {e}", "title_suggestions": [f"[OpenAI hata]: {e}"]} 