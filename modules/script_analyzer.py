# modules/script_analyzer.py
import os
import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
from modules.prompt_templates import VIDEO_STRATEGY_ANALYSIS_PROMPT

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Klasik Analiz Fonksiyonları ---
def classic_analysis(transcript, n_topics=5):
    """
    Basit TF-IDF, kelime frekansı, duygu analizi gibi klasik analizler döndürür.
    """
    # Temizle
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    cleaned = clean_text(transcript)
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('turkish') + stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    top_indices = tfidf_matrix.toarray()[0].argsort()[-n_topics:][::-1]
    topics = [feature_names[idx] for idx in top_indices]
    # Kelime frekansı
    words = [w for w in cleaned.split() if w not in stopwords.words('turkish') + stopwords.words('english')]
    freq = pd.Series(words).value_counts().head(10).to_dict()
    # (Basit) Duygu analizi placeholder
    sentiment = "N/A"  # Buraya Türkçe sentiment modeli entegre edilebilir
    return {
        "topics": topics,
        "word_frequencies": freq,
        "sentiment": sentiment
    }

# --- GPT Tabanlı Strateji Analizi ---
def gpt_strategy_analysis(title, description, transcript, konu_listesi, strateji_listesi, max_tokens=900):
    prompt = VIDEO_STRATEGY_ANALYSIS_PROMPT.format(
        KONU_LISTESI="\n- " + "\n- ".join(konu_listesi),
        STRATEJI_LISTESI="\n- " + "\n- ".join(strateji_listesi),
        title=title,
        description=description,
        transcript=transcript[:3500]
    )
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

# --- Ana Analiz Fonksiyonu ---
def analyze_video(title, description, transcript, konu_listesi, strateji_listesi):
    """
    Hem klasik hem GPT tabanlı analizleri birleştirir.
    """
    classic = classic_analysis(transcript)
    gpt = gpt_strategy_analysis(title, description, transcript, konu_listesi, strateji_listesi)
    return {
        "classic": classic,
        "gpt_analysis": gpt
    } 