#analyse.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import re
from collections import Counter
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from youtube_transcript_api import YouTubeTranscriptApi
import time
from trnlp import TrnlpWord
from sklearn.metrics.pairwise import cosine_similarity
from pytrends.request import TrendReq
import json
import os
from datetime import datetime
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import ast
from dotenv import load_dotenv
import openai
import numpy as np
from modules.script_analyzer import analyze_video
from modules.intro_analyzer import analyze_intro
from modules.script_generator import generate_script_from_analysis
from modules.title_suggester import suggest_titles

load_dotenv()
JSON_KEY_FILE = os.getenv("JSON_KEY_FILE")
GOOGLE_CREDENTIAL_PATH = os.getenv("GOOGLE_CREDENTIAL_PATH")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise EnvironmentError("API_KEY not found in environment variables.")

# Initialize OpenAI client (needed for the updated get_summary_openai)
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not openai_client.api_key:
     raise EnvironmentError("OPENAI_API_KEY not found or invalid. Please set it in your .env file.")

# Türkçe ve İngilizce stopword listesi
STOPWORDS = set([
    'the', 'and', 'to', 'of', 'in', 'a', 'for', 'on', 'with', 'is', 'how', 'why', 'what', 'that', 'this', 'it', 'at', 'by', 'from', 'as', 'an', 'be', 'are', 'your', 'my', 'you', 'we', 'our',
    've', 'ile', 'bir', 'için', 'en', 'ne', 'nasıl', 'neden', 'bu', 'o', 'da', 'de', 'ki', 'mi', 'ben', 'sen', 'biz', 'siz', 'onlar'
])

# --- Konu ve Strateji Listeleri (örnek) ---
KONU_LISTESI = [
    "Kişisel Başarı Alışkanlıkları",
    "Para Kazanma Stratejileri ve Yol Haritaları",
    "Girişimcilik Sırları ve Dersleri",
    "Finansal ve İş Gelişim Taktikleri",
    "Öğrenme ve Gelişim",
    "İlham Veren Başarı Hikayeleri ve Röportajlar",
    "Sistem Kurma ve İşletme Taktikleri",
    "Finansal Özgürlük ve Varlık Yönetimi",
    "Lokasyondan Bağımsızlık ve Seyahat",
    "Kişisel Huzur ve İş-Yaşam Dengesi"
]
STRATEJI_LISTESI = [
    "Listicle (madde madde anlatım)",
    "How-to (adım adım öğretici)",
    "Hikaye anlatımı",
    "Challenge (meydan okuma)",
    "Röportaj",
    "Testimonial (başarı hikayesi)",
    "Case study (örnek olay)",
    "Soru-cevap",
    "Problem/Çözüm odaklı",
    "Fayda/Değer vurgusu",
    "Sosyal kanıt",
    "Curiosity gap (merak boşluğu)",
    "Meydan okuma",
    "Şok edici bilgi",
    "Vaad/teklif (offer)",
    "Hook (kanca)",
    "CTA (çağrı)"
]

def clean_title(title):
    # Sadece harf ve rakamları bırak, küçük harfe çevir
    return re.sub(r'[^\w\s]', '', title).lower()

def extract_keywords_from_text(text, top_n=10):
    words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in STOPWORDS]
    freq_dist = FreqDist(words)
    return [word for word, _ in freq_dist.most_common(top_n)]

def extract_keywords_from_titles(titles):
    words = []
    for title in titles:
        cleaned = clean_title(title)
        words += [w for w in cleaned.split() if w not in STOPWORDS]
    return Counter(words).most_common(30)

def extract_patterns(titles):
    # Basit kalıp analizi: başlıkların ilk 2-3 kelimesi
    patterns = []
    for title in titles:
        cleaned = clean_title(title)
        tokens = cleaned.split()
        if len(tokens) >= 2:
            patterns.append(' '.join(tokens[:2]))
        if len(tokens) >= 3:
            patterns.append(' '.join(tokens[:3]))
    return Counter(patterns).most_common(15)

def get_titles_and_descriptions_by_type(sheet_name, worksheet_name):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIAL_PATH or JSON_KEY_FILE, scope)
    client = gspread.authorize(creds)
    worksheet = client.open(sheet_name).worksheet(worksheet_name)
    data = worksheet.get_all_records()
    time.sleep(2)  # Sheets okuma kotası için bekle
    top_titles = [row['title'] for row in data if row.get('source_type') == 'top' and row.get('title')]
    latest_titles = [row['title'] for row in data if row.get('source_type') == 'latest' and row.get('title')]
    top_descs = [row.get('description', '') for row in data if row.get('source_type') == 'top']
    latest_descs = [row.get('description', '') for row in data if row.get('source_type') == 'latest']
    return top_titles, latest_titles, top_descs, latest_descs

# --- Başlık örüntüsü ve format analizi fonksiyonları ---
def detect_listicle(title):
    return bool(re.match(r'^\d+[\s\w]*\b(to|ways?|habits?|things?|reasons?|steps?)\b', title, re.IGNORECASE))

def detect_emotional_trigger(title):
    triggers = ['why', 'still', 'never', 'always', 'afraid', 'fail', 'mistake', 'secret', 'truth', 'asla', 'neden', 'hala', 'korku', 'başarısız', 'sır', 'gerçek']
    return any(word in title.lower() for word in triggers)

def detect_contrast(title):
    contrast_words = ['but', 'not', "isn't", "doesn't", 'değil', 'ama', 'fakat', 'aslında', 'zannediyorsun']
    return any(word in title.lower() for word in contrast_words)

def detect_format(title, description=''):
    guest_keywords = ['with', 'konuk', 'interview', 'röportaj']
    qa_keywords = ['q&a', 'soru', 'cevap', 'sordum']
    story_keywords = ['story', 'hikaye', 'how i', 'my journey', 'anlatıyorum']
    text = (title + ' ' + description).lower()
    if any(word in text for word in guest_keywords):
        return 'guest'
    if any(word in text for word in qa_keywords):
        return 'q&a'
    if any(word in text for word in story_keywords):
        return 'storytelling'
    return 'other'

def analyze_patterns(titles, descriptions=None):
    results = {
        'listicle': 0,
        'emotional': 0,
        'contrast': 0,
        'guest': 0,
        'q&a': 0,
        'storytelling': 0,
        'other': 0
    }
    for i, title in enumerate(titles):
        desc = descriptions[i] if descriptions and i < len(descriptions) else ''
        if detect_listicle(title):
            results['listicle'] += 1
        if detect_emotional_trigger(title):
            results['emotional'] += 1
        if detect_contrast(title):
            results['contrast'] += 1
        fmt = detect_format(title, desc)
        results[fmt] += 1
    return results

def get_title_structure(title, description=''):
    title_lower = title.lower()
    if detect_listicle(title):
        return 'listicle'
    if '?' in title or title_lower.startswith(('how', 'why', 'what', 'when', 'where', 'who', 'which', 'nasıl', 'neden', 'ne', 'kim', 'nerede', 'hangi')):
        return 'question'
    if detect_contrast(title):
        return 'contrast'
    fmt = detect_format(title, description)
    if fmt in ['guest', 'q&a', 'storytelling']:
        return fmt
    return 'other'

def generalize_title(title):
    # Sayıları ve para birimlerini X/Y ile değiştir
    title = re.sub(r'\$?\d+[\d,\.]*', 'X', title)
    # Yılları Y ile değiştir
    title = re.sub(r'20\d{2}|19\d{2}', 'Y', title)
    # Sık geçen kelimeleri değişkenleştir
    title = re.sub(r'\b(days?|weeks?|months?|years?)\b', 'T', title, flags=re.IGNORECASE)
    # Parantez içini (Z) ile değiştir
    title = re.sub(r'\(.*?\)', '(Z)', title)
    # Kişi isimlerini ve özel isimleri X ile değiştir (basit örnek)
    title = re.sub(r'\b[A-Z][a-z]+\b', 'X', title)
    # Fazla boşlukları temizle
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def extract_title_formulas(titles):
    formulas = [generalize_title(t) for t in titles]
    return Counter(formulas).most_common(5)

def classify_title_type(title):
    title_lower = title.lower()
    if re.match(r'^(how to|nasıl)', title_lower):
        return 'howto'
    if re.match(r'^\d+ (ways?|habits?|things?|reasons?|steps?)', title_lower):
        return 'listicle'
    if re.match(r'^(why|what|when|where|who|which|neden|ne|ne zaman|nerede|kim|hangi)', title_lower) or '?' in title:
        return 'question'
    if re.search(r'(secret|truth|sır|gerçek)', title_lower):
        return 'secret/truth'
    if re.search(r'(my story|how i|the day i|benim hikayem|nasıl ... yaptım)', title_lower):
        return 'story/personal'
    if re.search(r'( vs | or | against | vs\.|veya|karşı)', title_lower):
        return 'comparison'
    if re.search(r'(interview|with|konuk|röportaj)', title_lower):
        return 'guest/interview'
    if re.search(r'(q&a|soru-cevap|soru cevap)', title_lower):
        return 'q&a'
    if re.search(r'(mistakes|never|don\'t|avoid|hata|asla|kaçın|yapmayın)', title_lower):
        return 'warning/negative'
    if re.search(r'(but|not|değil|ama|fakat|aslında|zannediyorsun)', title_lower):
        return 'contrast'
    return 'other'

def summarize_title_structures(structure_rows):
    summary = []
    header = [
        "Kanal", "Analiz Türü", "Başlık Tipi Dağılımı",
        "En Sık Başlık Başlangıçları (ilk 2 kelime)",
        "En Sık Başlık Başlangıçları (ilk 3 kelime)",
        "En Sık Başlık Başlangıçları (ilk 4 kelime)",
        "En Sık Başlık Formülleri"
    ]
    summary.append(header)
    from itertools import groupby
    rows = structure_rows[1:]  # başlık satırı hariç
    rows.sort(key=lambda x: (x[0], x[1]))  # Kanal, Analiz Türü
    for (kanal, analiz_turu), group in groupby(rows, key=lambda x: (x[0], x[1])):
        group = list(group)
        # Yeni başlık tipi dağılımı
        types = [classify_title_type(row[2]) for row in group]
        type_counts = Counter(types)
        total = sum(type_counts.values())
        type_dist = ', '.join([f'{k}: {v} ({v*100//total}%)' for k, v in type_counts.most_common()])
        # Başlık başlangıçları
        starts2 = []
        starts3 = []
        starts4 = []
        for row in group:
            tokens = row[2].split()
            if len(tokens) >= 2:
                starts2.append(' '.join(tokens[:2]))
            if len(tokens) >= 3:
                starts3.append(' '.join(tokens[:3]))
            if len(tokens) >= 4:
                starts4.append(' '.join(tokens[:4]))
        top2 = ', '.join([f'"{k}" ({v})' for k, v in Counter(starts2).most_common(3)])
        top3 = ', '.join([f'"{k}" ({v})' for k, v in Counter(starts3).most_common(3)])
        top4 = ', '.join([f'"{k}" ({v})' for k, v in Counter(starts4).most_common(3)])
        # Başlık formülleri
        formulas = extract_title_formulas([row[2] for row in group])
        top_formulas = ', '.join([f'"{k}" ({v})' for k, v in formulas])
        summary.append([kanal, analiz_turu, type_dist, top2, top3, top4, top_formulas])
    return summary

def write_analysis_to_sheet(sheet_name, analysis_rows):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIAL_PATH or JSON_KEY_FILE, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(sheet_name)
    try:
        worksheet = spreadsheet.worksheet('title_analiz')
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title='title_analiz', rows="1000", cols="30")
    worksheet.update(values=analysis_rows, range_name='A1')

def write_title_structure_to_sheet(sheet_name, structure_rows):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIAL_PATH or JSON_KEY_FILE, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(sheet_name)
    try:
        worksheet = spreadsheet.worksheet('title_structure')
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title='title_structure', rows="1000", cols="30")
    worksheet.update(values=structure_rows, range_name='A1')

def write_summary_to_sheet(sheet_name, summary_rows):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIAL_PATH or JSON_KEY_FILE, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(sheet_name)
    try:
        worksheet = spreadsheet.worksheet('title_structure_summary')
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title='title_structure_summary', rows="100", cols="30")
    worksheet.update(values=summary_rows, range_name='A1')

def write_transcripts_to_sheet(sheet_name, transcript_rows):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIAL_PATH or JSON_KEY_FILE, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open(sheet_name)
    try:
        worksheet = spreadsheet.worksheet('transcripts')
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title='transcripts', rows="10000", cols="3")
    worksheet.update(values=transcript_rows, range_name='A1')

def save_to_csv(data, filename):
    os.makedirs('data', exist_ok=True)
    import csv
    with open(f'data/{filename}', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def find_trend_matches(keywords, trending_topics):
    trending_lower = [t.lower() for t in trending_topics]
    return [kw for kw in keywords if kw.lower() in trending_lower]

# Güvenli JSON parse fonksiyonu
def safe_json_loads(x):
    try:
        if isinstance(x, str) and x.startswith("["):
            return json.loads(x)
        return []
    except Exception:
        return []


def get_summary_openai(text, max_tokens=800):
    """Generate summary and strategic analysis using the updated OpenAI API call and the initialized client."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        logging.warning("Skipping summary generation for short or invalid text.")
        return "[Özet için yetersiz metin]"
    try:
        # Use the client initialized at the top of the file
        response = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen bir YouTube içerik stratejisti ve metin yazarlığı uzmanısın.\n"
                        "Bir YouTube video transkriptini hem içerik olarak özetleyecek hem de stratejik açıdan analiz edeceksin.\n"
                        "Yanıtını Türkçe ver."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"""
Lütfen aşağıdaki YouTube video transkriptini analiz et:

1. Videonun temel konusunu ~250 kelime ile özetle ve giriş gelişme sonuç bölümlerini ayrı ayrı yaz. 
2. Videoda kullanılan dikkat çekme stratejilerini belirle:
   - Vurucu açılış cümlesi (hook) var mı?
   - Merak uyandıran ya da ters köşe yapılan kısımlar var mı?
   - Net ve güçlü bir vaat (promise) sunuluyor mu?
   - Duygusal veya entelektüel ikilik (duality) var mı?
   - İçeriden bilgi ya da özel bir bakış açısı sunuluyor mu?
3. Bu stratejilerin içerik pazarlaması ve YouTube algoritması açısından etkisini yorumla.

Transkript:
{text[:4800]}
                        """
                    )
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        summary = response.choices[0].message.content.strip()
        logging.info("OpenAI strategic summary generated successfully via get_summary_openai.")
        return summary
    except Exception as e:
        logging.error(f"OpenAI summary generation error in get_summary_openai: {e}", exc_info=True)
        return f"[OpenAI Strateji Analizi Hatası: {e}]"


def truncate_text(text, max_tokens=512):
    """
    Metni BERT modelinin maksimum token sınırına göre kısalt.
    """
    words = text.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens])
    return text

class ContentAnalyzer:
    def __init__(self):
        # NLTK gerekli dosyaları indir
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Türkçe stopwords listesi
        self.turkish_stopwords = set([
            've', 'ile', 'için', 'bu', 'bir', 'da', 'de', 'mi', 'ne', 'ama', 'fakat',
            'ancak', 'lakin', 'çünkü', 'zira', 'dolayı', 'gibi', 'kadar', 'göre',
            'karşı', 'doğru', 'yanında', 'beri', 'önce', 'sonra', 'öte', 'beri',
            'üstü', 'altı', 'içinde', 'dışında', 'arasında', 'yüzünden', 'dolayı',
            'sayesinde', 'nedeniyle', 'sebebiyle', 'göre', 'göre', 'karşı', 'doğru'
        ])
        
        self.stemmer = TrnlpWord()
        
        # Cache için dosya yolu
        self.cache_dir = "cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Google Trends istemcisi
        self.pytrends = TrendReq(hl='tr-TR', tz=360)
        
        # İşlenmiş video ID'leri için set
        self.processed_ids = set()
        self.load_processed_ids()

        # SQLite cache
        self.conn = sqlite3.connect('cache/cache.db')
        self.c = self.conn.cursor()
        self.c.execute('CREATE TABLE IF NOT EXISTS processed_ids(video_id TEXT PRIMARY KEY)')
        # BERT semantic similarity
        self.bert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        # Türkçe sentiment
        self.sentiment_tr = pipeline('sentiment-analysis', model="savasy/bert-base-turkish-sentiment-cased")
        # İngilizce sentiment
        self.sia = SentimentIntensityAnalyzer()

        # --- Sabit veya .env'den trend listesi ---
        trend_raw = os.getenv("TRENDING_TOPICS", "")
        if trend_raw:
            self.trending_topics = [t.strip() for t in trend_raw.split(",") if t.strip()]
        else:
            self.trending_topics = [
                'ai', 'yapay zeka', 'girişimcilik', 'finans', 'satış', 'ceo', 'influencer', 'freelance',
                'e-ticaret', 'eğitim', 'motivasyon', 'iş görüşmesi', 'liderlik', 'para', 'yatırım',
                'dijital pazarlama', 'kariyer', 'başarı', 'network', 'koçluk'
            ]

    def load_processed_ids(self):
        """İşlenmiş video ID'lerini cache'den yükler."""
        cache_file = os.path.join(self.cache_dir, "processed_ids.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.processed_ids = set(json.load(f))

    def save_processed_ids(self):
        """İşlenmiş video ID'lerini cache'e kaydeder."""
        cache_file = os.path.join(self.cache_dir, "processed_ids.json")
        with open(cache_file, 'w') as f:
            json.dump(list(self.processed_ids), f)

    def get_trending_topics(self, country='turkey'):
        try:
            trending = self.pytrends.trending_searches(pn=country)
            return trending[0].tolist() if not trending.empty else []
        except Exception as e:
            print(f"❌ Google Trends verisi alınamadı: {str(e)}")
            return []

    def title_transcript_similarity(self, title, transcript):
        """Başlık ve transkript arasındaki semantic similarity'yi hesaplar."""
        try:
            tfidf = TfidfVectorizer(stop_words='english').fit_transform([title, transcript])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except Exception as e:
            print(f"❌ Semantic similarity hesaplanamadı: {str(e)}")
            return 0.0

    def extract_topics_tfidf(self, text, top_n=5):
        """TF-IDF kullanarak en önemli konuları çıkarır."""
        try:
            tfidf = TfidfVectorizer(max_df=0.8, stop_words='english')
            tfidf_matrix = tfidf.fit_transform([text])
            scores = zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            return [w for w, s in sorted_scores[:top_n]]
        except Exception as e:
            print(f"❌ Konu çıkarılamadı: {str(e)}")
            return []

    def clean_turkish_text(self, text):
        """Türkçe metni temizler ve normalize eder."""
        # Küçük harfe çevir
        text = text.lower()
        # Noktalama işaretlerini kaldır
        text = re.sub(r'[^\w\s]', ' ', text)
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_with_trnlp(self, text):
        """Türkçe metni tokenize eder ve köklerini bulur."""
        text = self.clean_turkish_text(text)
        tokens = text.split()
        roots = []
        for token in tokens:
            if token not in self.turkish_stopwords and len(token) > 2:
                self.stemmer.setword(token)
                root = self.stemmer.stem
                if root:
                    roots.append(root)
        return roots

    def analyze_turkish_content(self, text):
        """Türkçe içerik analizi yapar."""
        # Metni tokenize et ve köklerini bul
        tokens = self.tokenize_with_trnlp(text)
        
        # Kelime frekanslarını hesapla
        word_freq = Counter(tokens)
        
        # En sık geçen kelimeleri bul
        common_words = word_freq.most_common(20)
        
        # Cümleleri ayır
        sentences = sent_tokenize(text)
        
        # Her cümlenin duygu analizini yap
        sentence_sentiments = []
        for sentence in sentences:
            sentiment = self.analyze_sentiment(sentence)
            sentence_sentiments.append({
                'sentence': sentence,
                'sentiment': sentiment
            })
        
        return {
            'word_frequencies': common_words,
            'sentence_sentiments': sentence_sentiments,
            'total_sentences': len(sentences),
            'total_words': len(tokens)
        }

    def analyze_content_structure(self, text):
        """İçerik yapısını analiz eder (giriş, gelişme, sonuç)."""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        # İçeriği üç bölüme ayır
        intro_end = int(total_sentences * 0.2)  # İlk %20
        body_end = int(total_sentences * 0.8)   # Son %20
        
        intro = ' '.join(sentences[:intro_end])
        body = ' '.join(sentences[intro_end:body_end])
        conclusion = ' '.join(sentences[body_end:])
        
        # Her bölümün analizini yap
        intro_analysis = self.analyze_turkish_content(intro)
        body_analysis = self.analyze_turkish_content(body)
        conclusion_analysis = self.analyze_turkish_content(conclusion)
        
        return {
            'introduction': intro_analysis,
            'body': body_analysis,
            'conclusion': conclusion_analysis
        }

    def save_processed_id(self, video_id):
        self.c.execute('INSERT OR IGNORE INTO processed_ids VALUES (?)', (video_id,))
        self.conn.commit()

    def is_processed(self, video_id):
        self.c.execute('SELECT 1 FROM processed_ids WHERE video_id=?', (video_id,))
        return self.c.fetchone() is not None

    def analyze_sentiment(self, text, lang='tr'):
        if lang == 'tr':
            try:
                result = self.sentiment_tr(text)
                return result[0]['label']
            except Exception as e:
                logging.error(f"Türkçe sentiment analizi hatası: {e}")
                return 'N/A'
        else:
            return self.sia.polarity_scores(text)

    def extract_keywords(self, text, top_n=10):
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in STOPWORDS]
        freq_dist = FreqDist(words)
        return [word for word, _ in freq_dist.most_common(top_n)]

    def semantic_similarity(self, title, transcript):
        try:
            # Metni kısalt
            title = self.truncate_text(title)
            transcript = self.truncate_text(transcript)
            
            embeddings = self.bert_model.encode([title, transcript])
            return util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception as e:
            logging.error(f"Semantic similarity hatası: {e}")
            return 0.0

    def fetch_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return video_id, ' '.join([t['text'] for t in transcript])
        except Exception as e:
            logging.warning(f"Transcript çekilemedi: {video_id} - {e}")
            return video_id, None

    def analyze_transcripts(self, df):
        analysis_results = []
        video_ids = df[df['has_caption'] == True]['video_id'].tolist()
        with ThreadPoolExecutor(max_workers=5) as executor:
            transcripts = list(executor.map(self.fetch_transcript, video_ids))
        transcript_map = {vid: txt for vid, txt in transcripts if txt}
        for _, video in df[df['has_caption'] == True].iterrows():
            if self.is_processed(video['video_id']):
                continue
            transcript_text = transcript_map.get(video['video_id'])
            if not transcript_text:
                logging.warning(f"Transcript bulunamadı: {video['video_id']}")
                continue
            keywords = self.extract_keywords(transcript_text)
            topics = self.extract_topics_tfidf(transcript_text)
            sentiment = self.analyze_sentiment(transcript_text, lang='tr')
            similarity = self.semantic_similarity(video['title'], transcript_text)
            trend_matches = find_trend_matches(keywords, self.trending_topics)
            summary = get_summary_openai(transcript_text)
            analysis_results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'transcript': transcript_text,
                'keywords': ','.join(keywords),
                'topics': ','.join(topics),
                'summary': summary,
                'sentiment': sentiment,
                'title_transcript_similarity': similarity,
                'trend_matches': ','.join(trend_matches)
            })
            self.save_processed_id(video['video_id'])
        results_df = pd.DataFrame(analysis_results)

        # Eğer analiz boşsa durdur.
        if results_df.empty:
            raise ValueError("Analiz sonuçları boş döndü! Transkriptleri kontrol et.")

        results_df.to_csv('data/analysis_report.csv', index=False)
        # Google Sheet'e de yazılabilir (isteğe bağlı):
        # write_analysis_to_sheet(GOOGLE_SHEET_NAME, results_df.values.tolist())
        return results_df

    def save_analysis_results(self, analysis_df, output_file="analysis_results.csv"):
        """Analiz sonuçlarını CSV dosyasına kaydeder."""
        analysis_df.to_csv(output_file, index=False)
        print(f"✅ Analiz sonuçları kaydedildi: {output_file}")

def load_data(base_path="data"):
    """Load metadata and transcripts from CSV files."""
    try:
        metadata_df = pd.read_csv(f"{base_path}/youtube_scrape_results.csv")
        transcripts_df = pd.read_csv(f"{base_path}/transcripts.csv")
        logging.info("Data loaded successfully.")
        return metadata_df, transcripts_df
    except FileNotFoundError as e:
        logging.error(f"Data files not found in {base_path}: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"One of the data files in {base_path} is empty: {e}")
        raise

def clean_text(text):
    """
    Metni temizler ve normalize eder
    """
    if not isinstance(text, str):
        return ""
    
    # Küçük harfe çevir
    text = text.lower()
    
    # Özel karakterleri kaldır
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_topics(texts, n_topics=5):
    """
    TF-IDF kullanarak ana konuları çıkarır
    """
    # Metinleri temizle
    cleaned_texts = [clean_text(text) for text in texts]
    
    # TF-IDF vektörlerini oluştur
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=stopwords.words('turkish') + stopwords.words('english')
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Her metin için en önemli kelimeleri bul
        topics = []
        for i in range(len(cleaned_texts)):
            if tfidf_matrix[i].nnz > 0:  # Boş olmayan metinler için
                # En yüksek TF-IDF değerlerine sahip kelimeleri al
                top_indices = tfidf_matrix[i].toarray()[0].argsort()[-n_topics:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                topics.append(top_words)
            else:
                topics.append([])
        
        return topics
    except Exception as e:
        logging.error(f"Topik çıkarma hatası: {e}")
        return [[] for _ in range(len(texts))]

def analyze_sentiment(texts):
    """
    Metinlerin duygu analizini yapar
    """
    try:
        # Türkçe duygu analizi için model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="savasy/bert-base-turkish-sentiment",
            tokenizer="savasy/bert-base-turkish-sentiment"
        )
        
        sentiments = []
        for text in texts:
            if not text or len(text.strip()) == 0:
                sentiments.append({"label": "NEUTRAL", "score": 1.0})
                continue
                
            # Metni parçalara böl (model limiti için)
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            chunk_sentiments = []
            
            for chunk in chunks:
                if chunk.strip():
                    result = sentiment_analyzer(chunk)[0]
                    chunk_sentiments.append(result)
            
            if chunk_sentiments:
                # En yüksek skora sahip duyguyu al
                main_sentiment = max(chunk_sentiments, key=lambda x: x['score'])
                sentiments.append(main_sentiment)
            else:
                sentiments.append({"label": "NEUTRAL", "score": 1.0})
        
        return sentiments
    except Exception as e:
        logging.error(f"Duygu analizi hatası: {e}")
        return [{"label": "NEUTRAL", "score": 1.0} for _ in range(len(texts))]

def generate_summary(text, max_length=150):
    """
    OpenAI ile metin özeti oluşturur
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Metni kısa ve öz bir şekilde özetle."},
                {"role": "user", "content": text[:4000]}  # Token limiti için
            ],
            max_tokens=max_length,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Özet oluşturma hatası: {e}")
        return text[:max_length] + "..."

def calculate_semantic_similarity(texts):
    """
    Metinler arasındaki semantik benzerliği hesaplar
    """
    try:
        # Çok dilli model kullan
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Metinleri temizle
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Embedding'leri hesapla
        embeddings = model.encode(cleaned_texts)
        
        # Benzerlik matrisini hesapla
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    except Exception as e:
        logging.error(f"Semantik benzerlik hesaplama hatası: {e}")
        return np.zeros((len(texts), len(texts)))

def create_analysis_report(metadata_df, transcripts_df, output_path="data"):
    """
    Orchestrates the analysis pipeline using modular functions and saves the results.
    """
    os.makedirs(output_path, exist_ok=True)
    
    if 'video_id' not in metadata_df.columns or 'video_id' not in transcripts_df.columns:
        logging.error("Missing 'video_id' column in one of the input dataframes.")
        raise ValueError("Input DataFrames must contain a 'video_id' column.")

    # Merge metadata and transcripts
    try:
        df = pd.merge(metadata_df, transcripts_df, on='video_id', how='inner', suffixes=('_meta', '_transcript'))
        if df.empty:
            logging.warning("No common videos found after merging metadata and transcripts. Analysis report will be empty.")
            # Create empty files to avoid downstream errors if expected
            pd.DataFrame().to_csv(f"{output_path}/analysis_report.csv", index=False)
            with open(f"{output_path}/training_data.jsonl", 'w', encoding='utf-8') as f:
                pass # Create an empty jsonl file
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error merging dataframes: {e}")
        raise

    # Limit DataFrame to first 20 rows for testing
    if not df.empty:
        #df = df.head(5)
        #logging.info("Testing mode: Processing only the first 5 videos from the merged data.")
        pass
    else:
        logging.warning("Merged DataFrame is empty, no videos to process.")
        # Ensure empty files are created to prevent downstream errors
        pd.DataFrame().to_csv(f"{output_path}/analysis_report.csv", index=False)
        with open(f"{output_path}/training_data.jsonl", 'w', encoding='utf-8') as f:
            pass 
        return pd.DataFrame()

    # >>> DEBUG BAŞLANGIÇ <<<
    print("\n--- DataFrame Info After Merge ---")
    try:
        # Use buffer to capture df.info() output as it prints to stdout
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        print(s)
    except Exception as info_e:
        print(f"Could not print df.info(): {info_e}")
        
    print("\n--- Title_meta Column NaN Count ---")
    try:
        print(df['title_meta'].isna().sum())
    except KeyError:
        print("'title_meta' column not found after merge.")
    except Exception as nan_e:
        print(f"Could not calculate NaN count for title_meta: {nan_e}")
        
    print("\n--- Title_meta Column Empty String Count ---")
    try:
        print((df['title_meta'] == '').sum())
    except KeyError:
        print("'title_meta' column not found after merge.")
    except Exception as empty_e:
        print(f"Could not calculate empty string count for title_meta: {empty_e}")
        
    print("\n--- First 5 Rows with Potentially problematic Titles (NaN or Empty in title_meta) ---")
    try:
        problematic_titles_df = df[df['title_meta'].isna() | (df['title_meta'] == '')].head()
        if not problematic_titles_df.empty:
            print(problematic_titles_df[['video_id', 'title_meta']])
        else:
            print("No rows found with NaN or empty titles in 'title_meta' in the first checks.")
    except KeyError:
        print("'title_meta' column not found after merge.")
    except Exception as head_e:
        print(f"Could not retrieve problematic titles head: {head_e}")
        
    print("--- DEBUG SON ---")
    # <<< DEBUG SON >>>

    all_results = []
    
    logging.info(f"Starting analysis for {len(df)} videos.")
    for index, row in df.iterrows():
        video_id = row['video_id']
        # Correctly access the title column with the suffix added by merge
        title = row.get('title_meta', '') 
        # Also adjust other columns if they potentially have suffixes from merge
        # Check df.columns after merge if unsure about other columns like description, published_at etc.
        description = row.get('description_meta', row.get('description', '')) # Try suffixed, fallback to original if needed
        transcript = row.get('transcript', '') # transcript comes from transcripts_df, suffix _transcript not expected here unless it existed in metadata too.

        if not title or not isinstance(title, str):
            logging.warning(f"Missing or invalid title for video_id: {video_id}. Skipping.")
            continue
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) < 10: # Basic check for empty transcript
            logging.warning(f"Missing or insufficient transcript for video_id: {video_id}. Skipping analysis for this video.")
            # Optionally, add a placeholder result or skip entirely
            # For now, skipping to avoid errors in analysis modules
            all_results.append({
                "video_id": video_id, "title": title, "description": description,
                "classic_analysis": "N/A (No Transcript)", "gpt_analysis": "N/A (No Transcript)",
                "intro_analysis": "N/A (No Transcript)", "script": "N/A (No Transcript)",
                "title_suggestions": [], "ai_summary": "N/A (No Transcript)",
                "view_count": row.get('view_count_meta', 0),
                "like_count": row.get('like_count_meta', 0),
                "comment_count": row.get('comment_count_meta', 0),
                "error_message": "Transcript missing or too short"
            })
            continue

        logging.info(f"Processing video: {title} (ID: {video_id})")

        try:
            # 1. Video Analysis (Classic + GPT)
            # KONU_LISTESI and STRATEJI_LISTESI are global here
            video_analysis_result = analyze_video(title, description, transcript, KONU_LISTESI, STRATEJI_LISTESI)
            
            # 2. Intro Analysis (First minute - approx 1000 chars)
            intro_text = transcript[:1000]
            intro_analysis_result = analyze_intro(intro_text) # This module should handle if intro_text is too short
            
            # 3. Generate Script based on GPT analysis of the full video
            # Ensure gpt_analysis content is valid before passing
            gpt_video_analysis_content = video_analysis_result.get('gpt_analysis', '')
            if not gpt_video_analysis_content or gpt_video_analysis_content.startswith("[OpenAI hata]"):
                 logging.warning(f"Skipping script generation for {video_id} due to previous GPT analysis error.")
                 generated_script = "N/A (GPT Analysis Error)"
            else:
                generated_script = generate_script_from_analysis(gpt_video_analysis_content)
            
            # 4. Suggest Titles based on GPT analysis
            if not gpt_video_analysis_content or gpt_video_analysis_content.startswith("[OpenAI hata]"):
                logging.warning(f"Skipping title suggestion for {video_id} due to previous GPT analysis error.")
                suggested_titles_list = ["N/A (GPT Analysis Error)"]
            else:
                suggested_titles_list = suggest_titles(
                    summary=gpt_video_analysis_content, # Using the full video's GPT analysis as summary
                    topics=", ".join(video_analysis_result.get('classic', {}).get('topics', [])),
                    sentiment=video_analysis_result.get('classic', {}).get('sentiment', 'N/A'),
                    trends=row.get('trend_matches', ''), # Assuming trend_matches is a string list from scraper
                    channel_name=row.get('channel_title', 'Fatih Çoban'), # Use actual channel_title if available
                    n=3
                )
            
            # 5. Generate AI Summary for the full transcript
            ai_summary_text = get_summary_openai(transcript)
            
            current_result = {
                "video_id": video_id,
                "title": title,
                "description": description,
                "transcript": transcript,
                "transcript_length": len(transcript),
                "classic_analysis": video_analysis_result.get('classic', {}),
                "gpt_analysis": gpt_video_analysis_content,
                "intro_analysis": intro_analysis_result.get('gpt_analysis', {}), # Storing GPT part of intro analysis
                "classic_intro_analysis": intro_analysis_result.get('classic', {}),
                "generated_script": generated_script,
                "suggested_titles": suggested_titles_list,
                "ai_summary": ai_summary_text,
                "view_count": row.get('view_count_meta', 0), # Assuming from metadata
                "like_count": row.get('like_count_meta', 0),
                "comment_count": row.get('comment_count_meta', 0),
                "publish_date": row.get('publish_date_meta', None)
            }
            all_results.append(current_result)

        except Exception as e:
            logging.error(f"Error processing video_id {video_id} ('{title}'): {e}", exc_info=True)
            all_results.append({
                "video_id": video_id, "title": title,
                "error_message": str(e)
            })
    
    results_df = pd.DataFrame(all_results)
    
    # Save main analysis report
    report_csv_path = f"{output_path}/analysis_report.csv"
    results_df.to_csv(report_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    logging.info(f"Analysis report saved to {report_csv_path}")
    
    # Prepare and save data for fine-tuning
    # This focuses on successful script and summary generation
    training_data = []
    logging.info(f"Preparing fine-tuning data. Checking {len(all_results)} items from analysis results.")
    for i, item in enumerate(all_results):
        # Ensure essential fields for training are present and valid
        has_error_message = item.get("error_message")
        
        # Check if generated script and AI summary are valid (not error messages or placeholders)
        generated_script_content = item.get("generated_script", "")
        ai_summary_content = item.get("ai_summary", "")

        script_is_valid = generated_script_content and not generated_script_content.startswith("[") and not generated_script_content == "N/A (GPT Analysis Error)" and len(generated_script_content.strip()) > 10
        summary_is_valid = ai_summary_content and not ai_summary_content.startswith("[") and len(ai_summary_content.strip()) > 10
        
        transcript_present = item.get('transcript') and len(item.get('transcript', '').strip()) > 10

        # Detailed logging for why an item might be skipped
        if not (transcript_present and script_is_valid and summary_is_valid and not has_error_message):
            logging.debug(f"Skipping item {i} for training data. Video ID: {item.get('video_id', 'N/A')}")
            if has_error_message:
                logging.debug(f"  Reason: Has error_message: {has_error_message}")
            if not transcript_present:
                logging.debug(f"  Reason: Transcript missing or too short. Length: {len(item.get('transcript', ''))}")
            if not script_is_valid:
                logging.debug(f"  Reason: Script not valid. Content snippet: '{generated_script_content[:100]}...'")
            if not summary_is_valid:
                logging.debug(f"  Reason: Summary not valid. Content snippet: '{ai_summary_content[:100]}...'")
            continue # Skip this item if any condition is not met

        # If all checks pass, create the training item
        training_item = {
            "messages": [
                {"role": "system", "content": "You are an expert YouTube content strategist and scriptwriter for the 'Fatih Çoban' channel. Your goal is to analyze video transcripts and generate engaging scripts and summaries."},
                {"role": "user", "content": f"Analyze this transcript and generate a script and a summary. Transcript: {item['transcript'][:3500]}"}, # Limiting transcript for prompt
                {"role": "assistant", "content": f"Generated Script: {generated_script_content}\\n\\nSummary: {ai_summary_content}"}
            ]
        }
        training_data.append(training_item)
        logging.debug(f"Item {i} (Video ID: {item.get('video_id')}) added to training data.")
            
    jsonl_path = f"{output_path}/training_data.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f"Fine-tuning data saved to {jsonl_path} with {len(training_data)} entries.")
    
    return results_df

def main():
    """
    Main function to run the analysis pipeline.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        logging.info("--- Starting YouTube Content Analysis Pipeline ---")
        
        # 1. Load data
        metadata_df, transcripts_df = load_data(base_path="data")
        
        if metadata_df.empty or transcripts_df.empty:
            logging.warning("Metadata or transcripts are empty. Pipeline will not produce full results.")
            # Optionally, create empty output files or stop
            if not os.path.exists("data/analysis_report.csv"):
                 pd.DataFrame().to_csv("data/analysis_report.csv", index=False)
            if not os.path.exists("data/training_data.jsonl"):
                with open("data/training_data.jsonl", 'w') as f:
                    pass
            return

        # 2. Create analysis report (which includes all modular analyses and OpenAI calls)
        analysis_df = create_analysis_report(metadata_df, transcripts_df, output_path="data")
        
        if not analysis_df.empty:
            logging.info(f"Successfully generated analysis report with {len(analysis_df)} processed videos.")
        else:
            logging.warning("Analysis report is empty after processing.")
            
        
        logging.info("--- YouTube Content Analysis Pipeline Finished ---")
        
    except FileNotFoundError:
        logging.error("Essential data files (metadata or transcripts) not found. Please ensure they are in the 'data' directory.")
    except ValueError as ve: # Catch specific errors like missing columns
        logging.error(f"ValueError in pipeline: {ve}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)
        # Consider re-raising or handling more gracefully depending on desired behavior
        raise

if __name__ == "__main__":
    main()
