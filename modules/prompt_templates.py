# modules/prompt_templates.py

# --- Video Strateji Analiz Promptu ---
VIDEO_STRATEGY_ANALYSIS_PROMPT = '''
Aşağıda bir YouTube videosunun başlığı, açıklaması ve transkripti var. Lütfen aşağıdaki analizleri Türkçe ve madde madde olarak yap:
1. **Konu Başlıkları:** Aşağıdaki listeden bu videoda işlenen konuları seç ve örnekle açıkla:
{KONU_LISTESI}
2. **Kullanılan Stratejiler:** Aşağıdaki stratejilerden hangileri bu videoda kullanılmış? (Her biri için örnek cümleyle açıkla)
{STRATEJI_LISTESI}
3. **Intro Analizi:** Videonun ilk 60 saniyesinde hangi teknik(ler) kullanılmış? (Curiosity gap, meydan okuma, şok, soru, hikaye, vaad, vs.)
4. **Giriş-Gelişme-Sonuç:** Her bölümde neler anlatılıyor? (Kısa özet)
5. **Hook, Story, Offer, CTA:** Bu bölümler var mı? Varsa örnekle açıkla.
6. **Hedef Kitle ve Çözümler:** İzleyicinin hangi problemlerine çözüm sunuluyor, hangi sorulara cevap veriliyor?
7. **Duygu ve Dil Analizi:** Hangi duygular tetikleniyor, nasıl bir dil kullanılmış?
8. **Kapanış ve CTA:** Videonun sonunda izleyiciye ne söyleniyor, hangi aksiyon isteniyor?

---

Başlık: {title}
Açıklama: {description}
Transcript: {transcript}
'''

# --- Intro Analiz Promptu ---
INTRO_ANALYSIS_PROMPT = '''
Aşağıda bir YouTube videosunun ilk 1 dakikalık transkripti var. Lütfen aşağıdaki analizleri Türkçe ve madde madde olarak yap:

1. **Intro Türü:** (Curiosity gap, meydan okuma, şok, soru, hikaye, vaad, istatistik, kişisel deneyim, vs.)
2. **Problem Tanımı:** İzleyiciye hangi problem/dert gösteriliyor?
3. **Çözüm Önerisi:** İzleyiciye ne vaat ediliyor?
4. **Hook/CTA:** İzleyiciyi tutmak için hangi teknik(ler) kullanılmış?

Transcript (ilk 1 dakika): {intro_transcript}
'''

# --- Script Üretim Promptu ---
SCRIPT_GENERATION_PROMPT = '''
Aşağıda bir YouTube videosunun detaylı strateji ve yapı analizi var. Bu analize uygun şekilde, intro, giriş, gelişme, sonuç, hook, story, offer, CTA gibi bölümleri içeren, izleyiciyi yakalayan ve kanalın stiline uygun bir Türkçe video scripti yaz:

{analysis}

Scripti bölümlere ayır (Intro, Giriş, Gelişme, Sonuç, CTA) ve her bölümü başlıkla belirt.
'''

# --- Başlık Öneri Promptu ---
TITLE_SUGGESTION_PROMPT = '''
Kanal: {channel_name}
Video özeti: {summary}
Temalar: {topics}
Duygu: {sentiment}
Trendler: {trends}
Soru: Bu içerik için 3 ilgi çekici, merak temelli Türkçe başlık öner.
''' 