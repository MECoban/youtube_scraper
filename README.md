# YouTube İçerik Analizi ve İçerik Üretim Sistemi

## Proje Amacı
Bu proje, Fatih Çoban YouTube kanalı için metin ve başlık yazarı bir sistem geliştirmeyi amaçlamaktadır. Belirlenen YouTube kanallarının en iyi ve en güncel videolarını analiz ederek, konu, başlık ve içerik analizi yapılmakta ve bu analizler doğrultusunda yeni içerik önerileri sunulmaktadır. Ayrıca, Türkiye'deki güncel konuları takip ederek, içerik üretiminde güncel trendleri de göz önünde bulundurmaktadır.

## Genişletilmiş YouTube İçerik Analizi
- **Transkript Analizi**: Anahtar kelime çıkarımı ve NLP ile konu başlıkları belirlenir.
- **Video Süresi / Etkileşim Analizi**: Süre ve izlenme sayısı korelasyonu, izlenme başına etki puanı hesaplanır.
- **Başlık + Transkript Korelasyonu**: Başlıkta geçen kelimelerin içerikte geçip geçmediği kontrol edilir.

## Türkiye'deki Güncel Konuların Çekilmesi
- **Google Trends API**: Anahtar kelime arama ve TR bazlı bölgesel trendler.
- **Haber Siteleri Scraper**: Kategori bazlı manşet kelime çıkarımı ve başlık kümeleme.
- **Twitter/X Trendleri**: Türkiye lokasyonu için trend konular.
- **Instagram / TikTok Hashtag Analizleri**: Manuel olarak takip edilebilir.

## Üretim Motoru: Başlık + Script Generator
- **Prompt Şablonları**: Başlık ve içerik için ayrı ayrı hazırlanır.
- **Girdi**: Konu, ton ve hedef kitle.
- **Çıktı**: Başlık önerileri ve script (100–300 kelime).

## Streamlit Arayüzü
- **Dashboard**: Güncel konular ve analizler.
- **İçerik Fikirleri**: Video öneri motoru.
- **Başlık ve Script Üretimi**: Kullanıcıya sunulan modüller.
- **Örnek Sonuçların Çıktısı**: PDF, kopyala veya Google Sheets.

## Kurulum ve Kullanım
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Google API anahtarınızı ve JSON anahtar dosyanızı ayarlayın.
3. Projeyi çalıştırın:
   ```bash
   python src/yt_scraper.py
   ```

## Geliştirici
- **İsim**: [Geliştirici Adı]
- **İletişim**: [E-posta Adresi]

## Lisans
Bu proje [Lisans Adı] altında lisanslanmıştır. 