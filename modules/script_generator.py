# modules/script_generator.py
import os
import openai
from modules.prompt_templates import SCRIPT_GENERATION_PROMPT

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Analiz Tabanlı Script Üretimi ---
def generate_script_from_analysis(analysis, max_tokens=900):
    prompt = SCRIPT_GENERATION_PROMPT.format(analysis=analysis)
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

# --- Brief Tabanlı Script Üretimi (Chat Asistanı için) ---
def generate_script_from_brief(brief, max_tokens=2000):
    prompt = f"""Aşağıda Fatih Çoban YouTube kanalı için bir video brief'i bulunmaktadır. Bu brief'e uygun olarak, hedef kitle olan gençleri etkileyecek, yaklaşık 15-20 dakikalık bir video için detaylı bir Türkçe YouTube video scripti yaz.

**Kullanıcı Brief'i:**
{brief}

**Script Yazım Kuralları ve Detayları:**

1.  **Genel Ton ve Stil:** Fatih Çoban'ın bilinen enerjik, motive edici, doğrudan ve samimi anlatım stilini kullan. Gençlere hitap eden, anlaşılır bir dil kullan.

2.  ****Giriş (Hook + Otorite + Değer Vaadi - İlk 2-3 dakika):**

Intro aşağıdaki 7 adıma göre kurgulanmalı (YouTube'da dikkat süresini maksimuma çıkarmak için):

1. **TNT Hazırlığı:** İlk cümle, video başlığına çok yakın olmalı. İlk görsel veya anlatım, küçük resimde vaat edilen şeyi desteklemeli.

2. **Şok Edici veya Merak Uyandırıcı Başlangıç:** İzleyiciyi ilk 5 saniyede durduran bir cümleyle başla. Örn: "Dijital pazarlamanın %90'ı aslında işe yaramıyor!"

3. **Merak Boşluğu:** İzleyicinin aklına "ne olacak şimdi?" dedirtecek bir belirsizlik veya hedef oluştur. (örn: "Bu yöntem gerçekten işe yarayacak mı?")

4. **Bağlam Ver:** Video konusunu ve neden önemli olduğunu 1-2 cümleyle açıkla.

5. **Emeği Göster (Input Bias):** Bu videoyu hazırlamak için yapılan zahmeti veya özel çabayı belirt. ("Bu videoyu hazırlamak için 3 gece boyunca tüm Facebook reklam panellerini analiz ettim.")

6. **Göstererek Anlat:** Açıklamaları mümkün olduğunca örnekle, görsel veya hikayeyle destekle. (örnek: bir ekran görüntüsü ya da gerçek bir vaka anlatımı)

7. **Yüksek Tempo:** İlk 30 saniye boyunca metin ya da görsel akışı çok hızlı olmalı (ortalama 1.5 saniyede bir bilgi değişimi gibi düşünülmeli).

Tüm bunlar, izleyiciye "bu video değerli, kalmaya değer" hissini vermeli.


Giriş (Hook + Otorite + Değer Vaadi - Yaklaşık 2-3 dakika):**
    *   **Hook:** İzleyiciyi ilk 15-30 saniyede yakalayacak güçlü bir kanca (şok edici bir istatistik, düşündürücü bir soru, kısa bir kişisel anekdot veya cesur bir iddia) ile başla.
    *   **Otorite İnşası:** Fatih Çoban'ın bu konudaki uzmanlığını ve deneyimini (geçmiş başarıları, bu konudaki bilgi birikimi, neden bu konuda konuşmaya yetkin olduğu) dolaylı veya doğrudan vurgula. İzleyiciye 'Neden Fatih Çoban'ı dinlemeliyim?' sorusunun cevabını ver.
    *   **Değer Vaadi:** Bu videoyu sonuna kadar izlediklerinde ne kazanacaklarını, hangi sorunlarına çözüm bulacaklarını veya hangi yeni bakış açılarını edineceklerini net bir şekilde belirt. Videonun ana faydasını vurgula.
    *   **Yüzeysellikten Kaçın:** 'Bu videoyu izlemelisin çünkü...' gibi basit ifadeler yerine, metin yazarlığı teknikleriyle merak uyandır ve ikna et.

3.  **Gelişme (Problemler, Çözümler ve Derinlemesine Analiz - Yaklaşık 10-14 dakika):**
    *   **Potansiyel Sorun/Soru Belirleme:** Konuyla ilgili olarak hedef kitlenin (Türkiye'deki gençler) karşılaşabileceği 3 ila 5 temel problemi, soruyu veya merak ettiği noktayı belirle.
    *   **Ayrıntılı Alt Bölümler:** Her bir problem/soru için ayrı bir alt başlık veya bölüm oluştur.
    *   **Detaylı Cevaplar ve Çözümler:** Her alt bölümde:
        *   Problemi/soruyu net bir şekilde tanımla.
        *   Fatih Çoban'ın kendi deneyimlerinden, gözlemlerinden veya araştırmalarından örnekler vererek konuyu somutlaştır.
        *   İzleyicinin hem içsel (örn: motivasyon eksikliği, korkular, yanlış inançlar) hem de dışsal (örn: bilgi eksikliği, kaynaklara ulaşım, fırsatları görememe) engellerini anladığını göster.
        *   Bu engelleri aşmak için pratik, uygulanabilir, adım adım stratejiler ve çözümler sun.
        *   İzleyiciyi bilgilendirirken aynı zamanda motive et ve 'ben de yapabilirim' hissini ver.
        *   Verilen bilgilerin doyurucu ve tatmin edici olduğundan emin ol.

4.  **Sonuç (Özet ve Güçlü CTA - Yaklaşık 1-2 dakika):**
    *   **Ana Mesajların Özeti:** Gelişme bölümünde anlatılan ana fikirleri ve çözümleri kısaca özetle.
    *   **Harekete Geçirici Mesaj (CTA):** İzleyiciyi sadece abone olmaya değil, aynı zamanda öğrendiklerini hayata geçirmeye teşvik et. Belki konuyla ilgili bir sonraki adımı (bir kaynak, bir eylem planı, bir düşünme egzersizi) öner.
    *   Videoyu pozitif ve motive edici bir mesajla kapat.

Lütfen bu detaylara uygun, Fatih Çoban'ın ağzından yazılmış gibi doğal ve akıcı bir script oluştur.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.75,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI hata]: {e}" 