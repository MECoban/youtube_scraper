# train_data_collector.py

import pandas as pd
import json
from datetime import datetime
import os
from yt_scraper import run_for_channel_links, channel_links
from analyse import ContentAnalyzer
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import subprocess
import glob
import sys
import re
from googleapiclient.discovery import build

load_dotenv()

class TrainingDataCollector:
    def __init__(self):
        self.analyzer = ContentAnalyzer()
        self.output_dir = "data/training"
        os.makedirs(self.output_dir, exist_ok=True)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_transcript(self, video_id):
        """
        Video transkriptini Türkçe olarak al.
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr'])
            return ' '.join([item['text'] for item in transcript_list])
        except Exception as e:
            print(f"❌ Transkript alınamadı ({video_id}): {str(e)}")
            return None

    def truncate_text(self, text, max_tokens=512):
        """
        Metni BERT modelinin maksimum token sınırına göre kısalt.
        """
        words = text.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens])
        return text

    def collect_training_data(self):
        """
        YouTube kanallarından video verilerini çek, analiz et ve eğitim verisi olarak kaydet.
        """
        print("📥 Video verileri çekiliyor...")
        df = run_for_channel_links(channel_links)
        
        print("🔍 Videolar analiz ediliyor...")
        training_data = []
        
        for _, row in df.iterrows():
            # Transkripti al
            transcript = self.get_transcript(row['video_id'])
            if not transcript:
                continue
                
            # Metni kısalt
            transcript = self.truncate_text(transcript)
            
            # Her video için eğitim verisi oluştur
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Sen Fatih Çoban YouTube kanalı için başlık ve script yazan uzman bir asistansın."
                    },
                    {
                        "role": "user",
                        "content": f"""Video başlığı: {row['title']}
Video özeti: {row.get('summary', '')}
Temalar: {row.get('topics', '')}
Duygu: {row.get('sentiment', '')}
Trendler: {row.get('trend_matches', '')}
"""
                    },
                    {
                        "role": "assistant",
                        "content": transcript
                    }
                ]
            }
            training_data.append(training_example)
        
        # JSONL dosyasına kaydet
        output_file = os.path.join(self.output_dir, f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in training_data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✅ Eğitim verisi kaydedildi: {output_file}")
        return output_file

    def collect_channel_specific_data(self, channel_name):
        """
        Belirli bir kanalın verilerini topla ve eğitim verisi olarak kaydet.
        """
        print(f"📥 {channel_name} kanalından veriler çekiliyor...")
        channel_df = run_for_channel_links([f"https://www.youtube.com/@{channel_name}"])
        
        print("🔍 Videolar analiz ediliyor...")
        training_data = []
        
        for _, row in channel_df.iterrows():
            # Transkripti al
            transcript = self.get_transcript(row['video_id'])
            if not transcript:
                continue
                
            # Metni kısalt
            transcript = self.truncate_text(transcript)
            
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Sen {channel_name} YouTube kanalı için başlık ve script yazan uzman bir asistansın."
                    },
                    {
                        "role": "user",
                        "content": f"""Video başlığı: {row['title']}
Video özeti: {row.get('summary', '')}
Temalar: {row.get('topics', '')}
Duygu: {row.get('sentiment', '')}
Trendler: {row.get('trend_matches', '')}
"""
                    },
                    {
                        "role": "assistant",
                        "content": transcript
                    }
                ]
            }
            training_data.append(training_example)
        
        # JSONL dosyasına kaydet
        output_file = os.path.join(self.output_dir, f"{channel_name}_training_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in training_data:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✅ {channel_name} eğitim verisi kaydedildi: {output_file}")
        return output_file

def set_active_model(model_name):
    """
    Aktif kullanılacak modeli data/active_model.txt dosyasına kaydeder.
    """
    with open("data/active_model.txt", "w", encoding="utf-8") as f:
        f.write(model_name.strip())
    print(f"✅ Aktif model güncellendi: {model_name}")

def get_active_model():
    """
    Aktif modeli data/active_model.txt dosyasından okur. Yoksa default modeli döner.
    """
    try:
        with open("data/active_model.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "ft:gpt-3.5-turbo-1106:personal:luisbergerai:BUNayvMY"

def run_fine_tune(model_name=None, suffix="feedback-enhanced"):
    """
    En güncel JSONL dosyası ile OpenAI fine-tune komutunu otomatik çalıştırır ve yeni modeli aktif olarak kaydeder.
    """
    jsonl_files = sorted(glob.glob("data/training/*.jsonl"), reverse=True)
    if not jsonl_files:
        print("❌ JSONL dosyası bulunamadı!")
        return
    latest_jsonl = jsonl_files[0]
    if not model_name:
        model_name = get_active_model()
    print(f"🚀 Fine-tune için kullanılacak dosya: {latest_jsonl}")
    cmd = [
        "openai", "api", "fine_tunes.create",
        "-t", latest_jsonl,
        "-m", model_name,
        "--suffix", suffix
    ]
    print(f"Çalıştırılan komut: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("--- OpenAI Fine-tune Çıktısı ---")
    print(result.stdout)
    if result.stderr:
        print("--- Hata/uyarı ---")
        print(result.stderr)
    # Yeni model adını çıktıdan yakala ve kaydet
    match = re.search(r'"fine_tuned_model":\s*"([^"]+)"', result.stdout)
    if match:
        new_model = match.group(1)
        set_active_model(new_model)
    else:
        print("⚠️ Yeni model adı otomatik bulunamadı. Lütfen çıktıyı kontrol edin.")
    # Fine-tune sonrası model performans tracker'ı tetikle
    subprocess.run(["python", "src/model_performance_tracker.py"])

def get_video_stats(video_id, youtube=None):
    """
    YouTube API ile video istatistiklerini döndürür.
    """
    if youtube is None:
        API_KEY = os.getenv('API_KEY')
        youtube = build('youtube', 'v3', developerKey=API_KEY)
    request = youtube.videos().list(
        part="statistics,contentDetails",
        id=video_id
    )
    response = request.execute()
    if not response['items']:
        return None
    stats = response['items'][0]['statistics']
    return {
        'video_id': video_id,
        'viewCount': stats.get('viewCount', 0),
        'likeCount': stats.get('likeCount', 0),
        'commentCount': stats.get('commentCount', 0)
    }

def save_video_performance(video_ids, out_csv="data/video_performance.csv"):
    """
    Birden fazla video ID'si için performans verilerini çekip CSV'ye kaydeder.
    """
    API_KEY = os.getenv('API_KEY')
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    import pandas as pd
    all_stats = []
    for vid in video_ids:
        stats = get_video_stats(vid, youtube)
        if stats:
            all_stats.append(stats)
    df = pd.DataFrame(all_stats)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Video performans verileri kaydedildi: {out_csv}")

def create_training_examples_from_scripts(
    transcript_csv="data/transcripts.csv",
    summary_csv="data/script_summaries.csv",
    out_jsonl="data/training/training_data.jsonl",
    min_length=100,
    style="eğitici"
):
    import pandas as pd
    import json
    import os
    if not os.path.exists(transcript_csv) or not os.path.exists(summary_csv):
        print("Gerekli dosyalar bulunamadı!")
        return
    df_script = pd.read_csv(transcript_csv)
    df_summary = pd.read_csv(summary_csv)
    summary_map = {row['video_id']: row for _, row in df_summary.iterrows()}
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_script.iterrows():
            transcript = str(row.get("transcript", "")).strip()
            if not transcript or len(transcript.split()) < min_length:
                continue
            video_id = row.get("video_id", "")
            summary_row = summary_map.get(video_id, {})
            # Assistant cevabı: transkript + özet + analiz
            assistant_content = f"""
[Transkript]
{transcript}

[Özet ve Analiz]
{summary_row.get('summary_analysis_tr', '')}

[Stil]
{style}
"""
            # User mesajı: daha doğal, bağlamsal
            user_content = f"""
Aşağıdaki bilgileri kullanarak '{row.get('title', '')}' başlıklı video için yeni bir video scripti ve başlık önerisi yap:
- Video özeti: {summary_row.get('summary_analysis_tr', '')[:300]}
- Temalar: {row.get('topics', '')}
- Duygu tonu: {row.get('sentiment', '')}
- Trend kavramlar: {row.get('trend_matches', '')}
- Stil: {style}
"""
            # (Opsiyonel) metadata
            metadata = {
                "video_id": video_id,
                "title": row.get("title", ""),
                "date": row.get("published_at", ""),
                "style": style
            }
            json.dump({
                "messages": [
                    {"role": "system", "content": "Sen Fatih Çoban YouTube kanalı için başlık ve script yazan uzman bir asistansın."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": metadata
            }, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Training JSONL oluşturuldu: {out_jsonl}")

def create_title_training_examples_from_scripts(
    transcript_csv="data/transcripts.csv",
    summary_csv="data/script_summaries.csv",
    out_jsonl="data/training/title_training_data.jsonl",
    min_length=100
):
    import pandas as pd
    import json
    import os
    if not os.path.exists(transcript_csv) or not os.path.exists(summary_csv):
        print("Gerekli dosyalar bulunamadı!")
        return
    df_script = pd.read_csv(transcript_csv)
    df_summary = pd.read_csv(summary_csv)
    summary_map = {row['video_id']: row for _, row in df_summary.iterrows()}
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df_script.iterrows():
            transcript = str(row.get("transcript", "")).strip()
            if not transcript or len(transcript.split()) < min_length:
                continue
            video_id = row.get("video_id", "")
            summary_row = summary_map.get(video_id, {})
            # User mesajı: video özeti ve analizden başlık isteği
            user_content = f"""
Aşağıdaki bilgileri kullanarak '{row.get('title', '')}' başlıklı video için yeni, ilgi çekici ve Türkçe bir başlık önerisi yap:
- Video özeti: {summary_row.get('summary_analysis_tr', '')[:300]}
- Temalar: {row.get('topics', '')}
- Duygu tonu: {row.get('sentiment', '')}
- Trend kavramlar: {row.get('trend_matches', '')}
"""
            # Assistant cevabı: sadece başlık
            assistant_content = f"{row.get('title', '')}"
            metadata = {
                "video_id": video_id,
                "title": row.get("title", ""),
                "date": row.get("published_at", "")
            }
            json.dump({
                "messages": [
                    {"role": "system", "content": "Sen Fatih Çoban YouTube kanalı için başlık öneren uzman bir asistansın."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": metadata
            }, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Title training JSONL oluşturuldu: {out_jsonl}")

def create_feedback_correction_examples(
    feedback_csv="data/user_feedback.csv",
    out_jsonl="data/training/feedback_correction_data.jsonl"
):
    import pandas as pd
    import json
    import os
    if not os.path.exists(feedback_csv):
        print("Gerekli feedback dosyası bulunamadı!")
        return
    df = pd.read_csv(feedback_csv)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            user_content = f"Kullanıcı tarafından seçilen başlık: {row.get('selected_title', '')}\nOrijinal başlık: {row.get('original_title', '')}\nKullanıcı yorumu: {row.get('user_comment', '')}"
            assistant_content = row.get("generated_script", "")
            metadata = {
                "video_id": row.get("video_id", ""),
                "date": row.get("date", "")
            }
            json.dump({
                "messages": [
                    {"role": "system", "content": "Sen Fatih Çoban YouTube kanalı için kullanıcı geri bildirimlerine göre script ve başlık düzelten bir asistansın."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ],
                "metadata": metadata
            }, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Feedback correction JSONL oluşturuldu: {out_jsonl}")

if __name__ == "__main__":
    collector = TrainingDataCollector()
    if len(sys.argv) > 1 and sys.argv[1] == "--fine-tune":
        run_fine_tune()
    else:
        # UYARI: Artık hibrit eğitim verisi için create_training_examples_from_scripts fonksiyonu kullanılmalıdır.
        # collector.collect_training_data() ve collector.collect_channel_specific_data() eski yöntemdir.
        # Yeni yöntem aşağıda çağrılmaktadır:
        create_training_examples_from_scripts(
            transcript_csv="data/transcripts.csv",
            summary_csv="data/script_summaries.csv",
            out_jsonl="data/training/training_data.jsonl",
            min_length=100
        )
        # Eğitim verisi üretimi sonrası model performans tracker'ı tetikle
        subprocess.run(["python", "src/model_performance_tracker.py"]) 