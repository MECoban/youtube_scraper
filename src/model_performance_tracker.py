# model_performance_tracker.py
import pandas as pd
import os
import json
from datetime import datetime

def track_model_performance(
    analysis_csv="data/analysis_report.csv",
    perf_csv="data/video_performance.csv",
    feedback_csv="data/user_feedback.csv",
    model_version_file="data/active_model.txt",
    out_csv="data/model_performance_log.csv"
):
    # Dosyaları yükle
    if not os.path.exists(analysis_csv) or not os.path.exists(perf_csv):
        print("Gerekli analiz veya performans dosyası bulunamadı!")
        return
    df_analysis = pd.read_csv(analysis_csv)
    df_perf = pd.read_csv(perf_csv)
    df_feedback = pd.read_csv(feedback_csv) if os.path.exists(feedback_csv) else pd.DataFrame()
    model_version = None
    if os.path.exists(model_version_file):
        with open(model_version_file, "r", encoding="utf-8") as f:
            model_version = f.read().strip()
    else:
        model_version = "unknown"

    # Ana metrikler
    avg_similarity = df_analysis["title_transcript_similarity"].mean() if "title_transcript_similarity" in df_analysis else None
    avg_trend_matches = df_analysis["trend_matches"].apply(lambda x: len(str(x).split(",")) if pd.notnull(x) else 0).mean() if "trend_matches" in df_analysis else None

    # Geri bildirimli başlıkların ortalama izlenme artışı
    avg_feedback_views = None
    if not df_feedback.empty and "video_id" in df_feedback and "video_id" in df_perf:
        merged = pd.merge(df_feedback, df_perf, on="video_id", how="left")
        avg_feedback_views = merged["viewCount"].astype(float).mean() if "viewCount" in merged else None

    # Zaman damgası
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Kayıt
    row = {
        "timestamp": now,
        "model_version": model_version,
        "avg_title_transcript_similarity": avg_similarity,
        "avg_trend_matches_count": avg_trend_matches,
        "avg_feedback_views": avg_feedback_views
    }
    # CSV'ye ekle
    if os.path.exists(out_csv):
        df_log = pd.read_csv(out_csv)
        df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    else:
        df_log = pd.DataFrame([row])
    df_log.to_csv(out_csv, index=False)
    print(f"✅ Model performans logu güncellendi: {out_csv}")

if __name__ == "__main__":
    track_model_performance() 