import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_FILE = 'data/title_embeddings.npy'
TITLES_FILE = 'data/title_list.csv'

model = SentenceTransformer(EMBEDDING_MODEL)

def build_title_embeddings(analysis_csv='data/analysis_report.csv'):
    if not os.path.exists(analysis_csv):
        print(f"{analysis_csv} bulunamadı!")
        return
    df = pd.read_csv(analysis_csv)
    titles = df['title'].astype(str).tolist()
    embeddings = model.encode(titles, show_progress_bar=True)
    np.save(EMBEDDING_FILE, embeddings)
    pd.DataFrame({'title': titles}).to_csv(TITLES_FILE, index=False)
    print(f"✅ Başlık embedding'leri {EMBEDDING_FILE} dosyasına kaydedildi.")

def find_similar_titles(query, top_k=5):
    if not os.path.exists(EMBEDDING_FILE) or not os.path.exists(TITLES_FILE):
        print("Embedding veya başlık dosyası bulunamadı! Önce build_title_embeddings() çalıştırın.")
        return []
    embeddings = np.load(EMBEDDING_FILE)
    df_titles = pd.read_csv(TITLES_FILE)
    query_emb = model.encode([query])[0]
    scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = [(df_titles.iloc[i]['title'], float(scores[i])) for i in top_idx]
    return results

if __name__ == "__main__":
    # Örnek kullanım: python src/semantic_search.py "örnek başlık"
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Sorgu: {query}")
        results = find_similar_titles(query)
        for title, score in results:
            print(f"{score:.3f} - {title}")
    else:
        build_title_embeddings() 