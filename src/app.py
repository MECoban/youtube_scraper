# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import nltk
import ast
import openai
from datetime import datetime
import dotenv
from googleapiclient.discovery import build
import json
import hashlib
import logging
import plotly.express as px
import pyperclip
from dotenv import load_dotenv
from modules.script_analyzer import analyze_video
from modules.intro_analyzer import analyze_intro
from modules.script_generator import generate_script_from_analysis
from modules.title_suggester import suggest_titles
from modules.chat_assistant import collect_brief, chat_assistant_generate_all, refine_script_and_titles_with_feedback
from modules.fatih_coban import FatihCobanAnalyzer, format_fatih_style_brief
import csv

st.set_page_config(page_title="YouTube Strateji Analiz & AI Asistan", layout="wide")
st.title("YouTube Strateji Analiz & AI Metin Asistanı")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    st.warning("Wordcloud veya matplotlib kurulu değil!")

logging.basicConfig(filename='data/app_errors.log', level=logging.WARNING, format='%(asctime)s %(levelname)s:%(message)s')

# --- Ortak Konu ve Strateji Listeleri ---
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

# --- Kanal ID -> Gösterilecek İsim Eşleştirme Sözlüğü ---
CHANNEL_NAME_MAP = {
    "UCl-Zrl0QhF66lu1aGXaTbfw": "Neil Patel",
    "UCGYZP9gjRq-PiJbHQIDNL3g": "Fatih Çoban",
    "UCQ4FNww3XoNgqIlkBqEAVCg": "İman Gadzhi",
    "UCRIYuAD9fUFITaYe_007GkQ": "Iman Gadzhi Business",
    "UCJLMboBYME_CLEfwsduI0wQ": "Tony Robbins",
    "UCOnWTeJRZLVLKsZicgSLcFQ": "Dean Graziosi",
    "UCs_6DXZROU29pLvgQdCx4Ww": "Dan Lok",
    "UCUyDOdBWhC1MCxEjC46d-zw": "Alex Hormozi",
    "UCctXZhXmG-kf3tlIXgVZUlw": "GaryVee",
    "UCEpzQk829RfkBIUmwg54g9A": "Beyhan Budak",
    "UCamLstJyCa-t5gfZegxsFMw": "Colin and Samir",
    "UCFM3FPTFHj5cXRA877sZ04Q": "Gürkan Zone",
    "UCxgAuX3XZROujMmGphN_scA": "Mark Tilbury",
    "UCG7lvpEkSTaKEJJSM2grmZA": "Ozan Tarık Çepni",
    "UCmGSJVG3mCRXVOP4yZrU1Dw": "Johnny Harris",
    "UCZGNLDywn8hgzqrC9Mlz_Pw": "Tai Lopez",
    "UCoOae5nYA7VqaXzerajD0lg": "Ali Abdaal",
    "UC-b3c7kxa5vU-bnmaROgvog": "The Futur",
    "UCV2y68vJ1IGfwdyYGe6PN4Q": "Bay Kalem"
    # Eğer scrape ettiğiniz başka kanallar varsa buraya ekleyin
}

# --- Veri Yükleme Fonksiyonları (veya doğrudan yükleme) ---
@st.cache_data
def load_and_prepare_data(meta_path="data/youtube_scrape_results.csv", analysis_path="data/analysis_report.csv"):
    if not os.path.exists(meta_path):
        st.error(f"Metadata dosyası bulunamadı: {meta_path}")
        return None, None
    if not os.path.exists(analysis_path):
        st.error(f"Analiz raporu bulunamadı: {analysis_path}")
        # Allow proceeding without analysis report for channel/video selection
        analysis_df = None
    else:
        try:
            analysis_df = pd.read_csv(analysis_path)
            # Convert potential JSON strings back to dicts/lists if needed (use safe_literal_eval)
            for col in ['classic_analysis', 'gpt_analysis', 'intro_analysis', 'classic_intro_analysis', 'suggested_titles']:
                 if col in analysis_df.columns:
                      analysis_df[col] = analysis_df[col].apply(safe_literal_eval)
        except Exception as e:
            st.error(f"Analiz raporu ({analysis_path}) okunurken hata: {e}")
            analysis_df = None # Set to None if loading fails

    try:
        metadata_df = pd.read_csv(meta_path)
        # Optional: Add channel title if available (requires modification in scraper or mapping)
        # For now, use channel_id
        metadata_df['channel_display'] = metadata_df['channel_id'] # Use channel_id as display name for now
        return metadata_df, analysis_df
    except Exception as e:
        st.error(f"Metadata dosyası ({meta_path}) okunurken hata: {e}")
        return None, analysis_df

def safe_literal_eval(val):
    try:
        # Handle NaN, None, or already parsed types
        if pd.isna(val) or not isinstance(val, str):
            return val
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        # If literal_eval fails, return the original string or a placeholder
        return val # Or return {} or [] depending on expected type

# --- Ana Uygulama Akışı ---
metadata_df, analysis_df = load_and_prepare_data()

if metadata_df is None:
    st.error("Metadata yüklenemedi, uygulama devam edemiyor.")
    st.stop()

# Kanal listesini oluştururken CHANNEL_NAME_MAP (ID -> İsim) kullanalım
def get_display_channels_from_ids(df, name_map):
    if df is None or df.empty or 'channel_id' not in df.columns:
        st.warning("Metadata dosyasında 'channel_id' sütunu bulunamadı veya dosya boş.")
        return []
    
    display_names = []
    processed_ids = set() # Aynı ID için birden fazla isim eklemeyi önle
    
    for cid in df['channel_id'].dropna().unique():
        if cid not in processed_ids:
            if cid in name_map:
                display_names.append(name_map[cid])
            elif cid: # Eğer map'te yoksa ve ID boş değilse, ID'yi kullan
                display_names.append(cid)
                st.warning(f"Kanal ID '{cid}' için CHANNEL_NAME_MAP'te bir isim bulunamadı. Lütfen map'i güncelleyin.")
            processed_ids.add(cid)
            
    return sorted(list(set(display_names))) # Benzersiz ve sıralı liste

channels_for_selectbox = get_display_channels_from_ids(metadata_df, CHANNEL_NAME_MAP)

if not channels_for_selectbox:
    st.warning("Kanal listesi boş. Scraper'ı çalıştırıp veri topladığınızdan ve CHANNEL_NAME_MAP'in doğru ID'lerle dolu olduğundan emin olun.")

# --- Sekmeleri Oluşturma ---
tab1, tab2 = st.tabs(["📊 Detaylı Video Analizi", "🤖 AI Chat Asistanı"])

# --- Sekme 1: Detaylı Video Analizi --- 
with tab1:
    st.header("Scrape Edilen Videoların Detaylı Analizi")
    
    if not channels_for_selectbox:
        st.error("Analiz edilecek kanal bulunamadı. Lütfen önce veri kazıma işlemini yapın ve CHANNEL_NAME_MAP sözlüğünü doğru ID'lerle güncelleyin.")
    else:
        selected_display_channel_name = st.selectbox(
            "Analiz için bir kanal seçin:", 
            options=channels_for_selectbox, 
            key="channel_select_tab1"
        )
        
        if selected_display_channel_name:
            # Seçilen gösterim adından orijinal channel_id'yi bul (MAP üzerinden)
            original_channel_id = None
            for cid, display_name in CHANNEL_NAME_MAP.items():
                if display_name == selected_display_channel_name:
                    original_channel_id = cid
                    break
            
            # Eğer map'te bulunamadıysa (bu, map'te olmayan bir ID doğrudan gösteriliyorsa olur)
            if not original_channel_id and selected_display_channel_name in metadata_df['channel_id'].unique():
                original_channel_id = selected_display_channel_name

            if not original_channel_id:
                st.error(f"Seçilen kanal '{selected_display_channel_name}' için geçerli bir Channel ID bulunamadı. CHANNEL_NAME_MAP'i kontrol edin.")
            else:
                # Filtrele ve devam et (channel_id kullanarak)
                channel_videos_df = metadata_df[metadata_df['channel_id'] == original_channel_id].copy()
                channel_videos_df.sort_values(by='published_at', ascending=False, inplace=True)
                st.subheader(f"{selected_display_channel_name} Kanalındaki Videolar") # Gösterim adı kullanılır
                
                video_id_to_analyze = None
                if not channel_videos_df.empty:
                    video_options_list = []
                    video_id_map = {}
                    for index, row in channel_videos_df.iterrows():
                        try:
                            view_count = int(row.get('view_count', 0))
                        except (ValueError, TypeError):
                            view_count = 0
                        published_at_display = str(row.get('published_at', ''))[:10]
                        label = f"{row.get('title', '[Başlıksız Video]')} ({view_count:,} izlenme) - {published_at_display}" 
                        video_options_list.append(label)
                        video_id_map[label] = row['video_id']
                    
                    if video_options_list:
                        selected_video_label = st.selectbox(
                            "Analiz için bir video seçin:", 
                            options=video_options_list, 
                            key=f"video_select_{selected_display_channel_name.replace(' ', '_').replace('(','_').replace(')','_')}"
                        )
                        if selected_video_label:
                            video_id_to_analyze = video_id_map[selected_video_label]
                            selected_row = channel_videos_df[channel_videos_df['video_id'] == video_id_to_analyze].iloc[0]
                            thumb_url = selected_row.get('thumbnail_url', '')
                            if thumb_url:
                                st.image(thumb_url, width=320, caption=selected_row.get('title', ''))
                    else:
                        st.info("Bu kanal için gösterilecek uygun video bulunamadı (filtreleri kontrol edin).")
                else:
                    st.info(f"'{selected_display_channel_name}' kanalında (filtrelenmiş) video bulunamadı.")

                # 3. Analiz Gösterimi
                if video_id_to_analyze and analysis_df is not None:
                    analysis_row = analysis_df[analysis_df['video_id'] == video_id_to_analyze]
                    
                    if not analysis_row.empty:
                        analysis_data = analysis_row.iloc[0]
                        st.subheader(f"Analiz Sonuçları: {analysis_data.get('title', video_id_to_analyze)}")
                        
                        # Analizleri sekmelerle göster
                        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                            "🤖 GPT Analizi", "📊 Klasik Analiz", "⏱️ Intro Analizi", "✍️ Üretilen Script", "💡 Başlık Önerileri"
                        ])
                        
                        with analysis_tab1:
                            st.markdown("**Strateji ve Yapı Analizi (GPT):**")
                            # Display GPT analysis safely
                            gpt_analysis_content = analysis_data.get('gpt_analysis', 'Analiz bulunamadı.')
                            if isinstance(gpt_analysis_content, dict):
                                st.json(gpt_analysis_content)
                            elif isinstance(gpt_analysis_content, str):
                                 st.markdown(gpt_analysis_content) # Assume markdown format
                            else:
                                st.write(gpt_analysis_content)

                            st.markdown("**AI Özeti:**")
                            ai_summary = analysis_data.get('ai_summary', 'Özet bulunamadı.')
                            st.markdown(ai_summary)
                            
                        with analysis_tab2:
                            st.markdown("**Klasik Analiz Sonuçları:**")
                            classic_analysis_content = analysis_data.get('classic_analysis', {})
                            if isinstance(classic_analysis_content, dict):
                                st.json(classic_analysis_content)
                            else:
                                 st.write(classic_analysis_content) # Fallback
                                 
                        with analysis_tab3:
                            st.markdown("**Intro Analizi (GPT - İlk ~1 dk):**")
                            intro_gpt = analysis_data.get('intro_analysis', 'Analiz bulunamadı.')
                            if isinstance(intro_gpt, dict):
                                st.json(intro_gpt)
                            elif isinstance(intro_gpt, str):
                                 st.markdown(intro_gpt)
                            else:
                                st.write(intro_gpt)
                                
                            st.markdown("**Klasik Intro Analizi:**")
                            intro_classic = analysis_data.get('classic_intro_analysis', {})
                            if isinstance(intro_classic, dict):
                                st.json(intro_classic)
                            else:
                                st.write(intro_classic)

                        with analysis_tab4:
                            st.markdown("**Analize Göre Üretilen Script:**")
                            # --- YENİ: Analizlerden zengin prompt oluştur ---
                            gpt_analysis = analysis_data.get('gpt_analysis', '')
                            classic_analysis = analysis_data.get('classic_analysis', '')
                            ai_summary = analysis_data.get('ai_summary', '')
                            intro_analysis = analysis_data.get('intro_analysis', '')
                            # Promptu markdown olarak birleştir
                            analysis_prompt = f"""
[GPT Analizi]
{gpt_analysis}

[Klasik Analiz]
{classic_analysis}

[AI Özeti]
{ai_summary}

[Intro Analizi]
{intro_analysis}
"""
                            # Script üretimini analiz odaklı prompt ile yap
                            try:
                                generated_script = generate_script_from_analysis(analysis_prompt)
                            except Exception as e:
                                generated_script = f"Script üretilemedi: {e}"
                            st.text_area("Script", generated_script, height=400, key=f"script_{video_id_to_analyze}")
                            if generated_script != 'Script bulunamadı.' and generated_script != "N/A (GPT Analysis Error)":
                                 pyperclip.copy(generated_script)
                                 if st.button("Scripti Kopyala", key=f"copy_{video_id_to_analyze}"):
                                      st.success("Script panoya kopyalandı!")

                        with analysis_tab5:
                            st.markdown("**Analize Göre Başlık Önerileri:**")
                            suggested_titles_list = analysis_data.get('suggested_titles', ['Öneri bulunamadı.'])
                            if isinstance(suggested_titles_list, list):
                                for title_suggestion in suggested_titles_list:
                                    st.write(f"- {title_suggestion}")
                            else:
                                st.write(suggested_titles_list) # Fallback if it's a string
                                
                    elif analysis_df is None:
                         st.warning("Analiz raporu yüklenemediği için bu videonun analizi gösterilemiyor.")
                    else:
                        st.warning(f"Bu video ({video_id_to_analyze}) için analiz raporunda veri bulunamadı.")
                elif video_id_to_analyze:
                     st.info("Analiz raporu yüklenmedi veya bulunamadı. Video analizi gösterilemiyor.")

# --- Helper function to save feedback ---
def save_chat_feedback(brief, titles, script, feedback, base_path="data"):
    feedback_file = os.path.join(base_path, "chat_assistant_feedback.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "timestamp": timestamp,
        "brief": brief,
        "suggested_titles": titles,
        "generated_script": script,
        "user_feedback": feedback
    }
    df_new = pd.DataFrame([new_data])
    if os.path.exists(feedback_file):
        df_existing = pd.read_csv(feedback_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(feedback_file, index=False, encoding='utf-8-sig')
    st.success("Geri bildiriminiz kaydedildi!")

# --- Sekme 2: AI Chat Asistanı (Sohbet Arayüzü) ---
with tab2:
    st.header("AI Chat Asistanı ile Fatih Çoban Tarzında Metin Yazın")
    st.info("Konu, hedef kitle, süre gibi bilgileri vererek başlayın. AI, Fatih Çoban'ın başarılı içeriklerini analiz ederek size özel bir script ve başlıklar oluşturacak.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_step" not in st.session_state:
        st.session_state.chat_step = "awaiting_topic"
    if "brief_details" not in st.session_state:
        st.session_state.brief_details = {
            "topic": None,
            "target_audience": None,
            "video_length": None,
            "strategy": None,
            "creator": "Fatih Çoban"  # Default creator
        }

    # Initialize chat script and titles
    if "current_chat_script" not in st.session_state:
        st.session_state.current_chat_script = ""
    if "current_chat_titles" not in st.session_state:
        st.session_state.current_chat_titles = []

    # Display initial message and topic selectbox if at the start of brief collection
    if st.session_state.chat_step == "awaiting_topic":
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Merhaba! Fatih Çoban tarzında bir YouTube videosu için script ve başlıklar oluşturalım.\n\nÖncelikle, aşağıdan bir konu seçin veya kendi konunuzu yazmak için 'Diğer/Kendi Konumu Yazmak İstiyorum' seçeneğini seçin."
            })
        with st.chat_message("assistant"):
            topic_options = KONU_LISTESI + ["Diğer/Kendi Konumu Yazmak İstiyorum"]
            selected_topic_idx = st.selectbox(
                "Konu Seçiniz:",
                options=list(range(len(topic_options))),
                format_func=lambda i: topic_options[i],
                key="topic_selectbox"
            )
            if topic_options[selected_topic_idx] == "Diğer/Kendi Konumu Yazmak İstiyorum":
                custom_topic = st.text_input("Kendi konunuzu yazın:", key="custom_topic_input")
                if custom_topic:
                    st.session_state.brief_details["topic"] = custom_topic
                    st.session_state.messages.append({"role": "user", "content": custom_topic})
                    st.session_state.chat_step = "awaiting_target_audience"
                    st.rerun()
            else:
                if st.button("Devam et", key="topic_continue_btn"):
                    chosen_topic = topic_options[selected_topic_idx]
                    st.session_state.brief_details["topic"] = chosen_topic
                    st.session_state.messages.append({"role": "user", "content": chosen_topic})
                    st.session_state.chat_step = "awaiting_target_audience"
                    st.rerun()
        # Stop further chat input until topic is selected
        st.stop()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Yanıtınız...", key="chat_input_main"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            ai_response = ""
            
            try:
                if st.session_state.chat_step == "awaiting_topic":
                    selected_topic = ""
                    try:
                        selection = int(prompt)
                        if 1 <= selection <= len(KONU_LISTESI):
                            selected_topic = KONU_LISTESI[selection-1]
                        else: # Number out of range, treat as custom
                            selected_topic = prompt
                    except ValueError: # Not a number, treat as custom
                        selected_topic = prompt
                    
                    st.session_state.brief_details["topic"] = selected_topic
                    ai_response = f"Anladım. Konu: **{selected_topic}**.\n\nŞimdi videonuzun hedef kitlesi kimler olacak?"
                    st.session_state.chat_step = "awaiting_target_audience"

                elif st.session_state.chat_step == "awaiting_target_audience":
                    st.session_state.brief_details["target_audience"] = prompt
                    ai_response = f"Hedef kitle: **{st.session_state.brief_details['target_audience']}**.\n\nVideonun tahmini uzunluğu ne kadar olmalı? (Örn: 5-7 dakika, 10 dakika)"
                    st.session_state.chat_step = "awaiting_video_length"

                elif st.session_state.chat_step == "awaiting_video_length":
                    st.session_state.brief_details["video_length"] = prompt
                    video_length_display = st.session_state.brief_details['video_length']
                    ai_response = f"Video uzunluğu: **{video_length_display}**.\n\n"
                    ai_response += "Kullanmak istediğiniz özel bir anlatım stratejisi var mı? Örneğin (ilk 5 tanesi gösteriliyor):\n"
                    for i, strat in enumerate(STRATEJI_LISTESI[:5]): # STRATEJI_LISTESI should be accessible
                        ai_response += f"\n{i+1}. {strat}"
                    ai_response += "\n\nBir numara seçebilir, kendi stratejinizi yazabilir veya 'yok' diyebilirsiniz."
                    st.session_state.chat_step = "awaiting_strategy"
                
                elif st.session_state.chat_step == "awaiting_strategy":
                    selected_strategy = None
                    if prompt.lower() not in ["yok", "hayır", "none", ""]:
                        try:
                            selection = int(prompt)
                            if 1 <= selection <= 5: # Based on showing 5 strategies
                                 selected_strategy = STRATEJI_LISTESI[selection-1]
                            else: # Number out of range
                                 selected_strategy = prompt
                        except ValueError: # Not a number
                            selected_strategy = prompt
                    st.session_state.brief_details["strategy"] = selected_strategy

                    ai_response = "Harika! Tüm temel bilgileri aldık. Şimdi sizin için bir script taslağı ve başlık önerileri oluşturuyorum...\n"
                    message_placeholder.markdown(ai_response + "⚙️ *AI çalışıyor... Lütfen bekleyin.*") # Show intermediate message
                    
                    # Initialize Fatih Çoban analyzer
                    fatih_analyzer = FatihCobanAnalyzer(metadata_df, analysis_df)
                    fatih_style_insights = fatih_analyzer.style_insights
                    fatih_content_strategy = fatih_analyzer.get_content_strategy()

                    # Update collect_brief function to use the new analyzer
                    def collect_brief(topic, audience, duration, style):
                        """Collect brief with Fatih Çoban's style integration"""
                        return format_fatih_style_brief(
                            topic=topic,
                            audience=audience,
                            duration=duration,
                            style=style,
                            insights=fatih_style_insights
                        )

                    # Format brief from collected details
                    formatted_brief = collect_brief(
                        topic=st.session_state.brief_details.get("topic"),
                        audience=st.session_state.brief_details.get("target_audience"),
                        duration=st.session_state.brief_details.get("video_length"),
                        style=st.session_state.brief_details.get("strategy") 
                        # 'style' in collect_brief matches 'strategy' here
                    )

                    # Call actual AI function
                    try:
                        ai_generation_result = chat_assistant_generate_all(formatted_brief)
                        
                        st.session_state.current_chat_script = ai_generation_result.get("script", "Script üretilemedi.")
                        st.session_state.current_chat_titles = ai_generation_result.get("title_suggestions", ["Başlık önerisi üretilemedi."])
                        
                        ai_response += "\n✅ Taslak script ve başlıklar hazır! Aşağıda güncellenmiş script ve başlıkları görebilirsiniz."
                        # Add script and titles to chat message
                        ai_response += f"\n\n**Script:**\n```\n{st.session_state.current_chat_script}\n```\n\n**Başlıklar:**\n" + '\n'.join([f"- {t}" for t in st.session_state.current_chat_titles])
                        st.session_state.chat_step = "refining_script"
                        
                        # Save interaction for future improvements
                        try:
                            save_chat_feedback(
                                brief=formatted_brief,
                                titles=st.session_state.current_chat_titles,
                                script=st.session_state.current_chat_script,
                                feedback=prompt
                            )
                        except Exception as save_error:
                            logging.error(f"Error saving feedback: {save_error}")
                            # Continue without stopping the process
                    except Exception as e_gen:
                        logging.error(f"Error calling chat_assistant_generate_all: {e_gen}", exc_info=True)
                        st.session_state.current_chat_script = f"Script üretirken bir hata oluştu: {e_gen}"
                        st.session_state.current_chat_titles = ["Başlık üretilemedi."]
                        ai_response += f"\n❌ Üzgünüm, AI metinleri oluşturulurken bir hata oluştu: {e_gen}"
                        # Keep in awaiting_strategy or reset, for now let user see the error and then they can try again or restart.
                        # For simplicity, we'll let it proceed to refining_script step but with error messages in content.

                elif st.session_state.chat_step == "refining_script":
                    # Gerçek AI ile iyileştirme
                    ai_response = f"Geri bildiriminiz alındı: '{prompt}'. Script ve başlıklar güncelleniyor..."
                    message_placeholder.markdown(ai_response + "⚙️ *AI çalışıyor... Lütfen bekleyin.*")
                    try:
                        formatted_brief = collect_brief(
                            topic=st.session_state.brief_details.get("topic"),
                            audience=st.session_state.brief_details.get("target_audience"),
                            duration=st.session_state.brief_details.get("video_length"),
                            style=st.session_state.brief_details.get("strategy")
                        )
                        result = refine_script_and_titles_with_feedback(
                            brief=formatted_brief,
                            current_script=st.session_state.current_chat_script,
                            current_titles=st.session_state.current_chat_titles,
                            feedback=prompt
                        )
                        st.session_state.current_chat_script = result.get("script", "Script güncellenemedi.")
                        st.session_state.current_chat_titles = result.get("title_suggestions", ["Başlık güncellenemedi."])
                        ai_response += "\n✅ Script ve başlıklar güncellendi! Aşağıda güncellenmiş script ve başlıkları görebilirsiniz."
                        # Add script and titles to chat message
                        ai_response += f"\n\n**Script:**\n```\n{st.session_state.current_chat_script}\n```\n\n**Başlıklar:**\n" + '\n'.join([f"- {t}" for t in st.session_state.current_chat_titles])
                        
                        # Save interaction for future improvements
                        try:
                            save_chat_feedback(
                                brief=formatted_brief,
                                titles=st.session_state.current_chat_titles,
                                script=st.session_state.current_chat_script,
                                feedback=prompt
                            )
                        except Exception as save_error:
                            logging.error(f"Error saving feedback: {save_error}")
                            # Continue without stopping the process
                    except Exception as e:
                        logging.error(f"Refinement AI error: {e}", exc_info=True)
                        ai_response += f"\n❌ AI ile güncelleme sırasında hata oluştu: {e}"

                else:
                    ai_response = "Bir sistem hatası oluştu, lütfen sohbeti yenileyin veya durumu geliştiriciye bildirin."
                    st.session_state.chat_step = "awaiting_topic" # Reset

                message_placeholder.markdown(ai_response)
                
                # Add title save and script copy buttons if we have content
                if st.session_state.chat_step == "refining_script" and st.session_state.current_chat_titles and st.session_state.current_chat_script:
                    with st.expander("Favori Başlık & Script Araçları", expanded=True):
                        st.subheader("Favori Başlığınızı Seçin")
                        selected_title_idx = st.radio(
                            "Favori başlığınızı seçin:",
                            options=range(len(st.session_state.current_chat_titles)),
                            format_func=lambda i: st.session_state.current_chat_titles[i],
                            key="favorite_title_radio"
                        )
                        
                        # Save favorite title to CSV
                        if st.button("Favori Başlığı Kaydet", key="save_fav_title_btn"):
                            favorite_title = st.session_state.current_chat_titles[selected_title_idx]
                            try:
                                # Save user's favorite title to a CSV
                                favorite_file = os.path.join("data", "favorite_titles.csv")
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                favorite_data = {
                                    "timestamp": timestamp,
                                    "topic": st.session_state.brief_details.get("topic", ""),
                                    "title": favorite_title,
                                    "audience": st.session_state.brief_details.get("target_audience", "")
                                }
                                
                                # Check if file exists
                                file_exists = os.path.isfile(favorite_file)
                                with open(favorite_file, "a", newline="", encoding="utf-8") as f:
                                    writer = csv.DictWriter(f, fieldnames=favorite_data.keys())
                                    if not file_exists:
                                        writer.writeheader()
                                    writer.writerow(favorite_data)
                                
                                st.success(f"Favori başlık kaydedildi! 🎯")
                            except Exception as save_error:
                                st.error(f"Başlık kaydedilirken hata oluştu: {save_error}")
                        
                        st.subheader("Script İşlemleri")
                        # Create a copy button for script
                        if st.button("Scripti Kopyala", key="copy_script_btn"):
                            try:
                                pyperclip.copy(st.session_state.current_chat_script)
                                st.success("Script panoya kopyalandı! 📋")
                            except Exception as copy_error:
                                st.error(f"Script kopyalanırken hata oluştu: {copy_error}")
                        
                        # Create a download button for script
                        script_bytes = st.session_state.current_chat_script.encode()
                        st.download_button(
                            label="Scripti İndir (TXT)",
                            data=script_bytes,
                            file_name=f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_script_btn"
                        )

            except Exception as e:
                full_response = f"Üzgünüm, bir hata oluştu: {e}"
                message_placeholder.markdown(full_response) # Use full_response here
                logging.error(f"Chat assistant error: {e}", exc_info=True)
                st.session_state.chat_step = "awaiting_topic" # Reset on error
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response}) # Use ai_response here

    # Display current script and titles
    st.divider()
    col_script, col_titles = st.columns([3, 1])
    with col_script:
        st.subheader("Mevcut Script Taslağı")
        st.text_area("Script", st.session_state.current_chat_script, height=400, key="chat_script_display_area")
        if st.session_state.current_chat_script:
             if st.button("Scripti Kopyala", key="copy_chat_script_main"):
                    pyperclip.copy(st.session_state.current_chat_script)
                    st.success("Script panoya kopyalandı!")
    with col_titles:
        st.subheader("Başlık Önerileri")
        if st.session_state.current_chat_titles:
            for title in st.session_state.current_chat_titles:
                st.write(f"- {title}")
        else:
            st.write("Henüz başlık önerisi yok.")

# Simplified NLTK download check
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
