# yt_scraper.py
from googleapiclient.discovery import build
import pandas as pd
import isodate
import re
import csv
import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import json
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

# === AYARLAR ===
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - yt_scraper - %(message)s')

print("🔍 Debug: Starting script...")
load_dotenv()
API_KEY = os.getenv('API_KEY')
# print(f"👉 Yüklenen API_KEY: {API_KEY}") # Avoid printing API key directly
# print(f"📁 Current working directory: {os.getcwd()}")
# print(f"📄 .env file exists: {os.path.exists('.env')}")

# === YouTube API istemcisi ===
youtube = build('youtube', 'v3', developerKey=API_KEY)

# === Kanal Linkleri ===
channel_links = [
    "https://www.youtube.com/@neilpatel",
    "https://www.youtube.com/@fatihcoban",
    "https://www.youtube.com/@ImanGadzhi",
    "https://www.youtube.com/@ImanGadzhiBusiness",
    "https://www.youtube.com/@TonyRobbinsLive",
    "https://www.youtube.com/@deangraziosi",
    "https://www.youtube.com/@DanLok",
    "https://www.youtube.com/@AlexHormozi",
    "https://www.youtube.com/@garyvee",
    "https://www.youtube.com/@psikologBeyhanBudak",
    "https://www.youtube.com/@ColinandSamir",
    "https://www.youtube.com/@gurkanzone",
    "https://www.youtube.com/@marktilbury",
    "https://www.youtube.com/@ozantarikcom",
    "https://www.youtube.com/@johnnyharris",
    "https://www.youtube.com/@tailopez",
    "https://www.youtube.com/@aliabdaal",
    "https://www.youtube.com/@thefutur",
    "https://www.youtube.com/@Baykalem"
]

# === Kanal ID çıkarıcı ===
def get_channel_id_by_username(username):
    request = youtube.channels().list(part="id", forUsername=username)
    response = request.execute()
    items = response.get("items")
    return items[0]['id'] if items else None

def extract_channel_id_from_url(url):
    if '/channel/' in url:
        return url.split('/channel/')[1].split('/')[0]
    elif '/@' in url:
        custom_name = url.split('/@')[1].split('/')[0]
        try:
            request = youtube.search().list(part="snippet", q=custom_name, type="channel", maxResults=1)
            response = request.execute()
            items = response.get("items")
            return items[0]['snippet']['channelId'] if items else None
        except HttpError as e:
            logging.error(f"YouTube API error while fetching channel ID for {custom_name}: {e}")
            return None
    return None

def get_channel_id_cached(channel_url):
    cache_file = "channel_id_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {} # Reset cache if corrupted
    else:
        cache = {}
    if channel_url in cache:
        return cache[channel_url]
    channel_id = extract_channel_id_from_url(channel_url)
    if channel_id:
        cache[channel_url] = channel_id
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    return channel_id

# === Video verisi çekme ===
def get_channel_videos(channel_id, max_videos=100):
    video_ids = []
    next_page_token = None
    while True:
        try:
            request = youtube.search().list(
                part='id',
                channelId=channel_id,
                maxResults=min(50, max_videos - len(video_ids)), # Fetch up to 50 or remaining
                pageToken=next_page_token,
                order='date',
                type='video'
            )
            response = request.execute()
            for item in response.get('items', []):
                if item.get('id', {}).get('videoId'): # Ensure videoId exists
                    video_ids.append(item['id']['videoId'])
            next_page_token = response.get('nextPageToken')
            if not next_page_token or len(video_ids) >= max_videos:
                break
        except HttpError as e:
            logging.error(f"YouTube API error in get_channel_videos for {channel_id}: {e}")
            break # Stop if there's an API error for this channel
    return video_ids[:max_videos]

def get_video_details(video_ids):
    videos = []
    if not video_ids: # Handle empty list
        return videos
    for i in range(0, len(video_ids), 50):
        batch_ids = [vid for vid in video_ids[i:i+50] if vid] # Filter out None or empty strings
        if not batch_ids:
            continue
        try:
            request = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=','.join(batch_ids)
            )
            response = request.execute()
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                thumbnails = snippet.get('thumbnails', {})
                duration_iso = item.get('contentDetails', {}).get('duration', 'PT0S')
                videos.append({
                    'video_id': item.get('id'),
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'channel_id': snippet.get('channelId', ''),      # Retained for cross-referencing
                    'channel_title': snippet.get('channelTitle', ''), # Added channel title
                    'thumbnail_url': thumbnails.get('medium', {}).get('url', thumbnails.get('default', {}).get('url', '')), # Added thumbnail URL
                    'duration': duration_iso,
                    'liveBroadcastContent': snippet.get('liveBroadcastContent', 'none'),
                    'view_count': int(item.get('statistics', {}).get('viewCount', 0)),
                    'like_count': int(item.get('statistics', {}).get('likeCount', 0)),
                    'comment_count': int(item.get('statistics', {}).get('commentCount', 0)),
                })
        except HttpError as e:
            logging.error(f"YouTube API error in get_video_details for batch {batch_ids}: {e}")
            # Continue to next batch if one fails
    return videos

def parse_duration_to_seconds(duration_iso):
    if not duration_iso or duration_iso == 'P0D':
        return 0
    try:
        duration_timedelta = isodate.parse_duration(duration_iso)
        return int(duration_timedelta.total_seconds())
    except isodate.ISO8601Error:
        logging.warning(f"Could not parse ISO 8601 duration: {duration_iso}")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error parsing duration \"{duration_iso}\": {e}", exc_info=True)
        return 0

def get_top_and_latest(channel_id):
    video_ids = get_channel_videos(channel_id) # Fetches latest by default
    if not video_ids:
        logging.warning(f"No video IDs found for channel {channel_id}.")
        return [], []
        
    raw_details = get_video_details(video_ids)
    if not raw_details:
        logging.warning(f"No video details found for videos from channel {channel_id}.")
        return [], []

    details = [
        video for video in raw_details
        if video and video.get('liveBroadcastContent', 'none') == 'none' and \
           parse_duration_to_seconds(video.get('duration', 'PT0S')) > 180
    ]
    
    logging.info(f"Channel {channel_id}: Found {len(raw_details)} total videos initially, filtered down to {len(details)} non-live, >3 min videos.")

    if not details:
        return [], []

    top_10_videos = sorted(details, key=lambda x: x.get('view_count', 0), reverse=True)[:10]
    latest_10_videos = sorted(details, key=lambda x: x.get('published_at', ''), reverse=True)[:10] # Already sorted by date, but re-sorting filtered ensures
    
    return top_10_videos, latest_10_videos

# === Ana çalışma fonksiyonu ===
def run_for_channel_links(channel_links):
    all_data_list = []
    for link in channel_links:
        logging.info(f"⏳ Processing channel: {link}")
        cid = get_channel_id_cached(link)
        if not cid:
            logging.warning(f"❌ Channel ID could not be retrieved for: {link}")
            continue
        
        top_10, latest_10 = get_top_and_latest(cid)

        # Add source_type and append to the main list
        for video_data in top_10:
            video_data_copy = video_data.copy()
            # channel_id and channel_title are now part of video_data from get_video_details
            video_data_copy['source_type_scrape'] = 'top'
            all_data_list.append(video_data_copy)

        for video_data in latest_10:
            video_data_copy = video_data.copy()
            video_data_copy['source_type_scrape'] = 'latest'
            all_data_list.append(video_data_copy)
            
    if not all_data_list:
        return pd.DataFrame()

    df = pd.DataFrame(all_data_list)
    if df.empty:
        return df

    def combine_source_types(series):
        types = sorted(list(set(series)))
        if 'top' in types and 'latest' in types:
            return 'top_and_latest'
        return types[0] if types else ''


    # Define aggregation functions
    agg_funcs = {col: 'first' for col in df.columns if col not in ['video_id', 'source_type_scrape']}
    agg_funcs['source_type_scrape'] = combine_source_types
    
    df_processed = df.groupby('video_id', as_index=False).agg(agg_funcs)
    df_processed.rename(columns={'source_type_scrape': 'source_type'}, inplace=True)
    
    return df_processed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def get_transcript(video_id):
    try:
        # logging.debug(f"Fetching transcript for video_id: {video_id}") # Can be too verbose
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        languages_to_try = ['tr', 'en']
        
        for lang_code in languages_to_try:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang_code])
                fetched_parts = transcript.fetch()
                
                # Direct access to the text attribute of transcript parts
                full_text = ' '.join([part.text for part in fetched_parts])
                logging.info(f"Successfully fetched manually created '{lang_code}' transcript for {video_id} (length: {len(full_text)}).")
                return full_text
                    
            except NoTranscriptFound:
                logging.debug(f"No manually created '{lang_code}' transcript for {video_id}.")
            except Exception as e_manual:
                 logging.warning(f"Error fetching manually created '{lang_code}' transcript for {video_id}: {e_manual}")

        for lang_code in languages_to_try:
            try:
                transcript = transcript_list.find_generated_transcript([lang_code])
                fetched_parts = transcript.fetch()
                
                # Direct access to the text attribute of transcript parts
                full_text = ' '.join([part.text for part in fetched_parts])
                logging.info(f"Successfully fetched auto-generated '{lang_code}' transcript for {video_id} (length: {len(full_text)}).")
                return full_text
                    
            except NoTranscriptFound:
                logging.debug(f"No auto-generated '{lang_code}' transcript for {video_id}.")
            except Exception as e_generated:
                 logging.warning(f"Error fetching auto-generated '{lang_code}' transcript for {video_id}: {e_generated}")
        
        logging.warning(f"No transcript found for {video_id} after trying all specified languages and types.")
        return None
    except TranscriptsDisabled:
        logging.warning(f"Transcripts are disabled for video_id: {video_id}.")
        return None
    except Exception as e:
        logging.error(f"Failed to process transcripts for video_id: {video_id} due to: {e}", exc_info=True)
        return None

def save_data_to_csv(df, base_path="data"):
    os.makedirs(base_path, exist_ok=True)
    
    metadata_csv_path = f"{base_path}/youtube_scrape_results.csv"
    
    # Define expected columns for metadata.csv to ensure consistency
    # These columns should come from the df_processed in run_for_channel_links
    expected_metadata_cols = [
        'video_id', 'title', 'description', 'published_at', 'channel_id',
        'channel_title', 'thumbnail_url', 'duration', 'liveBroadcastContent',
        'view_count', 'like_count', 'comment_count', 'source_type'
    ]
    
    # Ensure DataFrame has all expected columns, add if missing (e.g., if df is empty or from old run)
    for col in expected_metadata_cols:
        if col not in df.columns:
            logging.warning(f"Metadata DataFrame missing expected column: {col}. Adding it with default values before saving.")
            if col in ['view_count', 'like_count', 'comment_count']:
                df[col] = 0
            else:
                df[col] = '' # Or pd.NA for newer pandas versions

    # Save only the expected columns, in the defined order
    df_to_save = df[expected_metadata_cols]
    df_to_save.to_csv(metadata_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"Video metadata saved to {metadata_csv_path} with {len(df_to_save)} rows.")
    
    # Transcript saving part
    transcripts_data = []
    if not df_to_save.empty: # Only attempt to fetch transcripts if we have videos
        logging.info(f"Starting transcript fetching for {len(df_to_save)} videos listed in metadata.")
        for index, row in df_to_save.iterrows(): # Iterate over the saved metadata
            video_id = row['video_id']
            title = row.get('title', '[Başlık Yok]') 
            transcript_text = get_transcript(video_id)
            transcripts_data.append({
                'video_id': video_id,
                'title': title, # Keep title for easier reference in transcripts.csv
                'transcript': transcript_text if transcript_text else "", 
            })
            # Removed other metadata from here as it's already in youtube_scrape_results.csv
    else:
        logging.warning("No video metadata found to process for transcripts.")
            
    if transcripts_data:
        transcripts_df = pd.DataFrame(transcripts_data)
        # Explicitly define columns for transcripts.csv
        transcript_cols_to_save = ['video_id', 'title', 'transcript']
        transcripts_df = transcripts_df[transcript_cols_to_save]
        transcript_csv_path = f"{base_path}/transcripts.csv"
        transcripts_df.to_csv(transcript_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Transcripts saved to {transcript_csv_path}. Total rows: {len(transcripts_df)}.")
    else:
        logging.warning("No transcript data was processed or prepared for saving. Creating empty transcripts.csv if it doesn't exist.")
        # Create empty transcripts.csv if it doesn't exist, to prevent file not found errors downstream
        transcript_csv_path = f"{base_path}/transcripts.csv"
        if not os.path.exists(transcript_csv_path):
            pd.DataFrame(columns=['video_id', 'title', 'transcript']).to_csv(transcript_csv_path, index=False, encoding='utf-8-sig')


# === KODU ÇALIŞTIR ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape YouTube video data and transcripts.")
    # parser.add_argument("--max_videos", type=int, default=100, help="Max videos to process per channel for latest videos.") # This is handled by get_channel_videos
    args = parser.parse_args()

    logging.info("--- Starting YouTube Scraper ---")
    df_results = run_for_channel_links(channel_links)
    if not df_results.empty:
        save_data_to_csv(df_results)
        logging.info(f"--- YouTube Scraper finished. Processed {len(df_results)} unique videos. ---")
    else:
        logging.warning("--- YouTube Scraper finished. No video data was collected. ---")
        # Ensure empty files are created if no data
        save_data_to_csv(pd.DataFrame())
