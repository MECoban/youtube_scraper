import pandas as pd
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class FatihCobanAnalyzer:
    """Analyzer class for Fatih Çoban's content style and strategy"""
    
    def __init__(self, metadata_df: pd.DataFrame, analysis_df: pd.DataFrame):
        self.metadata_df = metadata_df
        self.analysis_df = analysis_df
        self.channel_id = "UCGYZP9gjRq-PiJbHQIDNL3g"  # Fatih Çoban's channel ID
        self.style_insights = self._analyze_style()
        
    def _analyze_style(self) -> Dict:
        """Analyze Fatih Çoban's content style from existing data"""
        fatih_videos = self.metadata_df[self.metadata_df['channel_id'] == self.channel_id]
        
        if fatih_videos.empty:
            return {}
        
        # Get analysis data for Fatih's videos
        fatih_analysis = self.analysis_df[self.analysis_df['video_id'].isin(fatih_videos['video_id'])]
        
        style_insights = {
            'common_topics': [],
            'tone': [],
            'structure': [],
            'title_patterns': [],
            'intro_patterns': [],
            'successful_videos': [],
            'content_categories': [],
            'engagement_patterns': [],
            'publishing_patterns': []
        }
        
        # Analyze titles and successful videos
        if not fatih_videos.empty:
            # Sort by view count to find most successful videos
            fatih_videos_sorted = fatih_videos.sort_values('view_count', ascending=False)
            style_insights['title_patterns'] = fatih_videos_sorted['title'].tolist()[:5]
            
            # Get details of top performing videos
            for _, video in fatih_videos_sorted.head(3).iterrows():
                style_insights['successful_videos'].append({
                    'title': video['title'],
                    'views': video['view_count'],
                    'likes': video['like_count'],
                    'comments': video['comment_count'],
                    'published_at': video['published_at']
                })
            
            # Analyze publishing patterns
            fatih_videos['published_at'] = pd.to_datetime(fatih_videos['published_at'])
            style_insights['publishing_patterns'] = {
                'average_views': fatih_videos['view_count'].mean(),
                'average_likes': fatih_videos['like_count'].mean(),
                'average_comments': fatih_videos['comment_count'].mean(),
                'publishing_frequency': self._analyze_publishing_frequency(fatih_videos)
            }
        
        # Analyze content structure from GPT analysis
        if not fatih_analysis.empty:
            for _, row in fatih_analysis.iterrows():
                gpt_analysis = row.get('gpt_analysis', {})
                if isinstance(gpt_analysis, dict):
                    if 'tone' in gpt_analysis:
                        style_insights['tone'].append(gpt_analysis['tone'])
                    if 'structure' in gpt_analysis:
                        style_insights['structure'].append(gpt_analysis['structure'])
                    if 'topics' in gpt_analysis:
                        style_insights['common_topics'].extend(gpt_analysis['topics'])
        
        return style_insights
    
    def _analyze_publishing_frequency(self, videos_df: pd.DataFrame) -> Dict:
        """Analyze video publishing frequency patterns"""
        if videos_df.empty:
            return {}
            
        videos_df = videos_df.sort_values('published_at')
        time_diffs = videos_df['published_at'].diff()
        
        return {
            'average_days_between_videos': time_diffs.mean().days,
            'most_common_publishing_day': videos_df['published_at'].dt.day_name().mode().iloc[0],
            'most_common_publishing_hour': videos_df['published_at'].dt.hour.mode().iloc[0]
        }
    
    def get_content_strategy(self) -> Dict:
        """Get comprehensive content strategy based on successful videos"""
        strategy = {
            'title_patterns': self._analyze_title_patterns(),
            'content_structure': self._analyze_content_structure(),
            'engagement_tactics': self._analyze_engagement_tactics(),
            'best_practices': self._generate_best_practices()
        }
        return strategy
    
    def _analyze_title_patterns(self) -> List[str]:
        """Analyze successful title patterns"""
        successful_titles = [video['title'] for video in self.style_insights.get('successful_videos', [])]
        patterns = []
        
        for title in successful_titles:
            # Add common patterns found in successful titles
            if 'nasıl' in title.lower():
                patterns.append("Nasıl yapılır formatı")
            if '?' in title:
                patterns.append("Soru formatı")
            if any(num in title for num in ['5', '7', '10']):
                patterns.append("Numaralı liste formatı")
            if 'sır' in title.lower() or 'gizli' in title.lower():
                patterns.append("Sır/gizli formatı")
            if 'yöntem' in title.lower() or 'teknik' in title.lower():
                patterns.append("Yöntem/teknik formatı")
            if 'başarı' in title.lower() or 'başarılı' in title.lower():
                patterns.append("Başarı hikayesi formatı")
        
        return list(set(patterns))
    
    def _analyze_content_structure(self) -> Dict:
        """Analyze successful content structure patterns"""
        structure = {
            'intro_elements': [],
            'main_content_patterns': [],
            'conclusion_elements': []
        }
        
        # Add structure analysis based on GPT analysis
        for analysis in self.style_insights.get('structure', []):
            if isinstance(analysis, dict):
                structure['intro_elements'].extend(analysis.get('intro', []))
                structure['main_content_patterns'].extend(analysis.get('main', []))
                structure['conclusion_elements'].extend(analysis.get('conclusion', []))
        
        return structure
    
    def _analyze_engagement_tactics(self) -> List[str]:
        """Analyze successful engagement tactics"""
        tactics = []
        successful_videos = self.style_insights.get('successful_videos', [])
        
        for video in successful_videos:
            # Analyze engagement patterns
            if video['comments'] > self.style_insights['publishing_patterns']['average_comments']:
                tactics.append("High comment engagement")
            if video['likes'] > self.style_insights['publishing_patterns']['average_likes']:
                tactics.append("High like engagement")
        
        return list(set(tactics))
    
    def _generate_best_practices(self) -> List[str]:
        """Generate best practices based on successful content"""
        practices = [
            "Merak uyandıran bir giriş yap",
            "Somut sayılar ve veri noktaları ekle",
            "Profesyonel ama samimi bir Türkçe ton kullan",
            "Kişisel anekdotlar ve deneyimler paylaş",
            "İzleyiciye anında değer sağlamaya odaklan",
            "Net bir çağrı-eylemi ile bitir",
            "Gerçek dünya örnekleri ve vaka çalışmaları kullan",
            "Karmaşık konuları anlaşılır parçalara böl",
            "Her bölümde uygulanabilir adımlar ve çıkarımlar sun",
            "Tutarlı marka kimliği ve mesajlaşma kullan"
        ]
        return practices

def generate_fatih_style_prompt(topic: str, audience: str, duration: str, style: Optional[str] = None) -> str:
    """Generate a prompt that follows Fatih Çoban's content style"""
    base_prompt = f"""
    Konu: {topic}
    Hedef Kitle: {audience}
    Süre: {duration}
    Stil: {style if style else 'Fatih Çoban tarzı'}
    
    İçerik Stil Kılavuzu (Fatih Çoban'ın başarılı içeriklerine dayanarak):
    - Tüm başlıklar ve içerikler yalnızca Fatih Çoban'ın YouTube kanalında yayınlanacak şekilde, onun üslubunda ve kendi anlatımıyla yazılmalı.
    - Başka kişi veya marka ismi (ör: Neil Patel, Gary Vee, vb.) başlıklarda ve içeriklerde kesinlikle geçmemeli.
    - Tüm başlıklar ve içerikler Türkçe olmalı ve Fatih Çoban'ın izleyici kitlesine hitap etmeli.
    - Doğrudan ve etkileyici bir ton kullan
    - Pratik değer ve uygulanabilir içgörülere odaklan
    - Giriş, ana noktalar ve sonuç ile net bir yapı oluştur
    - Gerçek dünya örnekleri ve kişisel deneyimler kullan
    - İş ve kişisel gelişime vurgu yap
    - Merak uyandıran bir giriş yap
    - Somut sayılar ve veri noktaları ekle
    - Net bir çağrı-eylemi ile bitir
    - Profesyonel ama samimi bir Türkçe ton kullan
    - Kişisel anekdotlar ve deneyimler paylaş
    - İzleyiciye anında değer sağlamaya odaklan
    - Karmaşık konuları anlaşılır parçalara böl
    - Her bölümde uygulanabilir adımlar ve çıkarımlar sun
    - Tutarlı marka kimliği ve mesajlaşma kullan
    """
    
    return base_prompt

def format_fatih_style_brief(topic: str, audience: str, duration: str, style: Optional[str] = None, 
                           insights: Optional[Dict] = None) -> str:
    """Format a complete brief following Fatih Çoban's style"""
    base_prompt = generate_fatih_style_prompt(topic, audience, duration, style)
    
    if insights:
        if insights.get('title_patterns'):
            base_prompt += "\n\nBaşarılı Başlık Örnekleri:\n"
            for title in insights['title_patterns'][:3]:
                base_prompt += f"- {title}\n"
        
        if insights.get('successful_videos'):
            base_prompt += "\n\nEn İyi Performans Gösteren Video Örnekleri:\n"
            for video in insights['successful_videos']:
                base_prompt += f"- {video['title']} ({video['views']:,} görüntülenme)\n"
    
    return base_prompt 