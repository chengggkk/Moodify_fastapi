from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
import re
import time
import random
import unicodedata
from typing import Optional
from difflib import SequenceMatcher

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str
    source: str

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")

# User agents rotation to avoid 403
USER_AGENTS = [
    # Chrome (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.118 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Safari/537.36",

    # Chrome (Mac)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Safari/537.36",

    # Firefox (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:125.0) Gecko/20100101 Firefox/125.0",

    # Firefox (Linux)
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",

    # Safari (macOS)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",

    # Edge (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.118 Safari/537.36 Edg/124.0.2478.80",

    # Mobile Chrome (Android)
    "Mozilla/5.0 (Linux; Android 14; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.118 Mobile Safari/537.36",

    # Mobile Safari (iPhone)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",

    # Mobile Firefox (Android)
    "Mozilla/5.0 (Android 14; Mobile; rv:126.0) Gecko/126.0 Firefox/126.0"
]
def normalize_text(text: str) -> str:
    """Normalize text for multilingual comparison"""
    if not text:
        return ""
    
    # Unicode normalization for CJK characters
    text = unicodedata.normalize('NFKC', text.lower().strip())
    
    # Remove punctuation but keep CJK characters
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between strings (works for multilingual)"""
    if not str1 or not str2:
        return 0.0
    
    # Basic ratio
    ratio = SequenceMatcher(None, str1, str2).ratio()
    
    # For CJK, also check character overlap
    if any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u309f' or 
           '\u30a0' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in str1 + str2):
        char_overlap = len(set(str1) & set(str2)) / max(len(set(str1)), len(set(str2)), 1)
        ratio = max(ratio, char_overlap * 0.8)
    
    return ratio

def score_match(query_artist: str, query_title: str, result_artist: str, result_title: str) -> int:
    """Simple scoring for song matching"""
    score = 0
    
    # Normalize for comparison
    norm_q_artist = normalize_text(query_artist)
    norm_q_title = normalize_text(query_title)
    norm_r_artist = normalize_text(result_artist)
    norm_r_title = normalize_text(result_title)
    
    # Artist matching
    artist_sim = calculate_similarity(norm_q_artist, norm_r_artist)
    if artist_sim > 0.9:
        score += 40
    elif artist_sim > 0.7:
        score += 30
    elif artist_sim > 0.5:
        score += 20
    
    # Title matching
    title_sim = calculate_similarity(norm_q_title, norm_r_title)
    if title_sim > 0.9:
        score += 40
    elif title_sim > 0.7:
        score += 30
    elif title_sim > 0.5:
        score += 20
    
    return score

def get_headers():
    """Enhanced headers with anti-bot bypass techniques"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Referer': 'https://www.google.com/',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        # Some sites also check these
        'TE': 'trailers'
    }

async def fetch_with_retry(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch URL with retry logic and anti-detection"""
    for attempt in range(max_retries):
        try:
            # Random delay between requests
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=get_headers(), timeout=15)
            
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403:
                print(f"403 error on attempt {attempt + 1}, retrying...")
                # Try mobile version for Genius
                if 'genius.com' in url and attempt == 1:
                    mobile_url = url.replace('genius.com', 'm.genius.com')
                    mobile_response = requests.get(mobile_url, headers=get_headers(), timeout=15)
                    if mobile_response.status_code == 200:
                        return mobile_response.text
            else:
                print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
    
    return None

def extract_lyrics_simple(html_content: str, url: str) -> str:
    """Simple lyrics extraction that works across multiple sites"""
    if not html_content:
        return ""
    
    lyrics = ""
    
    # Genius.com extraction
    if 'genius.com' in url:
        # Look for lyrics containers
        containers = re.findall(r'<div[^>]*data-lyrics-container="true"[^>]*>(.*?)</div>', html_content, re.DOTALL)
        for container in containers:
            # Skip headers
            if 'LyricsHeader' in container and len(container) < 500:
                continue
            
            # Extract text from container
            text = re.sub(r'<br\s*/?>', '\n', container)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&[a-zA-Z0-9#]+;', lambda m: {
                '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
                '&#x27;': "'", '&#39;': "'", '&nbsp;': ' '
            }.get(m.group(0), ''), text)
            
            if len(text.strip()) > 50:
                lyrics += text.strip() + "\n\n"
    
    # AZLyrics extraction
    elif 'azlyrics.com' in url:
        match = re.search(r'<!-- Usage of azlyrics\.com content.*?-->(.*?)<!-- MxM banner -->', html_content, re.DOTALL)
        if match:
            lyrics = re.sub(r'<[^>]*>', '', match.group(1)).strip()
    
    # Lyrics.com extraction
    elif 'lyrics.com' in url:
        matches = re.findall(r'<div[^>]*id="lyric-body-text"[^>]*>(.*?)</div>', html_content, re.DOTALL)
        for match in matches:
            lyrics += re.sub(r'<[^>]*>', '', match).strip() + "\n\n"
    
    # Generic extraction
    else:
        # Remove scripts and styles
        clean_html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL)
        
        # Look for div blocks with substantial text content
        divs = re.findall(r'<div[^>]*>(.*?)</div>', clean_html, re.DOTALL)
        candidates = []
        
        for div in divs:
            text = re.sub(r'<[^>]*>', '', div).strip()
            # Check if it looks like lyrics (has line breaks and reasonable length)
            if len(text) > 200 and text.count('\n') > 5:
                candidates.append(text)
        
        if candidates:
            lyrics = max(candidates, key=len)
    
    # Clean up final lyrics
    if lyrics:
        lyrics = re.sub(r'\n\s*\n\s*\n+', '\n\n', lyrics)
        lyrics = re.sub(r'[ \t]+', ' ', lyrics)
        lyrics = lyrics.strip()
    
    return lyrics

async def search_genius_api(artist: str, title: str) -> Optional[dict]:
    """Search Genius API with multilingual support"""
    if not GENIUS_API_KEY:
        return None
    
    try:
        # Try multiple search variations
        search_queries = [
            f"{artist} {title}",
            f"{title} {artist}",
            title
        ]
        
        best_match = None
        best_score = 0
        
        for query in search_queries:
            url = f"https://api.genius.com/search?q={requests.utils.quote(query)}"
            headers = {
                'Authorization': f'Bearer {GENIUS_API_KEY}',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if not response.ok:
                continue
            
            data = response.json()
            hits = data.get('response', {}).get('hits', [])
            
            for hit in hits:
                result = hit.get('result', {})
                if not result:
                    continue
                
                result_artist = result.get('primary_artist', {}).get('name', '')
                result_title = result.get('title', '')
                
                score = score_match(artist, title, result_artist, result_title)
                
                if score > best_score:
                    best_score = score
                    best_match = result
        
        # Lower threshold for multilingual content
        if best_match and best_score >= 40:  # Reduced from 70
            return best_match
        
        return None
        
    except Exception as e:
        print(f"Genius API error: {e}")
        return None

async def search_with_brave(artist: str, title: str) -> Optional[str]:
    """Search using Brave API"""
    if not BRAVE_API_KEY:
        return None
    
    try:
        # Add language-specific terms
        search_terms = "lyrics"
        if any('\u4e00' <= c <= '\u9fff' for c in artist + title):  # Chinese
            search_terms = "歌词 lyrics"
        elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in artist + title):  # Japanese
            search_terms = "歌詞 lyrics"
        elif any('\uac00' <= c <= '\ud7af' for c in artist + title):  # Korean
            search_terms = "가사 lyrics"
        
        query = f"{artist} {title} {search_terms}"
        url = f"https://api.search.brave.com/res/v1/web/search?q={requests.utils.quote(query)}"
        
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': BRAVE_API_KEY
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if not response.ok:
            return None
        
        data = response.json()
        results = data.get('web', {}).get('results', [])
        
        # Try top results from known lyrics sites
        lyrics_sites = ['genius.com', 'azlyrics.com', 'lyrics.com', 'metrolyrics.com']
        
        for result in results[:5]:
            url = result.get('url', '')
            if any(site in url for site in lyrics_sites):
                html_content = await fetch_with_retry(url)
                if html_content:
                    lyrics = extract_lyrics_simple(html_content, url)
                    if lyrics and len(lyrics) > 100:
                        return lyrics
        
        # Fallback: check descriptions for lyrics
        for result in results[:3]:
            desc = result.get('description', '')
            if len(desc) > 200 and any(word in desc.lower() for word in ['lyrics', 'verse', 'chorus', '歌词', '歌詞', '가사']):
                return desc
        
        return None
        
    except Exception as e:
        print(f"Brave search error: {e}")
        return None

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    """Main lyrics endpoint - simplified and multilingual"""
    try:
        artist = request.artist.strip()
        title = request.title.strip()
        
        if not artist or not title:
            raise HTTPException(status_code=400, detail="Artist and title are required")
        
        print(f"Searching lyrics: {artist} - {title}")
        
        lyrics = None
        source = ""
        
        # Try Genius API first
        song_data = await search_genius_api(artist, title)
        if song_data and song_data.get('url'):
            html_content = await fetch_with_retry(song_data['url'])
            if html_content:
                lyrics = extract_lyrics_simple(html_content, song_data['url'])
                if lyrics and len(lyrics) > 50:
                    source = "genius_api"
        
        # Fallback to Brave search
        if not lyrics or len(lyrics) < 50:
            lyrics = await search_with_brave(artist, title)
            if lyrics and len(lyrics) > 50:
                source = "brave_search"
        
        if not lyrics or len(lyrics) < 50:
            # Language-appropriate error message
            if any('\u4e00' <= c <= '\u9fff' for c in artist + title):
                error_msg = f"无法找到《{title}》的歌词"
            elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in artist + title):
                error_msg = f"《{title}》の歌詞が見つかりません"
            elif any('\uac00' <= c <= '\ud7af' for c in artist + title):
                error_msg = f"《{title}》의 가사를 찾을 수 없습니다"
            else:
                error_msg = f"Lyrics not found for '{title}' by {artist}"
            
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Final cleanup
        lyrics = lyrics.strip()
        
        return LyricsResponse(
            title=title,
            artist=artist,
            lyrics=lyrics,
            source=source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lyrics API error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "genius_api": bool(GENIUS_API_KEY),
        "brave_api": bool(BRAVE_API_KEY)
    }