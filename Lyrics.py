from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
import re
import time
import random
import unicodedata
from typing import Optional, List
from difflib import SequenceMatcher
from lxml import html, etree
import json

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str
    source: str
    formatted_lyrics: str  # Mistral AI formatted version

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# User agents rotation to avoid 403
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.118 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
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
            else:
                print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                
        except Exception as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
    
    return None

def extract_lyrics_with_lxml(html_content: str, url: str) -> str:
    """Extract lyrics using lxml for better parsing"""
    if not html_content:
        return ""
    
    try:
        # Parse HTML with lxml
        tree = html.fromstring(html_content)
        lyrics = ""
        
        # Genius.com extraction
        if 'genius.com' in url:
            # Look for lyrics containers using XPath
            lyrics_containers = tree.xpath('//div[@data-lyrics-container="true"]')
            for container in lyrics_containers:
                # Skip header containers
                if 'LyricsHeader' in etree.tostring(container, encoding='unicode'):
                    continue
                
                # Extract text content
                text_content = container.text_content()
                if len(text_content.strip()) > 50:
                    lyrics += text_content.strip() + "\n\n"
        
        # AZLyrics extraction
        elif 'azlyrics.com' in url:
            # Find the main lyrics div
            lyrics_divs = tree.xpath('//div[contains(@class, "lyrics") or contains(@style, "text-align")]')
            for div in lyrics_divs:
                text = div.text_content().strip()
                if len(text) > 200:
                    lyrics += text + "\n\n"
        
        # Lyrics.com extraction
        elif 'lyrics.com' in url:
            lyrics_divs = tree.xpath('//div[@id="lyric-body-text"]')
            for div in lyrics_divs:
                lyrics += div.text_content().strip() + "\n\n"
        
        # LyricsFind extraction
        elif 'lyricsFind.com' in url.lower():
            lyrics_divs = tree.xpath('//div[@class="lyrics"]')
            for div in lyrics_divs:
                lyrics += div.text_content().strip() + "\n\n"
        
        # Generic extraction for other sites
        else:
            # Try common lyrics selectors
            selectors = [
                '//div[contains(@class, "lyrics")]',
                '//div[contains(@id, "lyrics")]',
                '//div[contains(@class, "song-lyrics")]',
                '//pre[contains(@class, "lyrics")]',
                '//p[contains(@class, "lyrics")]'
            ]
            
            for selector in selectors:
                elements = tree.xpath(selector)
                for element in elements:
                    text = element.text_content().strip()
                    if len(text) > 100 and text.count('\n') > 3:
                        lyrics += text + "\n\n"
                        break
                if lyrics:
                    break
        
        # Clean up final lyrics
        if lyrics:
            lyrics = re.sub(r'\n\s*\n\s*\n+', '\n\n', lyrics)
            lyrics = re.sub(r'[ \t]+', ' ', lyrics)
            lyrics = lyrics.strip()
        
        return lyrics
        
    except Exception as e:
        print(f"lxml extraction error: {e}")
        return ""

def format_lyrics_mistral(lyrics: str, title: str, artist: str) -> str:
    """Format lyrics in Mistral AI format"""
    if not lyrics:
        return ""
    
    # Create structured format
    formatted = {
        "song_info": {
            "title": title,
            "artist": artist,
        },
        "lyrics": {
            "sections": []
        }
    }
    
    # Split lyrics into sections
    sections = lyrics.split('\n\n')
    current_section = None
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # Identify section types
        section_lower = section.lower()
        if any(marker in section_lower for marker in ['[verse', '[chorus', '[bridge', '[intro', '[outro', '[主歌', '[副歌', '[導歌']):
            # This is a section header
            current_section = {
                "type": section.strip('[]'),
                "lines": []
            }
            formatted["lyrics"]["sections"].append(current_section)
        else:
            # This is lyrics content
            if current_section is None:
                current_section = {
                    "type": "verse",
                    "lines": []
                }
                formatted["lyrics"]["sections"].append(current_section)
            
            # Split into lines
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            current_section["lines"].extend(lines)
    
    # Convert to formatted string
    result = f"# {title} by {artist}\n\n"
    
    for i, section in enumerate(formatted["lyrics"]["sections"]):
        if section["lines"]:
            result += f"## {section['type'].title()}\n"
            for line in section["lines"]:
                result += f"{line}\n"
            result += "\n"
    
    return result.strip()

async def search_with_brave_top3(artist: str, title: str) -> List[dict]:
    """Search using Brave API and return top 3 results"""
    if not BRAVE_API_KEY:
        return []
    
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
            return []
        
        data = response.json()
        results = data.get('web', {}).get('results', [])
        
        # Filter and prioritize lyrics sites
        lyrics_sites = ['genius.com', 'azlyrics.com', 'lyrics.com', 'metrolyrics.com', 'lyricsFind.com']
        prioritized_results = []
        other_results = []
        
        for result in results[:10]:  # Check top 10 results
            url = result.get('url', '')
            if any(site in url for site in lyrics_sites):
                prioritized_results.append(result)
            else:
                other_results.append(result)
        
        # Return top 3 (prioritize lyrics sites)
        final_results = (prioritized_results + other_results)[:3]
        return final_results
        
    except Exception as e:
        print(f"Brave search error: {e}")
        return []

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    """Main lyrics endpoint - using Brave search and lxml extraction"""
    try:
        artist = request.artist.strip()
        title = request.title.strip()
        
        if not artist or not title:
            raise HTTPException(status_code=400, detail="Artist and title are required")
        
        print(f"Searching lyrics: {artist} - {title}")
        
        lyrics = None
        source = ""
        
        # Get top 3 results from Brave search
        search_results = await search_with_brave_top3(artist, title)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No search results found")
        
        # Try each of the top 3 results
        for i, result in enumerate(search_results):
            url = result.get('url', '')
            print(f"Trying result {i+1}: {url}")
            
            html_content = await fetch_with_retry(url)
            if html_content:
                lyrics = extract_lyrics_with_lxml(html_content, url)
                if lyrics and len(lyrics) > 50:
                    source = f"brave_search_result_{i+1}"
                    break
        
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
        
        # Format lyrics using Mistral AI format
        formatted_lyrics = format_lyrics_mistral(lyrics, title, artist)
        
        # Final cleanup
        lyrics = lyrics.strip()
        
        return LyricsResponse(
            title=title,
            artist=artist,
            lyrics=lyrics,
            source=source,
            formatted_lyrics=formatted_lyrics
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
        "brave_api": bool(BRAVE_API_KEY),
        "features": ["brave_search", "lxml_extraction", "mistral_formatting"]
    }