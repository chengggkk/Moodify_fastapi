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
from urllib.parse import urljoin, urlparse
import cloudscraper  # pip install cloudscraper

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str
    source: str
    formatted_lyrics: str

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# Enhanced user agents with more realistic browser fingerprints
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0"
]

# Session pool for connection reuse
session_pool = []
MAX_SESSIONS = 5

def get_session():
    """Get or create a session with rotating configurations"""
    global session_pool
    
    if len(session_pool) < MAX_SESSIONS:
        # Create new session with cloudscraper for Cloudflare bypass
        session = cloudscraper.create_scraper(
            browser={
                'browser': random.choice(['chrome', 'firefox', 'safari']),
                'platform': random.choice(['windows', 'darwin', 'linux']),
                'mobile': False
            }
        )
        
        # Configure session
        session.headers.update(get_enhanced_headers())
        
        # Set random timeout
        session.timeout = random.uniform(15, 25)
        
        session_pool.append(session)
        return session
    else:
        # Rotate existing sessions
        session = random.choice(session_pool)
        session.headers.update(get_enhanced_headers())
        return session

def get_enhanced_headers():
    """Generate realistic browser headers with randomization"""
    base_headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': random.choice([
            'en-US,en;q=0.9',
            'en-US,en;q=0.9,es;q=0.8',
            'en-GB,en;q=0.9,en-US;q=0.8',
            'en-US,en;q=0.8,fr;q=0.6,es;q=0.4'
        ]),
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': random.choice(['none', 'same-origin', 'cross-site']),
        'Sec-Fetch-User': '?1',
        'Sec-CH-UA': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-CH-UA-Mobile': '?0',
        'Sec-CH-UA-Platform': f'"{random.choice(["Windows", "macOS", "Linux"])}"',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    # Randomly add/remove some headers to appear more human
    if random.random() > 0.3:
        base_headers['Referer'] = random.choice([
            'https://www.google.com/',
            'https://www.bing.com/',
            'https://duckduckgo.com/',
            'https://genius.com/'
        ])
    
    if random.random() > 0.5:
        base_headers['X-Forwarded-For'] = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
    
    return base_headers

def calculate_backoff_delay(attempt: int, base_delay: float = 2.0) -> float:
    """Calculate exponential backoff with jitter"""
    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
    return min(delay, 30)  # Cap at 30 seconds

async def fetch_with_enhanced_retry(url: str, max_retries: int = 5) -> Optional[str]:
    """Enhanced fetch with multiple anti-detection strategies"""
    
    for attempt in range(max_retries):
        try:
            # Progressive delay with jitter
            if attempt > 0:
                delay = calculate_backoff_delay(attempt)
                print(f"Waiting {delay:.2f}s before retry {attempt + 1}")
                time.sleep(delay)
            
            # Get session (rotates automatically)
            session = get_session()
            
            # For Genius specifically, try to get the page step by step
            if 'genius.com' in url:
                response = await fetch_genius_with_steps(session, url)
            else:
                response = session.get(url, timeout=random.uniform(15, 25))
            
            if response and response.status_code == 200:
                return response.text
            elif response and response.status_code == 403:
                print(f"403 Forbidden on attempt {attempt + 1} for {url}")
                
                # Try different strategies for 403
                if attempt < max_retries - 1:
                    # Strategy 1: Clear session and try with new one
                    if attempt == 1:
                        session.cookies.clear()
                        session.headers.update(get_enhanced_headers())
                    
                    # Strategy 2: Try mobile user agent
                    elif attempt == 2:
                        session.headers['User-Agent'] = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
                        
                    # Strategy 3: Try without some headers
                    elif attempt == 3:
                        headers_to_remove = ['DNT', 'Sec-CH-UA', 'Sec-CH-UA-Mobile', 'Sec-CH-UA-Platform']
                        for header in headers_to_remove:
                            session.headers.pop(header, None)
            
            elif response:
                print(f"HTTP {response.status_code} on attempt {attempt + 1}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
        except Exception as e:
            print(f"Request error on attempt {attempt + 1}: {e}")
    
    print(f"Failed to fetch {url} after {max_retries} attempts")
    return None

async def fetch_genius_with_steps(session, url: str):
    """Special handling for Genius.com with step-by-step approach"""
    try:
        # Step 1: Visit genius.com homepage first to establish session
        home_response = session.get('https://genius.com/', timeout=15)
        if home_response.status_code != 200:
            print(f"Failed to load Genius homepage: {home_response.status_code}")
        
        # Small delay
        time.sleep(random.uniform(1, 3))
        
        # Step 2: Add genius-specific headers
        session.headers.update({
            'Referer': 'https://genius.com/',
            'Origin': 'https://genius.com',
            'Host': 'genius.com'
        })
        
        # Step 3: Try to fetch the actual lyrics page
        response = session.get(url, timeout=20)
        return response
        
    except Exception as e:
        print(f"Genius step-by-step fetch error: {e}")
        return None

def extract_lyrics_with_enhanced_lxml(html_content: str, url: str) -> str:
    """Enhanced lyrics extraction with better error handling"""
    if not html_content:
        return ""
    
    try:
        # Parse HTML with lxml
        tree = html.fromstring(html_content)
        lyrics = ""
        
        # Enhanced Genius.com extraction
        if 'genius.com' in url:
            # Multiple selectors for Genius (they change frequently)
            selectors = [
                '//div[@data-lyrics-container="true"]',
                '//div[contains(@class, "Lyrics__Container")]',
                '//div[contains(@class, "lyrics")]',
                '//div[@id="lyrics-root"]',
                '//div[contains(@class, "SongPage__Section")]//div[contains(@class, "Lyrics")]'
            ]
            
            for selector in selectors:
                containers = tree.xpath(selector)
                for container in containers:
                    # Skip header/navigation containers
                    container_html = etree.tostring(container, encoding='unicode')
                    if any(skip in container_html.lower() for skip in ['header', 'nav', 'menu', 'footer']):
                        continue
                    
                    # Extract text content
                    text_content = container.text_content()
                    if len(text_content.strip()) > 50:
                        lyrics += text_content.strip() + "\n\n"
                
                if lyrics:
                    break
            
            # Fallback: try to find any div with substantial text content
            if not lyrics:
                all_divs = tree.xpath('//div')
                for div in all_divs:
                    text = div.text_content().strip()
                    # Check if this looks like lyrics (multiple lines, reasonable length)
                    if (len(text) > 200 and 
                        text.count('\n') > 5 and 
                        not any(skip in text.lower() for skip in ['cookie', 'privacy', 'advertisement', 'subscribe'])):
                        lyrics = text
                        break
        
        # Enhanced AZLyrics extraction
        elif 'azlyrics.com' in url:
            # AZLyrics puts lyrics in divs without classes sometimes
            selectors = [
                '//div[not(@class) and not(@id)]',  # Divs without class/id often contain lyrics
                '//div[contains(@style, "text-align")]',
                '//div[@class="lyrics"]'
            ]
            
            for selector in selectors:
                divs = tree.xpath(selector)
                for div in divs:
                    text = div.text_content().strip()
                    # AZLyrics lyrics are usually quite long
                    if len(text) > 300 and text.count('\n') > 8:
                        lyrics = text
                        break
                if lyrics:
                    break
        
        # Lyrics.com extraction
        elif 'lyrics.com' in url:
            selectors = [
                '//div[@id="lyric-body-text"]',
                '//pre[@id="lyric-body-text"]',
                '//div[contains(@class, "lyric-body")]'
            ]
            for selector in selectors:
                elements = tree.xpath(selector)
                for element in elements:
                    lyrics += element.text_content().strip() + "\n\n"
                if lyrics:
                    break
        
        # Generic extraction with better heuristics
        else:
            selectors = [
                '//div[contains(@class, "lyrics")]',
                '//div[contains(@id, "lyrics")]',
                '//div[contains(@class, "song-lyrics")]',
                '//pre[contains(@class, "lyrics")]',
                '//p[contains(@class, "lyrics")]',
                '//div[contains(@class, "lyric")]'
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
        
        # Final cleanup and validation
        if lyrics:
            lyrics = re.sub(r'\n\s*\n\s*\n+', '\n\n', lyrics)
            lyrics = re.sub(r'[ \t]+', ' ', lyrics)
            lyrics = lyrics.strip()
            
            # Additional validation - check if it actually looks like lyrics
            word_count = len(lyrics.split())
            line_count = lyrics.count('\n')
            
            # Lyrics should have reasonable word/line ratio
            if word_count < 50 or line_count < 4:
                return ""
            
            # Remove common non-lyrics content
            exclude_patterns = [
                r'advertisement|subscribe|cookie|privacy policy|terms of service',
                r'©.*rights reserved|copyright|all rights reserved',
                r'powered by|website by|designed by'
            ]
            
            for pattern in exclude_patterns:
                if re.search(pattern, lyrics, re.IGNORECASE):
                    # If these patterns make up significant portion, it's probably not lyrics
                    if len(re.findall(pattern, lyrics, re.IGNORECASE)) > 3:
                        continue
        
        return lyrics
        
    except Exception as e:
        print(f"Enhanced lxml extraction error: {e}")
        return ""

# Rest of your existing functions remain the same...
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
    """Main lyrics endpoint with enhanced anti-detection"""
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
        
        # Try each of the top 3 results with enhanced fetching
        for i, result in enumerate(search_results):
            url = result.get('url', '')
            print(f"Trying result {i+1}: {url}")
            
            html_content = await fetch_with_enhanced_retry(url)
            if html_content:
                lyrics = extract_lyrics_with_enhanced_lxml(html_content, url)
                if lyrics and len(lyrics) > 50:
                    source = f"enhanced_fetch_result_{i+1}"
                    print(f"Successfully extracted lyrics from {url}")
                    break
            else:
                print(f"Failed to fetch content from {url}")
        
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
        "features": ["enhanced_anti_detection", "cloudscraper", "session_rotation", "step_by_step_genius"]
    }