from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from lxml import html, etree
import os
from typing import Optional
from difflib import SequenceMatcher
import unicodedata
import re
import time
import random
import asyncio

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

def normalize_string(text: str) -> str:
    """Normalize string for comparison - based on working Node.js code"""
    if not text:
        return ""
    
    return text.lower().replace(r'[^\w\s]', '').replace(r'\s+', ' ').strip()

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()

def levenshtein_distance(str1: str, str2: str) -> int:
    """Levenshtein distance implementation from Node.js code"""
    matrix = [[0] * (len(str1) + 1) for _ in range(len(str2) + 1)]
    
    for i in range(len(str1) + 1):
        matrix[0][i] = i
    for j in range(len(str2) + 1):
        matrix[j][0] = j
    
    for j in range(1, len(str2) + 1):
        for i in range(1, len(str1) + 1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            matrix[j][i] = min(
                matrix[j][i-1] + 1,
                matrix[j-1][i] + 1,
                matrix[j-1][i-1] + cost
            )
    
    return matrix[len(str2)][len(str1)]

def is_close_match(str1: str, str2: str) -> bool:
    """Check if two strings are close matches - from Node.js code"""
    variations = [
        (str1, str2),
        (str1.replace(' ', ''), str2.replace(' ', '')),
        (str1.replace('&', 'and'), str2.replace('&', 'and')),
        (re.sub(r'ft|feat', 'featuring', str1), re.sub(r'ft|feat', 'featuring', str2)),
    ]
    
    for v1, v2 in variations:
        if v1 == v2:
            return True
        if levenshtein_distance(v1, v2) <= 2 and abs(len(v1) - len(v2)) <= 2:
            return True
    
    return False

def is_common_artist_variation(query: str, result: str) -> bool:
    """Check for common artist name variations - from Node.js code"""
    without_the1 = re.sub(r'^the\s+', '', query, flags=re.IGNORECASE)
    without_the2 = re.sub(r'^the\s+', '', result, flags=re.IGNORECASE)
    if without_the1 == result or without_the2 == query:
        return True
    
    base_name1 = re.split(r'\s+(?:ft|feat|featuring)\s+', query, flags=re.IGNORECASE)[0]
    base_name2 = re.split(r'\s+(?:ft|feat|featuring)\s+', result, flags=re.IGNORECASE)[0]
    if base_name1 == base_name2:
        return True
    
    return False

def calculate_song_similarity(query_artist: str, query_title: str, result_artist: str, result_title: str, original_artist: str, original_title: str) -> int:
    """Calculate song similarity score - exact implementation from Node.js"""
    score = 0
    
    # Exact matches (highest priority)
    if query_artist == result_artist:
        score += 40
    elif is_close_match(query_artist, result_artist):
        score += 30
    elif result_artist in query_artist or query_artist in result_artist:
        length_ratio = min(len(query_artist), len(result_artist)) / max(len(query_artist), len(result_artist))
        score += 15 * length_ratio
    
    if query_title == result_title:
        score += 40
    elif is_close_match(query_title, result_title):
        score += 30
    elif result_title in query_title or query_title in result_title:
        length_ratio = min(len(query_title), len(result_title)) / max(len(query_title), len(result_title))
        score += 15 * length_ratio
    
    # Bonus for exact case-sensitive matches
    if original_artist.lower() == result_artist.lower():
        score += 10
    if original_title.lower() == result_title.lower():
        score += 10
    
    # Penalty for extra words
    artist_word_diff = abs(len(query_artist.split()) - len(result_artist.split()))
    title_word_diff = abs(len(query_title.split()) - len(result_title.split()))
    score -= (artist_word_diff + title_word_diff) * 5
    
    # Bonus for common artist patterns
    if is_common_artist_variation(query_artist, result_artist):
        score += 15
    
    return max(0, score)

async def fetch_content_from_url(url: str) -> Optional[str]:
    """Enhanced function to fetch raw content from URL - from Node.js code"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}")
        
        html_content = response.text
        print(f"Fetched content length: {len(html_content)} characters")
        return html_content
        
    except Exception as e:
        print(f"Error fetching from {url}: {e}")
        return None

def remove_header_sections(content: str) -> str:
    """Remove header sections - from Node.js code"""
    content = re.sub(r'<div[^>]*class="[^"]*LyricsHeader[^"]*"[^>]*>[\s\S]*?</div>(?:\s*</div>)*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<button[^>]*class="[^"]*Contributors[^"]*"[^>]*>[\s\S]*?</button>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<div[^>]*class="[^"]*Dropdown[^"]*"[^>]*>[\s\S]*?</div>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<div[^>]*class="[^"]*MetadataTooltip[^"]*"[^>]*>[\s\S]*?</div>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<svg[\s\S]*?</svg>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<div[^>]*class="[^"]*(?:Header|Title|Menu|Nav)[^"]*"[^>]*>[\s\S]*?</div>', '', content, flags=re.IGNORECASE)
    return content

def extract_lyrics_from_container(content: str) -> str:
    """Extract lyrics content from container - from Node.js code"""
    if not content:
        return ''
    
    # Look for p tag content first
    p_match = re.search(r'<p[^>]*>([\s\S]*?)</p>', content)
    if p_match:
        content = p_match.group(1)
    
    # Clean up the content
    lyrics = content
    lyrics = re.sub(r'<a[^>]*href="[^"]*"[^>]*(?:class="[^"]*ReferentFragment[^"]*"[^>]*)?>([\s\S]*?)</a>', r'\1', lyrics)
    lyrics = re.sub(r'<br\s*/?>', '\n', lyrics, flags=re.IGNORECASE)
    lyrics = re.sub(r'<[^>]+>', '', lyrics)
    lyrics = lyrics.replace('&quot;', '"')
    lyrics = lyrics.replace('&#x27;', "'").replace('&#39;', "'")
    lyrics = lyrics.replace('&amp;', '&')
    lyrics = lyrics.replace('&lt;', '<')
    lyrics = lyrics.replace('&gt;', '>')
    lyrics = lyrics.replace('&nbsp;', ' ')
    lyrics = re.sub(r'[\u200B-\u200D\uFEFF]', '', lyrics)  # Remove zero-width characters
    lyrics = re.sub(r'[ \t]+', ' ', lyrics)
    lyrics = re.sub(r'\n\s*\n\s*\n', '\n\n', lyrics)
    lyrics = re.sub(r'^\s+|\s+$', '', lyrics, flags=re.MULTILINE)
    lyrics = lyrics.strip()
    
    return lyrics

def smart_deduplicate(content: str) -> str:
    """Smart deduplication - from Node.js code"""
    lines = content.split('\n')
    result = []
    seen_content = {}
    
    for line in lines:
        trimmed = line.strip()
        
        if not trimmed:
            result.append(line)
            continue
        
        # Keep section markers
        if re.match(r'^\[[\w\s:-]+\]$', trimmed):
            result.append(line)
            continue
        
        normalized = trimmed.lower()
        count = seen_content.get(normalized, 0)
        
        if count < 3:
            seen_content[normalized] = count + 1
            result.append(line)
    
    return '\n'.join(result)

def extract_from_lyrics_containers(html_content: str) -> str:
    """Enhanced container extraction method - from Node.js code"""
    print('Extracting from lyrics containers with enhanced logic...')
    
    container_pattern = r'<div[^>]*data-lyrics-container="true"[^>]*>([\s\S]*?)</div>'
    containers = re.findall(container_pattern, html_content, re.IGNORECASE)
    
    if not containers:
        print('No lyrics containers found')
        return ''
    
    print(f'Found {len(containers)} lyrics containers')
    
    all_lyrics = []
    
    for i, container_content in enumerate(containers):
        print(f'Processing container {i + 1}, raw size: {len(container_content)}')
        
        if 'LyricsHeader' in container_content and len(container_content) < 1000:
            print(f'Container {i + 1} appears to be header, skipping')
            continue
        
        container_content = remove_header_sections(container_content)
        lyrics = extract_lyrics_from_container(container_content)
        
        if lyrics and len(lyrics) > 30:
            print(f'Container {i + 1} extracted: {len(lyrics)} chars')
            all_lyrics.append(lyrics)
    
    if not all_lyrics:
        print('No valid lyrics found in any container')
        return ''
    
    combined = '\n\n'.join(all_lyrics)
    combined = smart_deduplicate(combined)
    combined = re.sub(r'\n{3,}', '\n\n', combined).strip()
    
    print(f'Combined lyrics: {len(combined)} characters from {len(all_lyrics)} containers')
    return combined

def extract_from_specific_genius_structure(html_content: str) -> str:
    """Specific extractor for Genius HTML structure - from Node.js code"""
    print('Attempting extraction from specific Genius structure...')
    
    specific_pattern = r'<div[^>]*data-lyrics-container="true"[^>]*class="[^"]*Lyrics__Container[^"]*"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)</p>[\s\S]*?</div>'
    matches = re.findall(specific_pattern, html_content, re.IGNORECASE)
    
    if matches:
        print(f'Found {len(matches)} specific structure matches')
        
        all_lyrics = []
        for i, match in enumerate(matches):
            print(f'Processing specific match {i + 1}')
            content = match
            
            content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('&quot;', '"')
            content = content.replace('&#x27;', "'").replace('&#39;', "'")
            content = content.replace('&amp;', '&')
            content = content.replace('&lt;', '<')
            content = content.replace('&gt;', '>')
            content = content.replace('&nbsp;', ' ')
            content = re.sub(r'[ \t]+', ' ', content)
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)
            content = content.strip()
            
            print(f'Specific match {i + 1} length: {len(content)}')
            if len(content) > 50:
                all_lyrics.append(content)
        
        combined_lyrics = '\n\n'.join(all_lyrics)
        if len(combined_lyrics) > 200:
            print(f'Specific extraction successful: {len(combined_lyrics)} characters')
            return combined_lyrics
    
    return ''

async def extract_genius_content_enhanced(html_content: str) -> str:
    """Enhanced Genius content extraction - from Node.js code"""
    print('Starting enhanced Genius content extraction...')
    
    # Method 1: Specific structure extractor
    result = extract_from_specific_genius_structure(html_content)
    if result and len(result) > 200:
        print('✓ Specific structure extraction successful')
        return result
    
    # Method 2: Enhanced container extraction
    result = extract_from_lyrics_containers(html_content)
    if result and len(result) > 200:
        print('✓ Enhanced container extraction successful')
        return result
    
    print('✗ All extraction methods failed')
    return ''

async def extract_generic_content(html_content: str, url: str) -> str:
    """Generic content extraction for other sites - from Node.js code"""
    print(f'Extracting content from generic site: {url}')
    
    if 'azlyrics.com' in url:
        lyrics_match = re.search(r'<!-- Usage of azlyrics\.com content[\s\S]*?-->([\s\S]*?)<!-- MxM banner -->', html_content)
        if lyrics_match:
            return re.sub(r'<[^>]*>', '', lyrics_match.group(1)).strip()
    
    elif 'lyrics.com' in url:
        lyrics_matches = re.findall(r'<div[^>]*id="lyric-body-text"[^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
        if lyrics_matches:
            return '\n'.join([re.sub(r'<[^>]*>', '', match).strip() for match in lyrics_matches]).strip()
    
    # Generic extraction
    cleaned_html = html_content
    cleaned_html = re.sub(r'<script[\s\S]*?</script>', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'<style[\s\S]*?</style>', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'<head[\s\S]*?</head>', '', cleaned_html, flags=re.IGNORECASE)
    
    text_blocks = re.findall(r'<div[^>]*>([\s\S]*?)</div>', cleaned_html, re.IGNORECASE)
    candidates = [re.sub(r'<[^>]*>', '', block).strip() for block in text_blocks]
    candidates = [text for text in candidates if len(text) > 200 and '\n' in text]
    candidates.sort(key=len, reverse=True)
    
    return candidates[0] if candidates else ''

async def search_song_on_genius(artist: str, title: str) -> Optional[dict]:
    """Search for song on Genius API - from Node.js code"""
    if not GENIUS_API_KEY:
        return None
    
    try:
        print(f'Searching for song on Genius API: {artist} - {title}')
        
        search_query = f"{artist} {title}"
        search_url = f"https://api.genius.com/search?q={requests.utils.quote(search_query)}"
        
        headers = {
            'Authorization': f'Bearer {GENIUS_API_KEY}',
            'Accept': 'application/json'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if not response.ok:
            raise Exception(f'Genius API search failed: {response.status_code}')
        
        data = response.json()
        
        if not data.get('response') or not data['response'].get('hits') or len(data['response']['hits']) == 0:
            print('No results found on Genius API')
            return None
        
        hits = data['response']['hits']
        best_match = None
        best_score = 0
        
        # Normalize inputs for comparison
        query_artist = normalize_string(artist)
        query_title = normalize_string(title)
        
        for hit in hits:
            result = hit.get('result')
            if not result:
                continue
            
            result_artist = normalize_string(result.get('primary_artist', {}).get('name', ''))
            result_title = normalize_string(result.get('title', ''))
            
            # Calculate improved similarity score
            score = calculate_song_similarity(
                query_artist, query_title,
                result_artist, result_title,
                artist, title
            )
            
            print(f"Candidate: {result.get('primary_artist', {}).get('name', '')} - {result.get('title', '')} (Score: {score})")
            
            if score > best_score:
                best_score = score
                best_match = result
        
        # Require minimum threshold for acceptance
        MIN_SCORE_THRESHOLD = 70
        if best_match and best_score >= MIN_SCORE_THRESHOLD:
            print(f"✓ Best match: {best_match['primary_artist']['name']} - {best_match['title']} (Score: {best_score})")
            return best_match
        
        print(f"✗ No good match found. Best score: {best_score} (threshold: {MIN_SCORE_THRESHOLD})")
        return None
        
    except Exception as e:
        print(f'Genius API search error: {e}')
        return None

async def fetch_lyrics_from_genius(song_data: dict) -> str:
    """Fetch lyrics from Genius - from Node.js code"""
    try:
        print(f"Fetching lyrics from Genius for: {song_data['title']}")
        
        if not song_data.get('url'):
            raise Exception('No URL provided for song')
        
        html_content = await fetch_content_from_url(song_data['url'])
        
        if not html_content:
            raise Exception('Failed to fetch song page content')
        
        lyrics = await extract_genius_content_enhanced(html_content)
        
        if not lyrics or len(lyrics) < 100:
            raise Exception('Failed to extract sufficient lyrics from Genius page')
        
        print(f"✓ Successfully extracted lyrics from Genius ({len(lyrics)} characters)")
        return lyrics
        
    except Exception as e:
        print(f'Enhanced Genius lyrics fetch error: {e}')
        raise e

async def fetch_lyrics_with_genius_api(artist: str, song: str) -> Optional[str]:
    """Primary function to fetch lyrics using Genius API - from Node.js code"""
    try:
        print('\n=== Attempting Genius API First ===')
        
        song_data = await search_song_on_genius(artist, song)
        
        if not song_data:
            print('Song not found on Genius API, will try Brave Search as fallback')
            return None
        
        lyrics = await fetch_lyrics_from_genius(song_data)
        
        if lyrics:
            print('Successfully fetched lyrics via Genius API')
            return lyrics
        
        print('Failed to extract lyrics from Genius page, will try Brave Search as fallback')
        return None
        
    except Exception as e:
        print(f'Genius API lyrics fetch error: {e}')
        return None

async def fetch_lyrics_with_brave_api(artist: str, song: str) -> Optional[str]:
    """Fallback function using Brave Search API - from Node.js code"""
    try:
        print('\n=== Fallback: Using Brave Search API ===')
        print(f'Fetching lyrics via Brave Search API: {artist} - {song}')
        
        search_query = f"{artist} {song} lyrics"
        brave_url = f"https://api.search.brave.com/res/v1/web/search?q={requests.utils.quote(search_query)}"
        
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': BRAVE_API_KEY
        }
        
        response = requests.get(brave_url, headers=headers, timeout=10)
        
        if not response.ok:
            raise Exception(f'Brave Search API request failed: {response.status_code}')
        
        data = response.json()
        print('Brave Search API search completed')
        
        if not data.get('web') or not data['web'].get('results') or len(data['web']['results']) == 0:
            raise Exception('No search results found')
        
        raw_content = None
        
        # Try top results
        for result in data['web']['results'][:3]:
            if not result.get('url'):
                continue
            
            if any(site in result['url'] for site in ['genius.com', 'azlyrics.com', 'lyrics.com', 'metrolyrics.com', 'lyricfind.com', 'songlyrics.com']):
                try:
                    print(f"Trying result: {result['url']}")
                    html_content = await fetch_content_from_url(result['url'])
                    
                    if html_content:
                        if 'genius.com' in result['url']:
                            raw_content = await extract_genius_content_enhanced(html_content)
                        else:
                            raw_content = await extract_generic_content(html_content, result['url'])
                        
                        if raw_content and len(raw_content) > 50:
                            print(f"Successfully extracted content (length: {len(raw_content)})")
                            break
                            
                except Exception as e:
                    print(f"Failed to fetch from {result['url']}: {e}")
                    continue
        
        # Fallback: check descriptions
        if not raw_content or len(raw_content) < 50:
            print('Trying fallback: checking search result descriptions...')
            for result in data['web']['results'][:5]:
                if result.get('description') and len(result['description']) > 200:
                    if any(keyword in result['description'] for keyword in ['lyrics', 'verse', 'chorus']):
                        raw_content = result['description']
                        print('Found potential lyrics in search result description')
                        break
        
        return raw_content
        
    except Exception as e:
        print(f'Brave Search API lyrics error: {e}')
        return None

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    """Get lyrics for a song - main endpoint"""
    try:
        artist = request.artist.strip()
        title = request.title.strip()
        
        if not artist or not title:
            raise HTTPException(status_code=400, detail="Artist and title are required")
        
        print(f'\n=== Enhanced Lyrics Fetch Request ===')
        print(f'Artist: {artist}')
        print(f'Song: {title}')
        print(f'Strategy: Genius API first, then Brave Search fallback')
        print(f'=====================================\n')
        
        raw_lyrics = None
        fetch_method = ''
        
        # Step 1: Try Genius API first
        raw_lyrics = await fetch_lyrics_with_genius_api(artist, title)
        
        if raw_lyrics and len(raw_lyrics) >= 50:
            fetch_method = 'genius_api'
            print(f'✓ Successfully fetched lyrics via Genius API ({len(raw_lyrics)} chars)')
        else:
            # Step 2: Fallback to Brave Search API
            print('⚠ Genius API failed, trying Brave Search API as fallback...')
            raw_lyrics = await fetch_lyrics_with_brave_api(artist, title)
            
            if raw_lyrics and len(raw_lyrics) >= 50:
                fetch_method = 'brave_search_api'
                print(f'✓ Successfully fetched lyrics via Brave Search API ({len(raw_lyrics)} chars)')
        
        if not raw_lyrics or len(raw_lyrics) < 50:
            raise HTTPException(
                status_code=404,
                detail=f"無法獲取《{title}》的歌詞，已嘗試 Genius API 和 Brave Search API"
            )
        
        print(f'Raw lyrics fetched via {fetch_method} ({len(raw_lyrics)} chars)')
        
        # Basic cleanup for final output
        formatted_lyrics = raw_lyrics.strip()
        print(f'Final lyrics length: {len(formatted_lyrics)} chars')
        
        return LyricsResponse(
            title=title,
            artist=artist,
            lyrics=formatted_lyrics,
            source=fetch_method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f'Enhanced Lyrics API Error: {e}')
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "genius_api": bool(GENIUS_API_KEY),
        "brave_api": bool(BRAVE_API_KEY)
    }