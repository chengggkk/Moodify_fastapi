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

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str
    source: str

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")

def normalize_string(text: str) -> str:
    """Normalize string for comparison"""
    if not text:
        return ""
    
    # Remove accents and normalize unicode
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1, str2).ratio()

def get_random_user_agent() -> str:
    """Get a random user agent to avoid blocking"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    ]
    return random.choice(user_agents)

async def search_genius_song(artist: str, title: str) -> Optional[dict]:
    """Search for song on Genius API"""
    if not GENIUS_API_KEY:
        return None
    
    try:
        search_query = f"{artist} {title}"
        url = "https://api.genius.com/search"
        headers = {
            "Authorization": f"Bearer {GENIUS_API_KEY}",
            "Accept": "application/json"
        }
        params = {"q": search_query}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("response", {}).get("hits"):
            return None
        
        # Find the best match
        normalized_artist = normalize_string(artist)
        normalized_title = normalize_string(title)
        
        best_match = None
        best_score = 0
        
        for hit in data["response"]["hits"]:
            result = hit.get("result", {})
            if not result:
                continue
                
            result_artist = normalize_string(result.get("primary_artist", {}).get("name", ""))
            result_title = normalize_string(result.get("title", ""))
            
            artist_score = calculate_similarity(normalized_artist, result_artist)
            title_score = calculate_similarity(normalized_title, result_title)
            
            # Combined score with higher weight on title
            combined_score = (artist_score * 0.4 + title_score * 0.6)
            
            if combined_score > best_score and combined_score > 0.7:
                best_score = combined_score
                best_match = result
        
        return best_match
        
    except Exception as e:
        print(f"Genius API search error: {e}")
        return None

def extract_lyrics_with_lxml(html_content: str) -> str:
    """Extract lyrics using lxml from Genius page"""
    try:
        # Parse HTML with lxml
        tree = html.fromstring(html_content)
        
        # Look for lyrics containers with the specific structure
        lyrics_containers = tree.xpath('//div[@data-lyrics-container="true"]')
        
        if not lyrics_containers:
            return ""
        
        all_lyrics = []
        
        for container in lyrics_containers:
            # Skip header containers (they usually contain metadata, not lyrics)
            if container.xpath('.//div[contains(@class, "LyricsHeader")]'):
                continue
            
            # Extract text content from paragraphs
            paragraphs = container.xpath('.//p')
            
            for p in paragraphs:
                # Get all text content including text in nested elements
                text_parts = []
                
                # Process all elements in the paragraph
                for element in p.iter():
                    if element.text:
                        text_parts.append(element.text)
                    if element.tail:
                        text_parts.append(element.tail)
                
                paragraph_text = ''.join(text_parts)
                
                if paragraph_text.strip():
                    # Clean up the text
                    clean_text = clean_lyrics_text(paragraph_text)
                    if clean_text:
                        all_lyrics.append(clean_text)
        
        # If no paragraphs found, try extracting all text from containers
        if not all_lyrics:
            for container in lyrics_containers:
                # Skip headers
                if container.xpath('.//div[contains(@class, "LyricsHeader")]'):
                    continue
                
                # Get all text content
                text_content = etree.tostring(container, method="text", encoding="unicode")
                clean_text = clean_lyrics_text(text_content)
                if clean_text:
                    all_lyrics.append(clean_text)
        
        return '\n\n'.join(all_lyrics).strip()
        
    except Exception as e:
        print(f"lxml extraction error: {e}")
        return ""

def clean_lyrics_text(text: str) -> str:
    """Clean extracted lyrics text"""
    if not text:
        return ""
    
    # Replace HTML line breaks that might have been missed
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Clean up whitespace
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove common metadata patterns
            if re.match(r'^\d+\s*Contributors?$', line, re.IGNORECASE):
                continue
            if re.match(r'^(Share|Embed|Translations?)$', line, re.IGNORECASE):
                continue
            if 'genius.com' in line.lower():
                continue
            
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

async def fetch_lyrics_from_genius(song_data: dict) -> str:
    """Fetch lyrics from Genius song page"""
    try:
        url = song_data.get("url")
        if not url:
            return ""
        
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }
        
        # Add delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        lyrics = extract_lyrics_with_lxml(response.text)
        return lyrics
        
    except Exception as e:
        print(f"Error fetching from Genius: {e}")
        return ""

async def search_with_brave(artist: str, title: str) -> Optional[str]:
    """Fallback search using Brave Search API"""
    if not BRAVE_API_KEY:
        return None
    
    try:
        search_query = f"{artist} {title} lyrics site:genius.com"
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        params = {"q": search_query, "count": 5}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("web", {}).get("results"):
            return None
        
        # Try to fetch lyrics from the first few results
        for result in data["web"]["results"][:3]:
            if "genius.com" in result.get("url", ""):
                try:
                    headers = {
                        'User-Agent': get_random_user_agent(),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    }
                    
                    time.sleep(random.uniform(1, 2))
                    
                    page_response = requests.get(result["url"], headers=headers, timeout=15)
                    page_response.raise_for_status()
                    
                    lyrics = extract_lyrics_with_lxml(page_response.text)
                    if lyrics and len(lyrics) > 50:
                        return lyrics
                        
                except Exception as e:
                    print(f"Error fetching from {result['url']}: {e}")
                    continue
        
        return None
        
    except Exception as e:
        print(f"Brave Search error: {e}")
        return None

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    """Get lyrics for a song"""
    try:
        artist = request.artist.strip()
        title = request.title.strip()
        
        if not artist or not title:
            raise HTTPException(status_code=400, detail="Artist and title are required")
        
        lyrics = ""
        source = ""
        
        # Try Genius API first
        song_data = await search_genius_song(artist, title)
        if song_data:
            lyrics = await fetch_lyrics_from_genius(song_data)
            if lyrics:
                source = "genius_api"
        
        # Fallback to Brave Search if Genius failed
        if not lyrics:
            lyrics = await search_with_brave(artist, title)
            if lyrics:
                source = "brave_search"
        
        if not lyrics:
            raise HTTPException(
                status_code=404, 
                detail=f"Lyrics not found for '{title}' by '{artist}'"
            )
        
        return LyricsResponse(
            title=title,
            artist=artist,
            lyrics=lyrics,
            source=source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "genius_api": bool(GENIUS_API_KEY),
        "brave_api": bool(BRAVE_API_KEY)
    }