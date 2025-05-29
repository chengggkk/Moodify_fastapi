from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import os
from typing import Optional

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str  # Note: Cannot actually return copyrighted lyrics
    source: str

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    """
    Search for song lyrics using title and artist.
    
    IMPORTANT LEGAL NOTE: This is a technical template only.
    Actual implementation should not return copyrighted lyrics content.
    Consider these alternatives:
    - Return metadata about the song
    - Provide links to official sources
    - Return brief factual information about the song
    """
    
    try:
        # Step 1: Try Genius API first
        genius_result = await search_genius_api(request.title, request.artist)
        
        if genius_result:
            return LyricsResponse(
                title=request.title,
                artist=request.artist,
                lyrics="[Copyrighted content - cannot display full lyrics]",
                source="genius"
            )
        
        # Step 2: Fallback to web search if Genius fails
        search_urls = await search_web_for_lyrics(request.title, request.artist)
        
        if search_urls:
            # Step 3: Extract content (but not full lyrics due to copyright)
            processed_content = await process_lyrics_content(search_urls[0])
            
            return LyricsResponse(
                title=request.title,
                artist=request.artist,
                lyrics="[Copyrighted content - cannot display full lyrics]",
                source="web_search"
            )
        
        raise HTTPException(status_code=404, detail="Lyrics not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def search_genius_api(title: str, artist: str) -> Optional[dict]:
    """
    Search for songs using Genius API.
    You'll need to register for a Genius API token.
    """
    genius_token = os.getenv("GENIUS_API_KEY")
    if not genius_token:
        return None
    
    headers = {"Authorization": f"Bearer {genius_token}"}
    search_url = "https://api.genius.com/search"
    
    params = {
        "q": f"{title} {artist}",
        "per_page": 1
    }
    
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data.get("response", {}).get("hits"):
            return data["response"]["hits"][0]
        return None
        
    except requests.RequestException:
        return None

async def search_web_for_lyrics(title: str, artist: str) -> list:
    """
    Use Brave Search API to find potential lyrics sources.
    Returns top 3 URLs for processing.
    """
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        return []
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_api_key
    }
    
    search_query = f"{title} {artist} lyrics"
    search_url = f"https://api.search.brave.com/res/v1/web/search"
    
    params = {
        "q": search_query,
        "count": 3,
        "result_filter": "web"
    }
    
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        urls = []
        
        for result in data.get("web", {}).get("results", []):
            urls.append(result.get("url"))
        
        return urls[:3]  # Return top 3 URLs
        
    except requests.RequestException:
        return []

async def process_lyrics_content(url: str) -> str:
    """
    Extract and process content from a webpage.
    
    IMPORTANT: This function should NOT extract full copyrighted lyrics.
    Instead, consider extracting:
    - Song metadata
    - Brief descriptions
    - Links to official sources
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # PLACEHOLDER: Here you would normally process with Mistral AI
        # But should NOT return copyrighted lyrics
        processed_content = await format_with_mistral(text)
        
        return processed_content
        
    except requests.RequestException:
        return "Content extraction failed"

async def format_with_mistral(content: str) -> str:
    """
    Use Mistral AI to format content.
    
    IMPORTANT: Should not be used to format copyrighted lyrics.
    Consider using for:
    - Song information formatting
    - Metadata organization
    - Brief descriptions
    """
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        return "Mistral API not available"
    
    # Placeholder for Mistral AI integration
    # You would implement the actual Mistral API call here
    # But ensure it doesn't process copyrighted lyrics
    
    return "[Processed content - not displaying copyrighted material]"

# Alternative endpoint that provides legal alternatives
@lyrics_router.get("/info/{title}/{artist}")
async def get_song_info(title: str, artist: str):
    """
    Returns legal information about a song instead of copyrighted lyrics.
    This is a more appropriate alternative.
    """
    return {
        "title": title,
        "artist": artist,
        "message": "For lyrics, please visit official sources like the artist's website, licensed lyric sites, or purchase the song.",
        "legal_alternatives": [
            "Official artist websites",
            "Licensed streaming services",
            "Purchase from digital music stores",
            "Official music videos with lyrics"
        ]
    }