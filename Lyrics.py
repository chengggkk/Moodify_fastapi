from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import lyricsgenius
import os
from typing import Optional
import re

# Router setup
lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

# Pydantic models
class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics_found: bool
    source: str = "genius"
    song_url: Optional[str] = None

# Environment variable for Genius API
GENIUS_CLIENT_ACCESS_TOKEN = os.getenv("GENIUS_API_KEY")

def get_genius_client():
    """Initialize Genius API client"""
    if not GENIUS_CLIENT_ACCESS_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="Genius API token not configured"
        )
    
    return lyricsgenius.Genius(
        GENIUS_CLIENT_ACCESS_TOKEN,
        remove_section_headers=True,
        skip_non_songs=True,
        timeout=15,
        retries=3
    )

def clean_lyrics_text(lyrics: str) -> str:
    """Clean and format lyrics text"""
    if not lyrics:
        return ""
    
    # Remove common artifacts from genius lyrics
    lyrics = re.sub(r'\d+Embed$', '', lyrics)  # Remove "123Embed" at end
    lyrics = re.sub(r'^.*?Lyrics\n', '', lyrics)  # Remove title/artist line
    lyrics = re.sub(r'\n+', '\n', lyrics)  # Remove excessive newlines
    
    return lyrics.strip()

@lyrics_router.post("/search", response_model=LyricsResponse)
async def search_lyrics(request: LyricsRequest):
    """
    Search for song lyrics using Genius API
    
    Returns information about whether lyrics were found, but does not return
    the actual lyrics content to respect copyright.
    """
    try:
        artist = request.artist.strip()
        title = request.title.strip()
        
        if not artist or not title:
            raise HTTPException(
                status_code=400, 
                detail="Both artist and title are required"
            )
        
        print(f"Searching for: {artist} - {title}")
        
        # Initialize Genius client
        genius = get_genius_client()
        
        # Search for the song
        song = genius.search_song(title, artist)
        
        if song is None:
            return LyricsResponse(
                title=title,
                artist=artist,
                lyrics_found=False,
                source="genius"
            )
        
        # Song found - return metadata only
        return LyricsResponse(
            title=song.title,
            artist=song.artist,
            lyrics_found=True,
            source="genius",
            song_url=song.url
        )
        
    except Exception as e:
        print(f"Error searching lyrics: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to search lyrics: {str(e)}"
        )

@lyrics_router.get("/artist/{artist_name}")
async def get_artist_info(artist_name: str, max_songs: int = 5):
    """
    Get basic information about an artist and their popular songs
    
    Returns song titles and metadata, but not lyrics content.
    """
    try:
        if max_songs > 20:
            max_songs = 20  # Limit to prevent abuse
        
        genius = get_genius_client()
        
        # Search for artist
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        
        if not artist:
            raise HTTPException(
                status_code=404, 
                detail=f"Artist '{artist_name}' not found"
            )
        
        # Extract song information (metadata only)
        songs_info = []
        for song in artist.songs:
            songs_info.append({
                "title": song.title,
                "url": song.url,
                "release_date": getattr(song, 'release_date', None),
                "stats": getattr(song, 'stats', {})
            })
        
        return {
            "artist_name": artist.name,
            "artist_url": artist.url,
            "song_count": len(songs_info),
            "songs": songs_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting artist info: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get artist information: {str(e)}"
        )

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "genius_api_configured": bool(GENIUS_CLIENT_ACCESS_TOKEN),
        "features": ["lyrics_search", "artist_info"]
    }

# Usage example:
"""
To use this API:

1. Install dependencies:
   pip install fastapi lyricsgenius uvicorn

2. Set your Genius API token as environment variable:
   export GENIUS_CLIENT_ACCESS_TOKEN="your_token_here"

3. Create main.py:
   from fastapi import FastAPI
   from lyrics_api import lyrics_router
   
   app = FastAPI()
   app.include_router(lyrics_router)

4. Run the server:
   uvicorn main:app --reload

5. Test endpoints:
   POST /lyrics/search
   {
     "title": "Stronger",
     "artist": "Kanye West"
   }
   
   GET /lyrics/artist/Kanye%20West?max_songs=3
"""