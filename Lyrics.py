from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import lyricsgenius
import os
from typing import Optional, List
import json

# Router setup
lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

# Pydantic models
class LyricsRequest(BaseModel):
    title: str
    artist: str

class ArtistSearchRequest(BaseModel):
    artist_name: str
    max_songs: int = 5
    sort: str = "popularity"  # or "title"
    include_features: bool = False

class AlbumSearchRequest(BaseModel):
    album_name: str
    artist_name: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics_found: bool
    song_url: Optional[str] = None
    lyrics_preview: Optional[str] = None  # First few lines only
    full_lyrics_available: bool = False

class SongInfo(BaseModel):
    title: str
    url: str
    artist: str
    album: Optional[str] = None
    release_date: Optional[str] = None

class ArtistResponse(BaseModel):
    artist_name: str
    artist_url: Optional[str] = None
    songs: List[SongInfo]
    song_count: int

class AlbumResponse(BaseModel):
    album_name: str
    artist_name: str
    album_url: Optional[str] = None
    tracks: List[SongInfo]
    track_count: int

# Initialize Genius client - will look for GENIUS_ACCESS_TOKEN env var
def get_genius_client():
    """Initialize Genius API client using environment variable"""
    try:
        # This will automatically look for GENIUS_ACCESS_TOKEN environment variable
        genius = lyricsgenius.Genius()
        
        # Configure genius client options
        genius.verbose = False  # Turn off status messages
        genius.remove_section_headers = True  # Remove section headers from lyrics
        genius.skip_non_songs = True  # Skip non-songs (track lists, etc.)
        genius.excluded_terms = ["(Remix)", "(Live)", "(Instrumental)"]  # Exclude these terms
        genius.timeout = 15
        genius.retries = 3
        
        return genius
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Genius client. Make sure GENIUS_ACCESS_TOKEN is set. Error: {str(e)}"
        )

def get_lyrics_preview(lyrics: str, max_lines: int = 4) -> str:
    """Get first few lines of lyrics for preview (respecting copyright)"""
    if not lyrics:
        return ""
    
    lines = lyrics.split('\n')
    preview_lines = [line.strip() for line in lines[:max_lines] if line.strip()]
    preview = '\n'.join(preview_lines)
    
    if len(lines) > max_lines:
        preview += "\n[... more lyrics available]"
    
    return preview

@lyrics_router.post("/search", response_model=LyricsResponse)
async def search_song(request: LyricsRequest):
    """
    Search for a specific song using the official lyricsgenius method
    """
    try:
        genius = get_genius_client()
        
        # Use the official search_song method as shown in docs
        song = genius.search_song(request.title, request.artist)
        
        if song is None:
            return LyricsResponse(
                title=request.title,
                artist=request.artist,
                lyrics_found=False,
                full_lyrics_available=False
            )
        
        # Provide preview only to respect copyright
        lyrics_preview = get_lyrics_preview(song.lyrics) if song.lyrics else None
        
        return LyricsResponse(
            title=song.title,
            artist=song.artist,
            lyrics_found=True,
            song_url=song.url,
            lyrics_preview=lyrics_preview,
            full_lyrics_available=bool(song.lyrics)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for song: {str(e)}"
        )

@lyrics_router.post("/artist", response_model=ArtistResponse)
async def search_artist(request: ArtistSearchRequest):
    """
    Search for an artist and their songs using the official lyricsgenius method
    """
    try:
        if request.max_songs > 20:
            request.max_songs = 20  # Prevent abuse
        
        genius = get_genius_client()
        
        # Use the official search_artist method as shown in docs
        artist = genius.search_artist(
            request.artist_name,
            max_songs=request.max_songs,
            sort=request.sort,
            include_features=request.include_features
        )
        
        if not artist:
            raise HTTPException(
                status_code=404,
                detail=f"Artist '{request.artist_name}' not found"
            )
        
        # Extract song information
        songs_info = []
        for song in artist.songs:
            songs_info.append(SongInfo(
                title=song.title,
                url=song.url,
                artist=song.artist,
                album=getattr(song, 'album', None),
                release_date=getattr(song, 'release_date', None)
            ))
        
        return ArtistResponse(
            artist_name=artist.name,
            artist_url=getattr(artist, 'url', None),
            songs=songs_info,
            song_count=len(songs_info)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for artist: {str(e)}"
        )

@lyrics_router.post("/album", response_model=AlbumResponse)
async def search_album(request: AlbumSearchRequest):
    """
    Search for an album using the official lyricsgenius method
    """
    try:
        genius = get_genius_client()
        
        # Use the official search_album method as shown in docs
        album = genius.search_album(request.album_name, request.artist_name)
        
        if not album:
            raise HTTPException(
                status_code=404,
                detail=f"Album '{request.album_name}' by '{request.artist_name}' not found"
            )
        
        # Extract track information
        tracks_info = []
        for track in album.tracks:
            tracks_info.append(SongInfo(
                title=track.song.title,
                url=track.song.url,
                artist=track.song.artist,
                album=album.name,
                release_date=getattr(track.song, 'release_date', None)
            ))
        
        return AlbumResponse(
            album_name=album.name,
            artist_name=album.artist.name,
            album_url=getattr(album, 'url', None),
            tracks=tracks_info,
            track_count=len(tracks_info)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for album: {str(e)}"
        )

@lyrics_router.get("/artist/{artist_name}/song/{song_title}")
async def get_artist_song(artist_name: str, song_title: str):
    """
    Get a specific song from an artist using the artist.song() method
    """
    try:
        genius = get_genius_client()
        
        # First search for the artist
        artist = genius.search_artist(artist_name, max_songs=50)
        
        if not artist:
            raise HTTPException(
                status_code=404,
                detail=f"Artist '{artist_name}' not found"
            )
        
        # Use the artist.song() method as shown in docs
        song = artist.song(song_title)
        
        if not song:
            raise HTTPException(
                status_code=404,
                detail=f"Song '{song_title}' not found for artist '{artist_name}'"
            )
        
        lyrics_preview = get_lyrics_preview(song.lyrics) if song.lyrics else None
        
        return {
            "title": song.title,
            "artist": song.artist,
            "url": song.url,
            "lyrics_preview": lyrics_preview,
            "full_lyrics_available": bool(song.lyrics)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting song: {str(e)}"
        )

@lyrics_router.post("/save-artist-lyrics/{artist_name}")
async def save_artist_lyrics(artist_name: str, max_songs: int = 10):
    """
    Save artist's lyrics to JSON file using the official save_lyrics() method
    Note: This saves files to the server filesystem
    """
    try:
        if max_songs > 50:
            max_songs = 50  # Prevent abuse
        
        genius = get_genius_client()
        
        # Search for artist
        artist = genius.search_artist(artist_name, max_songs=max_songs)
        
        if not artist:
            raise HTTPException(
                status_code=404,
                detail=f"Artist '{artist_name}' not found"
            )
        
        # Use the official save_lyrics() method as shown in docs
        filename = artist.save_lyrics()
        
        return {
            "message": f"Lyrics saved successfully",
            "artist": artist.name,
            "song_count": len(artist.songs),
            "filename": filename,
            "note": "File saved to server filesystem"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving lyrics: {str(e)}"
        )

@lyrics_router.post("/save-album-lyrics")
async def save_album_lyrics(request: AlbumSearchRequest):
    """
    Save album's lyrics to JSON file using the official save_lyrics() method
    """
    try:
        genius = get_genius_client()
        
        # Search for album
        album = genius.search_album(request.album_name, request.artist_name)
        
        if not album:
            raise HTTPException(
                status_code=404,
                detail=f"Album '{request.album_name}' by '{request.artist_name}' not found"
            )
        
        # Use the official save_lyrics() method as shown in docs
        filename = album.save_lyrics()
        
        return {
            "message": f"Album lyrics saved successfully",
            "album": album.name,
            "artist": album.artist.name,
            "track_count": len(album.tracks),
            "filename": filename,
            "note": "File saved to server filesystem"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving album lyrics: {str(e)}"
        )

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    genius_configured = bool(os.getenv("GENIUS_ACCESS_TOKEN"))
    
    return {
        "status": "healthy",
        "genius_api_configured": genius_configured,
        "environment_variable": "GENIUS_ACCESS_TOKEN",
        "features": [
            "song_search",
            "artist_search", 
            "album_search",
            "lyrics_saving",
            "official_lyricsgenius_methods"
        ]
    }

# Usage example and setup instructions:
"""
Setup Instructions:

1. Install dependencies:
   pip install fastapi lyricsgenius uvicorn

2. Get Genius API token:
   - Go to https://genius.com/api-clients
   - Create new API client
   - Copy your Client Access Token

3. Set environment variable (the name that lyricsgenius expects):
   export GENIUS_ACCESS_TOKEN="your_token_here"

4. Create main.py:
   from fastapi import FastAPI
   from lyrics_api import lyrics_router
   
   app = FastAPI(title="Lyrics API")
   app.include_router(lyrics_router)

5. Run server:
   uvicorn main:app --reload

6. Example API calls:

   POST /lyrics/search
   {
     "title": "To You",
     "artist": "Andy Shauf"
   }

   POST /lyrics/artist  
   {
     "artist_name": "Andy Shauf",
     "max_songs": 3,
     "sort": "title",
     "include_features": true
   }

   POST /lyrics/album
   {
     "album_name": "The Party",
     "artist_name": "Andy Shauf"
   }

   GET /lyrics/artist/Andy%20Shauf/song/To%20You

   POST /lyrics/save-artist-lyrics/Andy%20Shauf?max_songs=5
"""