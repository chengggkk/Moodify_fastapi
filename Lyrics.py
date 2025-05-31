from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
from lxml import html, etree
import re
from datetime import datetime

# Initialize router
lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

# Pydantic models
class LyricsRequest(BaseModel):
    html_content: str
    artist: str
    song: str
    url: Optional[str] = None

class LyricsResponse(BaseModel):
    success: bool
    lyrics: Optional[str] = None
    artist: str
    song: str
    source: str
    length: Optional[int] = None
    timestamp: str
    error: Optional[str] = None

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

async def call_openai(messages, temperature=0.3):
    """Call OpenAI API for lyrics formatting"""
    try:
        response = await openai.responses.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
def extract_genius_lyrics(html_content: str) -> str:
    """Simplified and more accurate Genius lyrics extraction"""
    try:
        # First, look for the main lyrics container
        container_pattern = r'<div[^>]*data-lyrics-container="true"[^>]*>(.*?)</div>(?:\s*</div>)*'
        container_match = re.search(container_pattern, html_content, re.DOTALL | re.IGNORECASE)
        
        if not container_match:
            return ""
        
        container_content = container_match.group(1)
        
        # Extract lyrics from <p> tags within the container
        # This captures the main lyrics content
        p_pattern = r'<p[^>]*>(.*?)</p>'
        p_matches = re.findall(p_pattern, container_content, re.DOTALL | re.IGNORECASE)
        
        if not p_matches:
            return ""
        
        # Process the lyrics content
        lyrics_parts = []
        for p_content in p_matches:
            # Clean the content
            cleaned = clean_lyrics_content(p_content)
            if cleaned and len(cleaned.strip()) > 20:  # Only add substantial content
                lyrics_parts.append(cleaned)
        
        # Join all parts
        result = '\n\n'.join(lyrics_parts)
        
        # Final cleanup
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()
        
    except Exception as e:
        print(f"Error in simplified extraction: {str(e)}")
        return ""

def clean_lyrics_content(content: str) -> str:
    """Clean individual lyrics content"""
    # Remove links but keep their text content
    content = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', content, flags=re.DOTALL)
    
    # Remove spans but keep their text content  
    content = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', content, flags=re.DOTALL)
    
    # Convert <br> tags to newlines
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
    
    # Remove any remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Decode HTML entities
    content = re.sub(r'&quot;', '"', content)
    content = re.sub(r'&#x27;|&#39;', "'", content)
    content = re.sub(r'&amp;', '&', content)
    content = re.sub(r'&nbsp;', ' ', content)
    content = re.sub(r'&lt;', '<', content)
    content = re.sub(r'&gt;', '>', content)
    
    # Clean up whitespace
    content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs to single space
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple newlines to double newline
    content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)  # Trim each line
    
    return content.strip()

# Replace your existing extract_genius_lyrics function with this:
def extract_genius_lyrics_updated(html_content: str) -> str:
    """Updated Genius lyrics extraction - simpler and more accurate"""
    try:
        # Method 1: Look for data-lyrics-container with <p> content
        container_pattern = r'<div[^>]*data-lyrics-container="true"[^>]*>.*?<p[^>]*>(.*?)</p>.*?</div>'
        match = re.search(container_pattern, html_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            lyrics_content = match.group(1)
            cleaned = clean_lyrics_content(lyrics_content)
            if cleaned and len(cleaned) > 50:
                return cleaned
        
        # Method 2: Fallback to original container approach but simplified
        container_regex = r'<div[^>]*data-lyrics-container="true"[^>]*>(.*?)</div>'
        containers = re.findall(container_regex, html_content, re.IGNORECASE | re.DOTALL)
        
        for container in containers:
            # Look for <p> tags
            p_matches = re.findall(r'<p[^>]*>(.*?)</p>', container, re.DOTALL)
            for p_content in p_matches:
                cleaned = clean_lyrics_content(p_content)
                if cleaned and len(cleaned) > 50:
                    return cleaned
        
        return ""
        
    except Exception as e:
        print(f"Error in extraction: {str(e)}")
        return ""

async def format_lyrics_with_ai(raw_lyrics: str, artist: str, song: str) -> str:
    """Format lyrics using OpenAI (same as original)"""
    try:
        messages = [
            {
                "role": "system",
                "content": "Clean and format song lyrics while preserving all actual lyrical content. Remove only website navigation, metadata, and formatting artifacts. Keep section labels like [Verse], [Chorus]. Support multiple languages including English, Chinese, Japanese, and Korean."
            },
            {
                "role": "user", 
                "content": f'Clean these lyrics for "{song}" by "{artist}":\n\n{raw_lyrics[:15000]}'
            }
        ]
        
        formatted = await call_openai(messages)
        return formatted.strip()
        
    except Exception as e:
        print(f"AI formatting error: {str(e)}")
        
        # Basic cleanup fallback (same as original)
        cleaned = re.sub(r'<[^>]*>', '', raw_lyrics)
        cleaned = re.sub(r'&quot;', '"', cleaned)
        cleaned = re.sub(r'&#x27;|&#39;', "'", cleaned)
        cleaned = re.sub(r'&amp;', '&', cleaned)
        cleaned = re.sub(r'&nbsp;', ' ', cleaned)
        cleaned = re.sub(r'\n{4,}', '\n\n', cleaned)
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()

@lyrics_router.post("/extract", response_model=LyricsResponse)
async def extract_lyrics(request: LyricsRequest):
    """Extract and format lyrics from HTML content"""
    try:
        if not request.html_content or not request.artist or not request.song:
            raise HTTPException(
                status_code=400,
                detail="HTML content, artist, and song parameters are required"
            )
        
        print(f"Processing lyrics: {request.artist} - {request.song}")
        
        # Determine extraction method based on URL or content
        raw_lyrics = ""
        source = "html_extraction"
        
        if request.url and 'genius.com' in request.url:
            raw_lyrics = extract_genius_lyrics(request.html_content)
            source = "genius"
        else:
            # Try Genius extraction first, then generic
            raw_lyrics = extract_genius_lyrics(request.html_content)
            if not raw_lyrics or len(raw_lyrics) < 50:
                raw_lyrics = extract_genius_lyrics_updated(request.html_content, request.url or "")
                source = "generic"
            else:
                source = "genius"
        
        if not raw_lyrics or len(raw_lyrics) < 50:
            print(f"Raw lyrics too short: {len(raw_lyrics) if raw_lyrics else 0} characters")
            print(f"HTML preview: {request.html_content[:500]}...")
            raise HTTPException(
                status_code=404,
                detail=f'Could not extract lyrics for "{request.song}" by {request.artist}. Please check the HTML content.'
            )
        
        print(f"Raw lyrics extracted: {len(raw_lyrics)} characters")
        
        # Format lyrics with AI
        formatted_lyrics = await format_lyrics_with_ai(raw_lyrics, request.artist, request.song)
        
        print(f"Formatted lyrics: {len(formatted_lyrics)} characters")
        
        return LyricsResponse(
            success=True,
            lyrics=formatted_lyrics,
            artist=request.artist,
            song=request.song,
            source=source,
            length=len(formatted_lyrics),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lyrics extraction error: {str(e)}")
        return LyricsResponse(
            success=False,
            artist=request.artist,
            song=request.song,
            source="error",
            timestamp=datetime.now().isoformat(),
            error=f"Failed to extract lyrics: {str(e)}"
        )

@lyrics_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "lyrics_extraction"}