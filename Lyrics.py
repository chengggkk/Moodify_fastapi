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
        response = await openai.ChatCompletion.acreate(
            model="gpt-4-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def extract_genius_lyrics(html_content: str) -> str:
    """Extract lyrics from Genius HTML using regex (matching original working function)"""
    try:
        # Look for lyrics containers using regex (same as original working code)
        container_regex = r'<div[^>]*data-lyrics-container="true"[^>]*>([\s\S]*?)</div>'
        containers = re.findall(container_regex, html_content, re.IGNORECASE)

        if not containers:
            return ""

        all_lyrics = []
        
        for content in containers:
            # Extract from paragraph tags if present
            p_match = re.search(r'<p[^>]*>([\s\S]*?)</p>', content)
            if p_match:
                content = p_match.group(1)

            # Clean HTML and format (exactly like original)
            cleaned = content
            cleaned = re.sub(r'<br\s*/?>', '\n', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'<[^>]+>', '', cleaned)
            cleaned = re.sub(r'&quot;', '"', cleaned)
            cleaned = re.sub(r'&#x27;|&#39;', "'", cleaned)
            cleaned = re.sub(r'&amp;', '&', cleaned)
            cleaned = re.sub(r'&nbsp;', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            
            if len(cleaned) > 50:
                all_lyrics.append(cleaned)

        result = '\n\n'.join(all_lyrics)
        result = re.sub(r'\n{3,}', '\n\n', result).strip()
        return result
        
    except Exception as e:
        print(f"Error extracting Genius lyrics: {str(e)}")
        return ""

def extract_generic_lyrics(html_content: str, url: str = "") -> str:
    """Extract lyrics from other sites using regex (matching original working function)"""
    try:
        if url and 'azlyrics.com' in url:
            match = re.search(r'<!-- Usage of azlyrics\.com content[\s\S]*?-->([\s\S]*?)<!-- MxM banner -->', html_content)
            if match:
                return re.sub(r'<[^>]*>', '', match.group(1)).strip()
        
        elif url and 'lyrics.com' in url:
            matches = re.findall(r'<div[^>]*id="lyric-body-text"[^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
            if matches:
                return '\n'.join([re.sub(r'<[^>]*>', '', m).strip() for m in matches]).strip()
        
        else:
            # Generic extraction - find largest text block (same as original)
            div_matches = re.findall(r'<div[^>]*>([\s\S]*?)</div>', html_content)
            text_blocks = []
            
            for match in div_matches:
                text = re.sub(r'<[^>]*>', '', match).strip()
                if len(text) > 200 and '\n' in text:
                    text_blocks.append(text)
            
            # Sort by length and return the longest
            if text_blocks:
                text_blocks.sort(key=len, reverse=True)
                return text_blocks[0]
            
        return ""
        
    except Exception as e:
        print(f"Error extracting generic lyrics: {str(e)}")
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
                raw_lyrics = extract_generic_lyrics(request.html_content, request.url or "")
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