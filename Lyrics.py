from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncOpenAI
import os
from lxml import html
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
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_openai(messages, temperature=0.3):
    """Call OpenAI API for lyrics formatting"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def extract_genius_lyrics(html_content: str) -> str:
    """Extract lyrics using lxml - looking for div id=lyrics or data-lyrics-container"""
    try:
        # Parse HTML with lxml
        doc = html.fromstring(html_content)
        
        # Method 1: Look for div with id="lyrics"
        lyrics_div = doc.xpath('//div[@id="lyrics"]')
        if lyrics_div:
            lyrics_text = get_text_content(lyrics_div[0])
            if lyrics_text and len(lyrics_text.strip()) > 50:
                return lyrics_text.strip()
        
        # Method 2: Look for div with data-lyrics-container="true"
        lyrics_containers = doc.xpath('//div[@data-lyrics-container="true"]')
        if lyrics_containers:
            lyrics_text = get_text_content(lyrics_containers[0])
            if lyrics_text and len(lyrics_text.strip()) > 50:
                return lyrics_text.strip()
        
        # Method 3: Look for any div containing substantial content
        all_divs = doc.xpath('//div')
        for div in all_divs:
            text = get_text_content(div)
            if text and len(text.strip()) > 200:
                return text.strip()
        
        return ""
        
    except Exception as e:
        print(f"Error extracting lyrics with lxml: {str(e)}")
        return ""

def get_text_content(element) -> str:
    """Extract clean text content from an lxml element"""
    try:
        # Get all text content, preserving structure
        parts = []
        
        # Walk through all elements and text
        for item in element.iter():
            # Handle line breaks
            if item.tag == 'br':
                parts.append('\n')
            # Get text content
            if item.text:
                parts.append(item.text)
            if item.tail:
                parts.append(item.tail)
        
        # Join and clean
        text = ''.join(parts)
        
        # Clean up text
        text = clean_text(text)
        
        return text
        
    except Exception as e:
        print(f"Error getting text content: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """Clean extracted text"""
    if not text:
        return ""
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace but preserve intentional breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip whitespace from each line
        cleaned_line = line.strip()
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines and cleaned_lines[-1]:  # Add empty line only if previous line had content
            cleaned_lines.append('')
    
    # Join lines back
    result = '\n'.join(cleaned_lines)
    
    # Limit consecutive empty lines to 1
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')
    
    return result.strip()

def extract_generic_lyrics(html_content: str, url: str = "") -> str:
    """Extract lyrics from other sites using lxml"""
    try:
        doc = html.fromstring(html_content)
        
        if url and 'azlyrics.com' in url:
            # Look for azlyrics specific content
            lyrics_divs = doc.xpath('//div[contains(@class, "ringtone")]//following-sibling::div[1]')
            if lyrics_divs:
                return get_text_content(lyrics_divs[0])
        
        elif url and 'lyrics.com' in url:
            # Look for lyrics.com specific content
            lyrics_divs = doc.xpath('//div[@id="lyric-body-text"]')
            if lyrics_divs:
                return get_text_content(lyrics_divs[0])
        
        # Generic approach - find div with most text
        all_divs = doc.xpath('//div')
        best_candidate = ""
        max_length = 0
        
        for div in all_divs:
            text = get_text_content(div)
            if text and len(text) > max_length and len(text) > 200:
                max_length = len(text)
                best_candidate = text
        
        return best_candidate
        
    except Exception as e:
        print(f"Error extracting generic lyrics: {str(e)}")
        return ""

async def format_lyrics_with_ai(raw_lyrics: str, artist: str, song: str) -> str:
    """Format lyrics using OpenAI"""
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
        return raw_lyrics.strip()

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
        
        # Extract lyrics using lxml
        raw_lyrics = ""
        source = "lxml_extraction"
        
        if request.url and 'genius.com' in request.url:
            raw_lyrics = extract_genius_lyrics(request.html_content)
            source = "genius_lxml"
        else:
            # Try Genius extraction first, then generic
            raw_lyrics = extract_genius_lyrics(request.html_content)
            if not raw_lyrics or len(raw_lyrics) < 50:
                raw_lyrics = extract_generic_lyrics(request.html_content, request.url or "")
                source = "generic_lxml"
            else:
                source = "genius_lxml"
        
        if not raw_lyrics or len(raw_lyrics) < 50:
            print(f"Raw lyrics too short: {len(raw_lyrics) if raw_lyrics else 0} characters")
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