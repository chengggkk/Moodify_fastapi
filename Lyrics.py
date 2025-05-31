from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
from lxml import html
import re
from datetime import datetime
import asyncio

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

# Initialize OpenAI client - Updated for newer OpenAI library versions
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_openai(messages, temperature=0.3):
    """Call OpenAI API for lyrics formatting"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def extract_genius_lyrics_lxml(html_content: str) -> str:
    """Extract lyrics from Genius HTML using lxml"""
    try:
        tree = html.fromstring(html_content)
        
        # Look for lyrics containers with data-lyrics-container="true"
        lyrics_containers = tree.xpath('//div[@data-lyrics-container="true"]')
        
        if not lyrics_containers:
            # Fallback to other common Genius selectors
            lyrics_containers = tree.xpath('//div[contains(@class, "lyrics")]') or \
                              tree.xpath('//div[contains(@class, "Lyrics__Container")]')
        
        if not lyrics_containers:
            return ""
        
        all_lyrics = []
        
        for container in lyrics_containers:
            # Get all text content from the container
            text_content = html.tostring(container, method='text', encoding='unicode')
            
            # Clean up the text
            if text_content and len(text_content.strip()) > 50:
                # Remove extra whitespace and normalize line breaks
                cleaned = re.sub(r'\s+', ' ', text_content.strip())
                cleaned = re.sub(r'\[([^\]]+)\]', r'\n[\1]\n', cleaned)  # Format section headers
                cleaned = re.sub(r'([。！？])', r'\1\n', cleaned)  # Add line breaks after Chinese punctuation
                cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove multiple line breaks
                all_lyrics.append(cleaned.strip())
        
        result = '\n\n'.join(all_lyrics)
        return result.strip()
        
    except Exception as e:
        print(f"Error extracting lyrics with lxml: {str(e)}")
        return ""

def extract_generic_lyrics_lxml(html_content: str, url: str = "") -> str:
    """Extract lyrics from other sites using lxml"""
    try:
        tree = html.fromstring(html_content)
        
        # Site-specific extraction patterns
        if url and 'azlyrics.com' in url:
            # Look for lyrics content - AZLyrics typically uses div without class after comment
            lyrics_divs = tree.xpath('//div[not(@*)]//text()[string-length(normalize-space(.)) > 10]')
            if lyrics_divs:
                return '\n'.join([text.strip() for text in lyrics_divs if text.strip()])
        
        elif url and 'lyrics.com' in url:
            # Look for lyric-body-text div
            lyrics_elements = tree.xpath('//div[@id="lyric-body-text"]//text()')
            if lyrics_elements:
                return '\n'.join([text.strip() for text in lyrics_elements if text.strip()])
        
        elif url and 'musixmatch.com' in url:
            # Musixmatch uses specific span classes
            lyrics_elements = tree.xpath('//span[contains(@class, "lyrics__content__ok")]//text()')
            if lyrics_elements:
                return '\n'.join([text.strip() for text in lyrics_elements if text.strip()])
        
        # Generic extraction - look for largest text blocks
        text_blocks = []
        
        # Try common lyrics container patterns
        potential_containers = tree.xpath('//div[contains(@class, "lyrics") or contains(@id, "lyrics") or contains(@class, "lyric") or contains(@class, "song-text") or contains(@class, "verse")]')
        
        for container in potential_containers:
            text = html.tostring(container, method='text', encoding='unicode')
            if text and len(text.strip()) > 200:
                # Clean common website artifacts
                cleaned_text = re.sub(r'(Share|Print|Email|Download|Subscribe|Advertisement|Ad|Cookie|Privacy)', '', text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())
                if len(cleaned_text) > 100:
                    text_blocks.append(cleaned_text)
        
        # If no specific containers found, get paragraphs with substantial text
        if not text_blocks:
            paragraphs = tree.xpath('//p[string-length(normalize-space(.)) > 100]')
            for p in paragraphs:
                text = html.tostring(p, method='text', encoding='unicode')
                if text and len(text.strip()) > 100:
                    text_blocks.append(text.strip())
        
        # Return the longest text block that looks like lyrics
        if text_blocks:
            # Filter out blocks that are likely navigation/ads
            lyrics_blocks = []
            for block in text_blocks:
                # Check if block contains typical lyrics indicators
                if (re.search(r'\n.*\n', block) or  # Multiple lines
                    re.search(r'\[.*\]', block) or   # Section markers
                    len(block.split()) > 50):        # Substantial length
                    lyrics_blocks.append(block)
            
            return max(lyrics_blocks, key=len) if lyrics_blocks else max(text_blocks, key=len)
        
        return ""
        
    except Exception as e:
        print(f"Error extracting generic lyrics: {str(e)}")
        return ""

async def format_lyrics_with_ai(raw_lyrics: str, artist: str, song: str) -> str:
    """Format lyrics using OpenAI"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are a lyrics formatter. Your job is to clean and format song lyrics while preserving all actual lyrical content. 

Rules:
- Remove website navigation, ads, metadata, and formatting artifacts
- Keep all actual song lyrics intact
- Preserve section labels like [Verse], [Chorus], [Bridge], etc.
- Maintain proper line breaks and verse structure
- Support multiple languages (English, Chinese, Japanese, Korean, etc.)
- Remove duplicate lines that appear to be website artifacts
- Keep the natural flow and rhythm of the lyrics
- Don't add or change any lyrical content

Output only the cleaned lyrics, nothing else."""
            },
            {
                "role": "user", 
                "content": f'Clean these lyrics for "{song}" by "{artist}":\n\n{raw_lyrics[:12000]}'  # Reduced to avoid token limits
            }
        ]
        
        formatted = await call_openai(messages)
        return formatted.strip() if formatted else raw_lyrics
        
    except Exception as e:
        print(f"AI formatting error: {str(e)}")
        
        # Enhanced fallback cleanup
        cleaned = raw_lyrics
        
        # Remove HTML entities and tags
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        cleaned = re.sub(r'&quot;', '"', cleaned)
        cleaned = re.sub(r'&#x27;|&#39;', "'", cleaned)
        cleaned = re.sub(r'&amp;', '&', cleaned)
        cleaned = re.sub(r'&nbsp;', ' ', cleaned)
        cleaned = re.sub(r'&lt;', '<', cleaned)
        cleaned = re.sub(r'&gt;', '>', cleaned)
        
        # Remove common website artifacts
        artifacts = [
            r'Advertisement.*?\n',
            r'Cookie.*?\n',
            r'Privacy.*?\n',
            r'Terms of Service.*?\n',
            r'Share on.*?\n',
            r'Print.*?\n',
            r'Download.*?\n',
            r'Subscribe.*?\n',
            r'Follow us.*?\n',
            r'Copyright.*?\n',
            r'All rights reserved.*?\n'
        ]
        
        for pattern in artifacts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Limit consecutive newlines
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)  # Trim lines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Normalize spaces
        
        return cleaned.strip()

def validate_lyrics_content(lyrics: str) -> bool:
    """Validate that extracted content looks like actual lyrics"""
    if not lyrics or len(lyrics.strip()) < 50:
        return False
    
    # Check for common non-lyrics indicators
    non_lyrics_indicators = [
        'javascript', 'function', 'document.', 'window.',
        'advertisement', 'cookie policy', 'privacy policy',
        'terms of service', 'subscribe', 'newsletter'
    ]
    
    lyrics_lower = lyrics.lower()
    for indicator in non_lyrics_indicators:
        if indicator in lyrics_lower:
            return False
    
    # Positive indicators of lyrics
    lines = lyrics.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    if len(non_empty_lines) < 5:  # Too few lines for lyrics
        return False
    
    # Check for typical lyrics structure
    has_sections = bool(re.search(r'\[.*\]', lyrics))  # Section markers
    has_repetition = len(set(non_empty_lines)) < len(non_empty_lines) * 0.8  # Some repetition expected
    
    return True

@lyrics_router.post("/extract", response_model=LyricsResponse)
async def extract_lyrics(request: LyricsRequest):
    """Extract and format lyrics from HTML content"""
    try:
        # Validate input
        if not request.html_content or not request.artist or not request.song:
            raise HTTPException(
                status_code=400,
                detail="HTML content, artist, and song parameters are required"
            )
        
        if len(request.html_content) > 2_000_000:  # 2MB limit
            raise HTTPException(
                status_code=400,
                detail="HTML content too large (max 2MB)"
            )
        
        print(f"Processing lyrics: {request.artist} - {request.song}")
        
        # Determine extraction method based on URL or content
        raw_lyrics = ""
        source = "html_extraction"
        
        if request.url and 'genius.com' in request.url:
            raw_lyrics = extract_genius_lyrics_lxml(request.html_content)
            source = "genius"
        else:
            raw_lyrics = extract_generic_lyrics_lxml(request.html_content, request.url or "")
            source = "generic"
        
        # Validate extracted content
        if not validate_lyrics_content(raw_lyrics):
            raise HTTPException(
                status_code=404,
                detail=f'Could not extract valid lyrics for "{request.song}" by {request.artist}. The content may not contain lyrics or may be from an unsupported source.'
            )
        
        # Format lyrics with AI
        try:
            formatted_lyrics = await asyncio.wait_for(
                format_lyrics_with_ai(raw_lyrics, request.artist, request.song),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            print("AI formatting timed out, using fallback cleanup")
            formatted_lyrics = await format_lyrics_with_ai.__wrapped__(raw_lyrics, request.artist, request.song)
        
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
    return {
        "status": "healthy", 
        "service": "lyrics_extraction",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0"
    }