from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
import re
from typing import Optional, Dict, Any, List
import asyncio
import json
import random
import os
from openai import AsyncOpenAI
from lxml import html
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

# Pydantic models for request/response
class AudioFeatureRequest(BaseModel):
    title: str
    artist: str

class MultipleAudioFeatureRequest(BaseModel):
    tracks: List[AudioFeatureRequest]

class AudioFeature(BaseModel):
    title: str
    artist: str
    bpm: Optional[int] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None  # or "valence" if from Spotify
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    loudness: Optional[str] = None  # Consider converting to float (dB)
    error: Optional[str] = None  # To track any errors for individual tracks

class MultipleAudioFeatureResponse(BaseModel):
    results: List[AudioFeature]
    total_processed: int
    successful: int
    failed: int

class AudioFeatureService:
    def __init__(self, brave_api_key: Optional[str], openai_api_key: Optional[str]):
        self.brave_api_key = brave_api_key
        self.openai_api_key = openai_api_key
        self.brave_search_url = "https://api.search.brave.com/res/v1/web/search"
        self.use_api = brave_api_key is not None and brave_api_key != "YOUR_BRAVE_API_KEY_HERE"
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            print("✅ OpenAI client initialized")
        else:
            self.openai_client = None
            print("❌ OpenAI API key not provided - this service requires OpenAI!")
    
    async def brave_search_tunebat(self, title: str, artist: str) -> str:
        """Step 1: Search for tunebat page using Brave Search API"""
        print(f"\n🔍 STEP 1: Brave Search for '{title}' by '{artist}'")
        
        if not self.brave_api_key or self.brave_api_key == "YOUR_BRAVE_API_KEY_HERE":
            print("❌ Brave Search API key not configured")
            raise HTTPException(
                status_code=500, 
                detail="Brave Search API key not configured. Please set a valid API key."
            )
        
        query = f"site:tunebat.com {title} {artist}"
        print(f"🔍 Search query: '{query}'")
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": 3,
            "search_lang": "en",
            "country": "US",
            "safesearch": "off",
            "freshness": "",
            "text_decorations": False,
            "spellcheck": True
        }
        
        async with httpx.AsyncClient() as client:
            try:
                print("📡 Making Brave Search request...")
                response = await client.get(
                    self.brave_search_url,
                    headers=headers,
                    params=params,
                    timeout=15.0
                )
                
                print(f"📡 Brave Search response status: {response.status_code}")
                
                if response.status_code == 422:
                    print("❌ Search API error 422")
                    raise HTTPException(status_code=422, detail="Brave Search API configuration error")
                elif response.status_code == 401:
                    print("❌ Search API error 401 - Unauthorized")
                    raise HTTPException(status_code=401, detail="Invalid Brave Search API key")
                elif response.status_code == 429:
                    print("❌ Search API error 429 - Rate limit")
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                response.raise_for_status()
                search_results = response.json()
                
                if not search_results.get("web", {}).get("results"):
                    print("❌ No search results found")
                    raise HTTPException(status_code=404, detail="No TuneBat results found")
                
                # Find the best TuneBat URL
                for result in search_results["web"]["results"]:
                    url = result.get("url", "")
                    if "tunebat.com" in url and "/Info/" in url:
                        print(f"✅ Found TuneBat URL: {url}")
                        return url
                
                # Fallback to first result if no /Info/ URL found
                first_url = search_results["web"]["results"][0].get("url", "")
                print(f"⚠️ Using first result: {first_url}")
                return first_url
                
            except httpx.HTTPError as e:
                print(f"❌ HTTP error during search: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    
    async def fetch_and_extract_chunk(self, url: str) -> str:
        """Step 2: Fetch page and extract 6000-13000 character chunk"""
        print(f"\n🌐 STEP 2: Fetching page and extracting content chunk")
        print(f"🌐 URL: {url}")
        
        # Rotate through multiple realistic user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        selected_ua = random.choice(user_agents)
        print(f"🌐 Using User-Agent: {selected_ua[:50]}...")
        
        # More realistic browser headers to avoid detection
        headers = {
            'User-Agent': selected_ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Charset': 'UTF-8',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive',
            'DNT': '1',
            # Add referer to look like organic traffic
            'Referer': 'https://www.google.com/'
        }
        
        # Use longer timeout and retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                timeout=30.0,  # Longer timeout
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                try:
                    # Progressive delay: 3, 5, 8 seconds
                    delay = 5 + (attempt * 2)
                    print(f"⏱️ Attempt {attempt + 1}/{max_retries}: Sleeping {delay} seconds to imitate human...")
                    await asyncio.sleep(delay)
                    
                    print(f"🌐 Making HTTP request (attempt {attempt + 1})...")
                    response = await client.get(url)
                    
                    print(f"🌐 Response status: {response.status_code}")
                    
                    if response.status_code == 403:
                        print(f"❌ Access denied (403) on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            print(f"🔄 Retrying in {delay + 5} seconds...")
                            await asyncio.sleep(delay + 5)
                            continue
                        else:
                            print("❌ All attempts failed - TuneBat is blocking requests")
                            raise HTTPException(
                                status_code=403, 
                                detail="TuneBat is blocking requests. The site may have anti-bot protection. Try again later or use a VPN."
                            )
                    
                    response.raise_for_status()
                    
                    # Handle encoding properly
                    try:
                        # Try to decode with the response's declared encoding
                        if response.encoding:
                            html_content = response.content.decode(response.encoding)
                        else:
                            # Fallback to UTF-8
                            html_content = response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            # Try with latin-1 as fallback
                            html_content = response.content.decode('latin-1')
                        except UnicodeDecodeError:
                            # Last resort - use response.text with error handling
                            html_content = response.text
                    
                    print(f"📄 HTML content received: {len(html_content):,} characters")
                    print(f"📄 Content encoding: {response.encoding}")

                    tree = html.fromstring(html_content)

                    # Now you can use XPath
                    divs = tree.xpath('//div[contains(@class, "dr-ag")]')

                    texts = [div.text_content() for div in divs]
                    all_text = '\n'.join(texts)

                    print(all_text)
                    return all_text
                        
                except httpx.TimeoutException:
                    print(f"⏰ Request timed out on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        print("🔄 Retrying...")
                        continue
                    else:
                        raise HTTPException(status_code=408, detail="Request timed out after multiple attempts")
                
                except httpx.HTTPError as e:
                    print(f"❌ HTTP error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        print("🔄 Retrying...")
                        continue
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to fetch page after {max_retries} attempts: {str(e)}")
        
        # This shouldn't be reached, but just in case
        raise HTTPException(status_code=500, detail="Unexpected error in fetch loop")
    
    async def extract_with_openai(self, html_chunk: str, title: str, artist: str) -> Dict[str, Any]:
        """Step 3: Extract audio features from HTML chunk using OpenAI"""
        print(f"\n🤖 STEP 3: OpenAI extraction from HTML chunk")
        
        if not self.openai_client:
            print("❌ OpenAI client not available!")
            raise HTTPException(status_code=500, detail="OpenAI API key required")
        
        if not html_chunk:
            print("❌ No HTML chunk provided")
            return {}
        
        try:
            chunk_length = len(html_chunk)
            print(f"📄 HTML chunk length: {chunk_length:,} characters")
            
            prompt = f"""
You are an expert at extracting audio features from TuneBat HTML content. Extract ALL available data from this HTML chunk.

SONG: "{title}" by "{artist}"

EXTRACT THESE ELEMENTS FROM THE HTML:

**MUSICAL INFORMATION:**
- Key (e.g., "G Major", "C# Minor")
- Camelot (e.g., "9B", "4A") 
- BPM (beats per minute, integer)
- Duration (e.g., "4:30", "3:15")

**RELEASE INFORMATION:**
- Release Date (e.g., "July 29, 2003")
- Explicit (e.g., "Yes", "No")
- Album name
- Label (e.g., "Universal Music Taiwan (JVR)")

**AUDIO FEATURES (0-100 integers):**
- Popularity
- Energy  
- Danceability
- Happiness/Valence
- Acousticness
- Instrumentalness
- Liveness
- Speechiness
- Loudness (e.g., "-7 dB")

COMMON PATTERNS TO LOOK FOR:

1. **Progress Circle Values:**
   - ant-progress-text with title or data attributes
   - Numbers followed by feature names like "Energy", "Danceability"
   - Percentage values in spans or divs

2. **Bold Text Patterns:**
   - **G Major** (key)
   - **9B** (camelot)  
   - **137** (BPM)
   - **4:30** (duration)

3. **Structured Data:**
   - JSON-LD or data attributes
   - Meta tags with audio features
   - Structured lists or tables

4. **Text Content:**
   - Release Date: July 29, 2003
   - Album: Album Name
   - Label: Record Label
   - Key signatures, BPM values, duration timestamps

IMPORTANT RULES:
- Extract EXACT values only - do not estimate or guess
- For missing values, return null
- Look for both visible text and HTML attributes/data
- Audio feature values should be integers 0-100 (except BPM and loudness)
- Pay attention to HTML structure and class names

Return ONLY valid JSON:

{{
    "key": null,
    "camelot": null,
    "bpm": null,
    "duration": null,
    "release_date": null,
    "explicit": null,
    "album": null,
    "label": null,
    "popularity": null,
    "energy": null,
    "danceability": null,
    "happiness": null,
    "acousticness": null,
    "instrumentalness": null,
    "liveness": null,
    "speechiness": null,
    "loudness": null
}}

HTML CHUNK:
{html_chunk}
"""
            
            print(f"🤖 Sending {chunk_length:,} characters to OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise data extraction specialist. Extract audio features and metadata from TuneBat HTML content. Return only valid JSON with exact values found in the HTML."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"🤖 OpenAI response ({len(ai_response)} chars): {ai_response}")
            
            # Parse JSON response
            try:
                # Clean markdown if present
                if "```json" in ai_response:
                    ai_response = ai_response.split("```json")[1].split("```")[0]
                elif "```" in ai_response:
                    parts = ai_response.split("```")
                    if len(parts) >= 3:
                        ai_response = parts[1]
                
                features = json.loads(ai_response)
                print(f"✅ Successfully parsed JSON: {features}")
                
                # Clean and validate extracted data
                cleaned_features = {}
                for key, value in features.items():
                    if value is not None and value != "null" and value != "":
                        # Validate numeric fields
                        if key in ['bpm', 'popularity', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                            if isinstance(value, (int, float)) and 0 <= value <= 300:  # Allow higher BPM
                                cleaned_features[key] = int(value)
                            elif isinstance(value, str) and value.isdigit():
                                val = int(value)
                                if 0 <= val <= 300:
                                    cleaned_features[key] = val
                        # Validate string fields
                        elif key in ['key', 'camelot', 'duration', 'release_date', 'explicit', 'album', 'label', 'loudness']:
                            if isinstance(value, str) and len(value.strip()) > 0:
                                cleaned_features[key] = value.strip()
                
                print(f"✅ Cleaned and validated features: {cleaned_features}")
                return cleaned_features
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse OpenAI JSON response: {e}")
                print(f"❌ Raw response: {ai_response}")
                return {}
                
        except Exception as e:
            print(f"❌ OpenAI extraction error: {str(e)}")
            return {}
    
    async def get_audio_features(self, title: str, artist: str) -> AudioFeature:
        """Main method: 3-step process to get audio features"""
        print(f"\n🎵 ======= AUDIO FEATURE EXTRACTION =======")
        print(f"🎵 Song: '{title}' by '{artist}'")
        print(f"🎵 Process: Brave Search → HTML Chunk → OpenAI")
        
        try:
            # Step 1: Brave Search
            url = await self.brave_search_tunebat(title, artist)
            
            # Step 2: Fetch page and extract chunk
            html_chunk = await self.fetch_and_extract_chunk(url)
            
            # Step 3: OpenAI extraction
            features = await self.extract_with_openai(html_chunk, title, artist)
            
            # Create result
            result = AudioFeature(
                title=title,
                artist=artist,
                **features
            )
            
            print(f"\n🎯 ======= FINAL RESULT =======")
            print(f"🎯 Extracted features: {len(features)} fields")
            for key, value in features.items():
                if value is not None:
                    print(f"🎯 {key}: {value}")
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    async def get_single_track_features_safe(self, title: str, artist: str) -> AudioFeature:
        """Safe wrapper for single track processing with error handling"""
        try:
            return await self.get_audio_features(title, artist)
        except Exception as e:
            print(f"❌ Error processing '{title}' by '{artist}': {str(e)}")
            return AudioFeature(
                title=title,
                artist=artist,
                error=str(e)
            )

    async def get_multiple_audio_features(self, tracks: List[AudioFeatureRequest]) -> MultipleAudioFeatureResponse:
        """Process multiple tracks concurrently with a limit of 5 simultaneous requests"""
        print(f"\n🎵 ======= MULTIPLE AUDIO FEATURE EXTRACTION =======")
        print(f"🎵 Processing {len(tracks)} tracks")
        print(f"🎵 Max concurrent requests: 5")
        
        if len(tracks) > 5:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 5 tracks allowed per request"
            )
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        async def process_with_semaphore(track: AudioFeatureRequest):
            async with semaphore:
                return await self.get_single_track_features_safe(track.title, track.artist)
        
        # Process all tracks concurrently
        start_time = asyncio.get_event_loop().time()
        
        tasks = [process_with_semaphore(track) for track in tracks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Process results and count successes/failures
        processed_results = []
        successful = 0
        failed = 0
        
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions that weren't caught by the safe wrapper
                failed += 1
                processed_results.append(AudioFeature(
                    title="Unknown",
                    artist="Unknown",
                    error=str(result)
                ))
            elif result.error:
                failed += 1
                processed_results.append(result)
            else:
                successful += 1
                processed_results.append(result)
        
        print(f"\n🎯 ======= BATCH PROCESSING COMPLETE =======")
        print(f"🎯 Total tracks: {len(tracks)}")
        print(f"🎯 Successful: {successful}")
        print(f"🎯 Failed: {failed}")
        print(f"🎯 Processing time: {total_time:.2f} seconds")
        print(f"🎯 Average time per track: {total_time/len(tracks):.2f} seconds")
        
        return MultipleAudioFeatureResponse(
            results=processed_results,
            total_processed=len(tracks),
            successful=successful,
            failed=failed
        )

# Initialize service
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
audio_service = AudioFeatureService(BRAVE_API_KEY, OPENAI_API_KEY)

@audio_feature_router.get("/extract", response_model=AudioFeature)
async def extract_audio_features(
    title: str = Query(..., description="Song title"),
    artist: str = Query(..., description="Artist name")
):
    """
    Extract audio features using 3-step process:
    1. Brave Search for TuneBat page
    2. Extract 6000-13000 char HTML chunk
    3. OpenAI extract all elements
    
    - **title**: The song title
    - **artist**: The artist name
    """
    return await audio_service.get_audio_features(title, artist)

@audio_feature_router.post("/extract", response_model=AudioFeature)
async def extract_audio_features_post(request: AudioFeatureRequest):
    """
    Extract audio features using POST method for single track
    """
    return await audio_service.get_audio_features(request.title, request.artist)

@audio_feature_router.post("/extract_multiple", response_model=MultipleAudioFeatureResponse)
async def extract_multiple_audio_features(request: MultipleAudioFeatureRequest):
    """
    Extract audio features for multiple tracks (up to 5 tracks simultaneously)
    
    - **tracks**: List of tracks with title and artist
    - Maximum 5 tracks per request
    - Processes tracks concurrently for faster response
    """
    return await audio_service.get_multiple_audio_features(request.tracks)

@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "audio_feature_extractor",
        "process": "brave_search → html_chunk → openai",
        "brave_api": "configured" if audio_service.use_api else "not_configured",
        "openai_api": "configured" if audio_service.openai_client else "not_configured",
        "max_concurrent_tracks": 5
    }