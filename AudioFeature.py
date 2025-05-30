from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
import re
from typing import Optional, Dict, Any
import asyncio
import json
import random
import os
from openai import AsyncOpenAI
from bs4 import BeautifulSoup

audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

# Pydantic models for request/response
class AudioFeatureRequest(BaseModel):
    title: str
    artist: str

class AudioFeature(BaseModel):
    title: str
    artist: str
    key: Optional[str] = None
    camelot: Optional[str] = None
    bpm: Optional[int] = None
    duration: Optional[str] = None
    release_date: Optional[str] = None
    explicit: Optional[str] = None
    album: Optional[str] = None
    label: Optional[str] = None
    popularity: Optional[int] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    loudness: Optional[str] = None

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
    
    async def fetch_and_extract_component(self, url: str) -> str:
        """Step 2: Fetch page and extract dr-ag component using BeautifulSoup"""
        print(f"\n🌐 STEP 2: Fetching page and extracting dr-ag component")
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
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
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
                    delay = 3 + (attempt * 2)
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
                    
                    html_content = response.text
                    print(f"📄 HTML content received: {len(html_content):,} characters")
                    
                    # Extract dr-ag component using BeautifulSoup
                    print("🔍 Extracting dr-ag component with BeautifulSoup...")
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Try multiple ways to find the dr-ag component
                    dr_ag_div = None
                    
                    # Method 1: Direct class search
                    dr_ag_div = soup.find('div', class_='dr-ag')
                    if dr_ag_div:
                        print("✅ Found dr-ag component using direct class search")
                    
                    # Method 2: Search in class list (in case there are multiple classes)
                    if not dr_ag_div:
                        dr_ag_div = soup.find('div', class_=lambda x: x and 'dr-ag' in x)
                        if dr_ag_div:
                            print("✅ Found dr-ag component using class list search")
                    
                    # Method 3: Search by partial class name
                    if not dr_ag_div:
                        dr_ag_divs = soup.find_all('div', class_=re.compile(r'dr-ag'))
                        if dr_ag_divs:
                            dr_ag_div = dr_ag_divs[0]
                            print("✅ Found dr-ag component using regex search")
                    
                    # Method 4: Look for divs containing ant-progress elements (fallback)
                    if not dr_ag_div:
                        print("🔍 Searching for div containing ant-progress elements...")
                        potential_divs = soup.find_all('div')
                        for div in potential_divs:
                            if div.find('div', class_=re.compile(r'ant-progress')):
                                # Check if this div contains multiple progress circles
                                progress_count = len(div.find_all('div', class_=re.compile(r'ant-progress')))
                                if progress_count >= 5:  # Should have at least 5 audio features
                                    dr_ag_div = div
                                    print(f"✅ Found container with {progress_count} progress circles")
                                    break
                    
                    if dr_ag_div:
                        component_html = str(dr_ag_div)
                        print(f"✅ Successfully extracted component: {len(component_html)} characters")
                        
                        # Quick validation - check if it contains expected audio features
                        feature_count = len(re.findall(r'ant-progress-text.*?title="(\d+|[^"]*dB)"', component_html))
                        print(f"🔍 Found {feature_count} audio feature values in component")
                        
                        if feature_count >= 5:  # Should have at least 5 features
                            print(f"🔍 Component preview: {component_html[:300]}...")
                            return component_html
                        else:
                            print("⚠️ Component found but doesn't contain enough audio features")
                    
                    print("❌ No suitable dr-ag component found")
                    raise HTTPException(status_code=404, detail="No audio feature component found on the page")
                        
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
    
    async def extract_with_openai(self, component_html: str, title: str, artist: str) -> Dict[str, Any]:
        """Step 3: Extract audio features from component using OpenAI"""
        print(f"\n🤖 STEP 3: OpenAI extraction from component")
        
        if not self.openai_client:
            print("❌ OpenAI client not available!")
            raise HTTPException(status_code=500, detail="OpenAI API key required")
        
        if not component_html:
            print("❌ No component HTML provided")
            return {}
        
        try:
            component_length = len(component_html)
            print(f"📄 Component HTML length: {component_length:,} characters")
            
            prompt = f"""
You are an expert at extracting audio features from TuneBat HTML components. Extract ALL available data from this dr-ag component.

SONG: "{title}" by "{artist}"

EXTRACT THESE ELEMENTS:

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

EXTRACTION PATTERNS TO LOOK FOR:

1. **Ant Design Progress Circles:**
   <span class="ant-progress-text" title="VALUE">VALUE</span>
   <span class="ant-typography fd89q">FEATURE_NAME</span>

2. **Bold Text Patterns:**
   **G Major** (key)
   **9B** (camelot)  
   **137** (BPM)
   **4:30** (duration)

3. **Label Patterns:**
   Release Date:** July 29, 2003**
   Explicit:** No**
   Album:** 葉惠美**
   Label: **Universal Music Taiwan (JVR)**

4. **Text Content:**
   Look for any text that contains musical keys, camelots, BPM values, etc.

IMPORTANT RULES:
- Extract EXACT values only - do not estimate or guess
- For missing values, return null
- Pay attention to bold text markers (**text**)
- Look for both English and non-English text (albums may be in other languages)

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

COMPONENT HTML:
{component_html}
"""
            
            print(f"🤖 Sending {component_length:,} characters to OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise data extraction specialist. Extract audio features and metadata from TuneBat HTML components. Return only valid JSON with exact values found in the HTML."
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
        print(f"🎵 Process: Brave Search → BeautifulSoup → OpenAI")
        
        try:
            # Step 1: Brave Search
            url = await self.brave_search_tunebat(title, artist)
            
            # Step 2: Fetch page and extract component
            component_html = await self.fetch_and_extract_component(url)
            
            # Step 3: OpenAI extraction
            features = await self.extract_with_openai(component_html, title, artist)
            
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
    2. BeautifulSoup extract dr-ag component  
    3. OpenAI extract all elements
    
    - **title**: The song title
    - **artist**: The artist name
    """
    return await audio_service.get_audio_features(title, artist)

@audio_feature_router.post("/extract", response_model=AudioFeature)
async def extract_audio_features_post(request: AudioFeatureRequest):
    """
    Extract audio features using POST method
    """
    return await audio_service.get_audio_features(request.title, request.artist)

@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "audio_feature_extractor",
        "process": "brave_search → beautifulsoup → openai",
        "brave_api": "configured" if audio_service.use_api else "not_configured",
        "openai_api": "configured" if audio_service.openai_client else "not_configured"
    }