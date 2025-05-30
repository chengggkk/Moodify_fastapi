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
    bpm: Optional[int] = None
    time_signature: Optional[str] = None
    camelot: Optional[str] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None
    loudness: Optional[str] = None
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    popularity: Optional[int] = None

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
            print("❌ OpenAI API key not provided - this scraper requires OpenAI!")
    
    async def search_tunebat(self, title: str, artist: str) -> str:
        """Search for tunebat results using Brave Search API"""
        print(f"🔍 Starting search for: '{title}' by '{artist}'")
        
        if not self.brave_api_key or self.brave_api_key == "YOUR_BRAVE_API_KEY_HERE":
            print("❌ Brave Search API key not configured")
            raise HTTPException(
                status_code=500, 
                detail="Brave Search API key not configured. Please set a valid API key."
            )
        
        query = f"tunebat {title} {artist}"
        print(f"🔍 Search query: '{query}'")
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": 1,
            "search_lang": "en",
            "country": "US",
            "safesearch": "off",
            "freshness": "",
            "text_decorations": False,
            "spellcheck": True
        }
        
        async with httpx.AsyncClient() as client:
            try:
                print("📡 Making search request...")
                response = await client.get(
                    self.brave_search_url,
                    headers=headers,
                    params=params,
                    timeout=15.0
                )
                
                print(f"📡 Search response status: {response.status_code}")
                
                # Handle specific error codes
                if response.status_code == 422:
                    print("❌ Search API error 422")
                    error_detail = "Invalid request parameters or API key issue"
                    try:
                        error_json = response.json()
                        if "message" in error_json:
                            error_detail = error_json["message"]
                    except:
                        pass
                    raise HTTPException(
                        status_code=422, 
                        detail=f"Brave Search API error: {error_detail}. Check your API key and parameters."
                    )
                elif response.status_code == 401:
                    print("❌ Search API error 401 - Unauthorized")
                    raise HTTPException(
                        status_code=401, 
                        detail="Unauthorized: Invalid or expired Brave Search API key"
                    )
                elif response.status_code == 429:
                    print("❌ Search API error 429 - Rate limit")
                    raise HTTPException(
                        status_code=429, 
                        detail="Rate limit exceeded. Please try again later."
                    )
                
                response.raise_for_status()
                search_results = response.json()
                
                if not search_results.get("web", {}).get("results"):
                    print("❌ No search results found")
                    raise HTTPException(status_code=404, detail="No search results found")
                
                first_result = search_results["web"]["results"][0]
                url = first_result.get("url", "")
                print(f"✅ Found TuneBat URL: {url}")
                return url
                
            except httpx.HTTPError as e:
                print(f"❌ HTTP error during search: {str(e)}")
                if "422" in str(e):
                    raise HTTPException(
                        status_code=422, 
                        detail="Brave Search API configuration error. Please check your API key and subscription status."
                    )
                raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    
    def extract_audio_features_component(self, html_content: str) -> str:
        """Extract the specific dr-ag component containing audio features using BeautifulSoup"""
        print("🔍 Extracting dr-ag component with BeautifulSoup...")
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the div with class "dr-ag"
            dr_ag_div = soup.find('div', class_='dr-ag')
            
            if dr_ag_div:
                component_html = str(dr_ag_div)
                print(f"✅ Found dr-ag component: {len(component_html)} characters")
                print(f"🔍 Component preview: {component_html[:200]}...")
                return component_html
            else:
                print("❌ No dr-ag component found")
                return ""
                
        except Exception as e:
            print(f"❌ BeautifulSoup extraction error: {str(e)}")
            return ""
    
    async def extract_with_openai_component(self, component_html: str, title: str, artist: str) -> Dict[str, Any]:
        """Use OpenAI to extract audio features from the specific dr-ag component"""
        print("🤖 Starting AI extraction from dr-ag component...")
        
        if not self.openai_client:
            print("❌ OpenAI client not available!")
            raise HTTPException(status_code=500, detail="OpenAI API key required for this service")
        
        if not component_html:
            print("❌ No component HTML provided")
            return {}
        
        try:
            component_length = len(component_html)
            print(f"📄 Component HTML length: {component_length:,} characters")
            
            model = "gpt-4o-mini"
            max_tokens = 500
            
            prompt = f"""
You are an expert at extracting audio features from TuneBat HTML components. You must be EXTREMELY PRECISE and extract the EXACT values shown in the HTML.

Extract the following audio features for "{title}" by "{artist}" from this dr-ag component:

CRITICAL PATTERN TO LOOK FOR:
The HTML contains Ant Design progress circles with this EXACT structure:
<span class="ant-progress-text" title="VALUE">VALUE</span>
followed by
<span class="ant-typography fd89q">FEATURE_NAME</span>

EXAMPLE PATTERNS:
- <span class="ant-progress-text" title="86 ">86 </span> ... <span class="ant-typography fd89q">popularity</span>
- <span class="ant-progress-text" title="19 ">19 </span> ... <span class="ant-typography fd89q">energy</span>
- <span class="ant-progress-text" title="41 ">41 </span> ... <span class="ant-typography fd89q">danceability</span>
- <span class="ant-progress-text" title="16 ">16 </span> ... <span class="ant-typography fd89q">happiness</span>
- <span class="ant-progress-text" title="64 ">64 </span> ... <span class="ant-typography fd89q">acousticness</span>
- <span class="ant-progress-text" title="0 ">0 </span> ... <span class="ant-typography fd89q">instrumentalness</span>
- <span class="ant-progress-text" title="21 ">21 </span> ... <span class="ant-typography fd89q">liveness</span>
- <span class="ant-progress-text" title="4 ">4 </span> ... <span class="ant-typography fd89q">speechiness</span>
- <span class="ant-progress-text" title="-11 dB">-11 dB</span> ... <span class="ant-typography fd89q">loudness</span>

EXTRACTION RULES:
1. Find EACH ant-progress-text span and match it with the corresponding feature name
2. Extract the EXACT numeric value from the title attribute (the value inside title="...")
3. For loudness, include the "dB" unit
4. DO NOT make up or estimate values - only extract what is explicitly shown
5. If a value is not found, return null

FEATURES TO EXTRACT:
- BPM (beats per minute) - integer value
- Key (musical key like "F# Major", "C Minor", etc.) - string
- Time Signature (like "4/4", "3/4") - string  
- Camelot (like "2B", "8A", "12B") - string
- Energy (0-100) - integer
- Danceability (0-100) - integer  
- Happiness/Valence (0-100) - integer
- Loudness (like "-11 dB") - string with dB
- Acousticness (0-100) - integer
- Instrumentalness (0-100) - integer
- Liveness (0-100) - integer
- Speechiness (0-100) - integer
- Popularity (0-100) - integer

Return ONLY valid JSON with the extracted values. Use null for missing values:

{{
    "bpm": null,
    "key": null,
    "time_signature": null,
    "camelot": null,
    "energy": null,
    "danceability": null,
    "happiness": null,
    "loudness": null,
    "acousticness": null,
    "instrumentalness": null,
    "liveness": null,
    "speechiness": null,
    "popularity": null
}}

DR-AG COMPONENT HTML:
{component_html}
"""
            
            print(f"🤖 Sending {component_length:,} characters to OpenAI {model}...")
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise data extraction specialist. Extract audio features from TuneBat dr-ag component by finding ant-progress-text spans and their corresponding feature names. Return only valid JSON with EXACT values from the HTML."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0,  # Deterministic output
                max_tokens=max_tokens
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"🤖 OpenAI response ({len(ai_response)} chars): {ai_response}")
            
            # Parse JSON response
            try:
                # Remove any markdown code blocks if present
                if "```json" in ai_response:
                    ai_response = ai_response.split("```json")[1].split("```")[0]
                elif "```" in ai_response:
                    # Handle cases where there's just ``` without json
                    parts = ai_response.split("```")
                    if len(parts) >= 3:
                        ai_response = parts[1]
                    else:
                        # Try to find JSON in the response
                        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                        if json_match:
                            ai_response = json_match.group(0)
                
                features = json.loads(ai_response)
                print(f"✅ Successfully parsed JSON: {features}")
                
                # Clean and validate the extracted data
                cleaned_features = {}
                for key, value in features.items():
                    if value is not None and value != "null" and value != "":
                        # Additional validation
                        if key in ['bpm', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'popularity']:
                            if isinstance(value, (int, float)) and 0 <= value <= 200:  # Reasonable range
                                cleaned_features[key] = int(value)
                            elif isinstance(value, str) and value.isdigit():
                                val = int(value)
                                if 0 <= val <= 200:
                                    cleaned_features[key] = val
                        elif key in ['key', 'time_signature', 'camelot', 'loudness']:
                            if isinstance(value, str) and len(value.strip()) > 0:
                                cleaned_features[key] = value.strip()
                
                print(f"✅ Cleaned and validated features: {cleaned_features}")
                return cleaned_features
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse OpenAI JSON response: {e}")
                print(f"❌ Raw response was: {ai_response}")
                
                # Try to extract individual values if JSON parsing fails
                return self.extract_fallback_patterns(ai_response)
                
        except Exception as e:
            print(f"❌ OpenAI extraction error: {str(e)}")
            return {}
    
    def extract_fallback_patterns(self, response_text: str) -> Dict[str, Any]:
        """Fallback extraction from AI response if JSON parsing fails"""
        print("🔧 Attempting fallback pattern extraction from AI response...")
        features = {}
        
        # Try to extract individual values using regex
        patterns = {
            'bpm': [r'"bpm":\s*(\d+)', r'bpm["\s:]*(\d+)', r'(\d+)\s*bpm'],
            'energy': [r'"energy":\s*(\d+)', r'energy["\s:]*(\d+)'],
            'danceability': [r'"danceability":\s*(\d+)', r'danceability["\s:]*(\d+)'],
            'happiness': [r'"happiness":\s*(\d+)', r'happiness["\s:]*(\d+)'],
            'popularity': [r'"popularity":\s*(\d+)', r'popularity["\s:]*(\d+)'],
            'acousticness': [r'"acousticness":\s*(\d+)', r'acousticness["\s:]*(\d+)'],
            'instrumentalness': [r'"instrumentalness":\s*(\d+)', r'instrumentalness["\s:]*(\d+)'],
            'liveness': [r'"liveness":\s*(\d+)', r'liveness["\s:]*(\d+)'],
            'speechiness': [r'"speechiness":\s*(\d+)', r'speechiness["\s:]*(\d+)'],
            'key': [r'"key":\s*"([^"]+)"', r'key["\s:]*"?([A-G][#♯♭b]?\s*(?:major|minor))"?'],
            'camelot': [r'"camelot":\s*"([^"]+)"', r'camelot["\s:]*"?(\d+[AB])"?'],
            'loudness': [r'"loudness":\s*"([^"]+)"', r'loudness["\s:]*"?(-?\d+(?:\.\d+)?\s*dB)"?']
        }
        
        for feature, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if feature in ['bpm', 'energy', 'danceability', 'happiness', 'popularity', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                        if value.isdigit():
                            features[feature] = int(value)
                    else:
                        features[feature] = value
                    print(f"✅ Extracted {feature}: {features[feature]}")
                    break
        
        return features
    
    async def scrape_tunebat_page(self, url: str, title: str = "", artist: str = "") -> Dict[str, Any]:
        """Scrape audio features from tunebat page using AI only"""
        print(f"🌐 Starting to scrape URL: {url}")
        
        if not url:
            print("❌ No URL provided")
            raise HTTPException(status_code=404, detail="No URL found in search results")
        
        # Multiple user agents to rotate
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        selected_ua = random.choice(user_agents)
        print(f"🌐 Using User-Agent: {selected_ua[:50]}...")
        
        # Headers to mimic a real browser request
        headers = {
            'User-Agent': selected_ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/'
        }
        
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=20.0
        ) as client:
            try:
                # Add random delay to appear more human-like
                delay = random.uniform(1, 3)
                print(f"⏱️  Waiting {delay:.2f} seconds before request...")
                await asyncio.sleep(delay)
                
                print("🌐 Making HTTP request...")
                response = await client.get(url)
                
                print(f"🌐 Response status: {response.status_code}")
                
                if response.status_code == 403:
                    print("❌ Access denied (403)")
                    raise HTTPException(
                        status_code=403, 
                        detail="Access denied by tunebat. Try using a VPN or consider alternative data sources like Spotify Web API."
                    )
                
                response.raise_for_status()
                
                html_content = response.text
                print(f"📄 HTML content received: {len(html_content):,} characters")
                
                # Send directly to OpenAI without any preprocessing
                print("🤖 Sending full HTML content to OpenAI for extraction...")
                features = await self.extract_with_openai_full(html_content, title, artist)
                
                return features
                
            except httpx.HTTPError as e:
                print(f"❌ HTTP error during scraping: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to scrape page: {str(e)}")
    
    async def get_audio_features(self, title: str, artist: str) -> AudioFeature:
        """Main method to get audio features using AI-only approach"""
        print(f"\n🎵 ==> STARTING AI-ONLY AUDIO FEATURE EXTRACTION <==")
        print(f"🎵 Song: '{title}' by '{artist}'")
        
        features = {}
        
        if self.use_api:
            print("📡 Using Brave API search...")
            try:
                # Primary strategy: Search tunebat via Brave API
                url = await self.search_tunebat(title, artist)
                features = await self.scrape_tunebat_page(url, title, artist)
            except HTTPException as e:
                print(f"❌ API search failed: {e.detail}")
                if e.status_code in [422, 401, 500]:
                    print("🔄 Falling back to direct approach...")
                    # Fallback to non-API methods
                    features = await self.try_direct_tunebat_approach(title, artist)
                else:
                    raise e
        else:
            print("🔄 Skipping API, using direct approach...")
            # Skip API and go directly to fallback methods
            features = await self.try_direct_tunebat_approach(title, artist)
        
        print(f"\n🎯 ==> FINAL AI EXTRACTION RESULT <==")
        result = AudioFeature(
            title=title,
            artist=artist,
            **features
        )
        print(f"🎯 Returning: {result.dict()}")
        return result
    
    async def try_direct_tunebat_approach(self, title: str, artist: str) -> Dict[str, Any]:
        """Try to construct tunebat URL directly"""
        print("🔧 Trying direct TuneBat approach...")
        
        # Clean the title and artist for URL construction
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_artist = re.sub(r'[^\w\s-]', '', artist).strip()
        
        print(f"🔧 Cleaned title: '{clean_title}'")
        print(f"🔧 Cleaned artist: '{clean_artist}'")
        
        # Try common tunebat URL patterns
        possible_urls = [
            f"https://tunebat.com/Info/{clean_title.replace(' ', '-')}-{clean_artist.replace(' ', '-')}",
            f"https://tunebat.com/Info/{clean_artist.replace(' ', '-')}-{clean_title.replace(' ', '-')}",
        ]
        
        for i, url in enumerate(possible_urls):
            print(f"🔧 Trying direct URL {i+1}: {url}")
            try:
                features = await self.scrape_tunebat_page(url, title, artist)
                if features:  # If we got any features, return them
                    print(f"✅ Direct URL {i+1} worked!")
                    return features
            except Exception as e:
                print(f"❌ Direct URL {i+1} failed: {str(e)}")
                continue
        
        print("❌ All direct URLs failed")
        return {}

# Initialize the service with API keys
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
audio_service = AudioFeatureService(BRAVE_API_KEY, OPENAI_API_KEY)

@audio_feature_router.get("/search", response_model=AudioFeature)
async def get_audio_features(
    title: str = Query(..., description="Song title"),
    artist: str = Query(..., description="Artist name")
):
    """
    Get audio features for a song by searching tunebat using AI extraction
    
    - **title**: The song title
    - **artist**: The artist name
    """
    try:
        audio_features = await audio_service.get_audio_features(title, artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@audio_feature_router.post("/search", response_model=AudioFeature)
async def get_audio_features_post(request: AudioFeatureRequest):
    """
    Get audio features for a song by searching tunebat using AI extraction (POST method)
    """
    try:
        audio_features = await audio_service.get_audio_features(request.title, request.artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in POST endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Health check endpoint
@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio_feature_router_ai_only"}