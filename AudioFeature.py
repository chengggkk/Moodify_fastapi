from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, Any
import asyncio
import json
import random
import os

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
    def __init__(self, brave_api_key: Optional[str]):
        self.brave_api_key = brave_api_key
        self.brave_search_url = "https://api.search.brave.com/res/v1/web/search"
        self.use_api = brave_api_key is not None and brave_api_key != "YOUR_BRAVE_API_KEY_HERE"
    
    async def search_tunebat(self, title: str, artist: str) -> str:
        """Search for tunebat results using Brave Search API"""
        if not self.brave_api_key or self.brave_api_key == "YOUR_BRAVE_API_KEY_HERE":
            raise HTTPException(
                status_code=500, 
                detail="Brave Search API key not configured. Please set a valid API key."
            )
        
        query = f"tunebat {title} {artist}"
        
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
                response = await client.get(
                    self.brave_search_url,
                    headers=headers,
                    params=params,
                    timeout=15.0
                )
                
                # Handle specific error codes
                if response.status_code == 422:
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
                    raise HTTPException(
                        status_code=401, 
                        detail="Unauthorized: Invalid or expired Brave Search API key"
                    )
                elif response.status_code == 429:
                    raise HTTPException(
                        status_code=429, 
                        detail="Rate limit exceeded. Please try again later."
                    )
                
                response.raise_for_status()
                
                search_results = response.json()
                
                if not search_results.get("web", {}).get("results"):
                    raise HTTPException(status_code=404, detail="No search results found")
                
                first_result = search_results["web"]["results"][0]
                return first_result.get("url", "")
                
            except httpx.HTTPError as e:
                if "422" in str(e):
                    raise HTTPException(
                        status_code=422, 
                        detail="Brave Search API configuration error. Please check your API key and subscription status."
                    )
                raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    
    async def scrape_tunebat_page(self, url: str) -> Dict[str, Any]:
        """Scrape audio features from tunebat page with proper HTML structure parsing"""
        if not url:
            raise HTTPException(status_code=404, detail="No URL found in search results")
        
        # Multiple user agents to rotate
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        # Headers to mimic a real browser request
        headers = {
            'User-Agent': random.choice(user_agents),
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
            'Referer': 'https://www.google.com/'  # Appear to come from Google
        }
        
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=20.0
        ) as client:
            try:
                # Add random delay to appear more human-like
                await asyncio.sleep(random.uniform(1, 3))
                
                response = await client.get(url)
                
                if response.status_code == 403:
                    raise HTTPException(
                        status_code=403, 
                        detail="Access denied by tunebat. Try using a VPN or consider alternative data sources like Spotify Web API."
                    )
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract audio features from tunebat page
                features = {}
                
                # Method 1: Look for Ant Design progress circles (based on your paste-2.txt)
                progress_circles = soup.find_all('div', class_='ant-progress-circle')
                
                for circle in progress_circles:
                    # Find the value in the progress text span
                    progress_text = circle.find('span', class_='ant-progress-text')
                    if progress_text:
                        value_text = progress_text.get('title', '').strip()
                        
                        # Find the corresponding label
                        label_element = circle.find_next('span', class_='ant-typography')
                        if label_element:
                            label = label_element.get_text().strip().lower()
                            
                            # Parse the value based on the label
                            if label == 'popularity' and value_text.isdigit():
                                features['popularity'] = int(value_text)
                            elif label == 'energy' and value_text.isdigit():
                                features['energy'] = int(value_text)
                            elif label == 'danceability' and value_text.isdigit():
                                features['danceability'] = int(value_text)
                            elif label == 'happiness' and value_text.isdigit():
                                features['happiness'] = int(value_text)
                            elif label == 'acousticness' and value_text.isdigit():
                                features['acousticness'] = int(value_text)
                            elif label == 'instrumentalness' and value_text.isdigit():
                                features['instrumentalness'] = int(value_text)
                            elif label == 'liveness' and value_text.isdigit():
                                features['liveness'] = int(value_text)
                            elif label == 'speechiness' and value_text.isdigit():
                                features['speechiness'] = int(value_text)
                            elif label == 'loudness' and 'db' in value_text.lower():
                                features['loudness'] = value_text
                
                # Method 2: Look for standard TuneBat layout patterns
                # BPM - look for various patterns
                bpm_patterns = [
                    r'(\d+)\s*BPM',
                    r'BPM[:\s]*(\d+)',
                    r'Tempo[:\s]*(\d+)',
                    r'(\d+)\s*beats per minute'
                ]
                
                page_text = soup.get_text()
                for pattern in bpm_patterns:
                    bpm_match = re.search(pattern, page_text, re.IGNORECASE)
                    if bpm_match and not features.get('bpm'):
                        features['bpm'] = int(bpm_match.group(1))
                        break
                
                # Key - look for musical keys
                key_patterns = [
                    r'Key[:\s]*([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)',
                    r'([A-G][#♯♭b]?)\s*(?:major|minor|maj|min)',
                    r'Key:\s*([A-G][#♯♭b]?)'
                ]
                
                for pattern in key_patterns:
                    key_match = re.search(pattern, page_text, re.IGNORECASE)
                    if key_match and not features.get('key'):
                        features['key'] = key_match.group(1).strip()
                        break
                
                # Time Signature
                time_sig_patterns = [
                    r'Time Signature[:\s]*(\d+/\d+)',
                    r'(\d+/\d+)\s*time',
                    r'Time[:\s]*(\d+/\d+)'
                ]
                
                for pattern in time_sig_patterns:
                    time_match = re.search(pattern, page_text, re.IGNORECASE)
                    if time_match and not features.get('time_signature'):
                        features['time_signature'] = time_match.group(1)
                        break
                
                # Camelot Key
                camelot_patterns = [
                    r'Camelot[:\s]*(\d+[AB])',
                    r'(\d+[AB])\s*Camelot',
                    r'Camelot Key[:\s]*(\d+[AB])'
                ]
                
                for pattern in camelot_patterns:
                    camelot_match = re.search(pattern, page_text, re.IGNORECASE)
                    if camelot_match and not features.get('camelot'):
                        features['camelot'] = camelot_match.group(1)
                        break
                
                # Method 3: Look for structured data (JSON-LD, microdata, etc.)
                json_scripts = soup.find_all('script', type='application/ld+json')
                for script in json_scripts:
                    try:
                        data = json.loads(script.string)
                        # Extract any relevant audio features from structured data
                        if isinstance(data, dict):
                            # Look for common schema.org properties
                            if 'tempo' in data:
                                features['bpm'] = int(data['tempo'])
                            if 'musicalKey' in data:
                                features['key'] = data['musicalKey']
                    except:
                        continue
                
                # Method 4: Look for table-based data
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text().strip().lower()
                            value = cells[1].get_text().strip()
                            
                            if 'bpm' in label or 'tempo' in label:
                                bpm_match = re.search(r'(\d+)', value)
                                if bpm_match and not features.get('bpm'):
                                    features['bpm'] = int(bpm_match.group(1))
                            
                            elif 'key' in label:
                                key_match = re.search(r'([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)', value)
                                if key_match and not features.get('key'):
                                    features['key'] = key_match.group(1).strip()
                
                # Method 5: Look for div/span elements with specific classes or data attributes
                # Common TuneBat-style selectors
                selectors_map = {
                    'bpm': ['.bpm', '.tempo', '[data-bpm]', '.track-bpm'],
                    'key': ['.key', '.musical-key', '[data-key]', '.track-key'],
                    'energy': ['.energy', '[data-energy]'],
                    'danceability': ['.danceability', '[data-danceability]'],
                    'camelot': ['.camelot', '.camelot-key', '[data-camelot]']
                }
                
                for feature, selectors in selectors_map.items():
                    if features.get(feature):
                        continue
                        
                    for selector in selectors:
                        elements = soup.select(selector)
                        for element in elements:
                            text = element.get_text().strip()
                            
                            if feature == 'bpm':
                                bpm_match = re.search(r'(\d+)', text)
                                if bpm_match:
                                    features['bpm'] = int(bpm_match.group(1))
                                    break
                            
                            elif feature == 'key':
                                key_match = re.search(r'([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)', text)
                                if key_match:
                                    features['key'] = key_match.group(1).strip()
                                    break
                            
                            elif feature in ['energy', 'danceability'] and text.isdigit():
                                features[feature] = int(text)
                                break
                            
                            elif feature == 'camelot':
                                camelot_match = re.search(r'(\d+[AB])', text)
                                if camelot_match:
                                    features['camelot'] = camelot_match.group(1)
                                    break
                        
                        if features.get(feature):
                            break
                
                return features
                
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Failed to scrape page: {str(e)}")
    
    async def get_audio_features(self, title: str, artist: str) -> AudioFeature:
        """Main method to get audio features with fallback strategies"""
        features = {}
        
        if self.use_api:
            try:
                # Primary strategy: Search tunebat via Brave API
                url = await self.search_tunebat(title, artist)
                features = await self.scrape_tunebat_page(url)
            except HTTPException as e:
                if e.status_code in [422, 401, 500]:
                    # Fallback to non-API methods
                    features = await self.try_direct_tunebat_approach(title, artist)
                else:
                    raise e
        else:
            # Skip API and go directly to fallback methods
            features = await self.try_direct_tunebat_approach(title, artist)
        
        return AudioFeature(
            title=title,
            artist=artist,
            **features
        )
    
    async def try_direct_tunebat_approach(self, title: str, artist: str) -> Dict[str, Any]:
        """Try to construct tunebat URL directly or use DuckDuckGo search"""
        # Clean the title and artist for URL construction
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_artist = re.sub(r'[^\w\s-]', '', artist).strip()
        
        # Try common tunebat URL patterns
        possible_urls = [
            f"https://tunebat.com/Info/{clean_title.replace(' ', '-')}-{clean_artist.replace(' ', '-')}",
            f"https://tunebat.com/Info/{clean_artist.replace(' ', '-')}-{clean_title.replace(' ', '-')}",
        ]
        
        for url in possible_urls:
            try:
                features = await self.scrape_tunebat_page(url)
                if features:  # If we got any features, return them
                    return features
            except:
                continue
        
        # If direct URLs don't work, try DuckDuckGo search (no API key required)
        return await self.search_with_duckduckgo(title, artist)
    
    async def search_with_duckduckgo(self, title: str, artist: str) -> Dict[str, Any]:
        """Use DuckDuckGo search as fallback (no API key required)"""
        query = f"site:tunebat.com {title} {artist}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            try:
                # DuckDuckGo instant answer API
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for tunebat URLs in results
                    for result in data.get("Results", []):
                        url = result.get("FirstURL", "")
                        if "tunebat.com" in url:
                            try:
                                return await self.scrape_tunebat_page(url)
                            except:
                                continue
                
                return {}
                
            except:
                return {}
    
    async def search_without_api(self, title: str, artist: str) -> Dict[str, Any]:
        """Last resort: try to get basic info from other sources"""
        # This could be expanded to include other music databases
        # For now, return empty dict
        return {}
    
    async def search_alternative_sources(self, title: str, artist: str) -> Dict[str, Any]:
        """Search for audio features from alternative sources"""
        query = f'"{title}" "{artist}" BPM key tempo'
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": 5,  # Get more results to find good sources
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.brave_search_url,
                    headers=headers,
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                
                search_results = response.json()
                features = {}
                
                # Look through search results for music data
                for result in search_results.get("web", {}).get("results", []):
                    snippet = result.get("description", "")
                    title_text = result.get("title", "")
                    
                    # Try to extract BPM from snippets
                    bpm_match = re.search(r'(\d+)\s*BPM', snippet + " " + title_text, re.IGNORECASE)
                    if bpm_match and not features.get('bpm'):
                        features['bpm'] = int(bpm_match.group(1))
                    
                    # Try to extract key
                    key_match = re.search(r'Key[:\s]*([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)', snippet + " " + title_text, re.IGNORECASE)
                    if key_match and not features.get('key'):
                        features['key'] = key_match.group(1).strip()
                    
                    # If we found basic info, that's good enough for fallback
                    if features.get('bpm') and features.get('key'):
                        break
                
                return features
                
            except Exception:
                return {}

# Initialize the service with fallback options
# You can get a Brave API key from https://api.search.brave.com/
# Or set to None to use fallback methods only
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
audio_service = AudioFeatureService(BRAVE_API_KEY)

@audio_feature_router.get("/search", response_model=AudioFeature)
async def get_audio_features(
    title: str = Query(..., description="Song title"),
    artist: str = Query(..., description="Artist name")
):
    """
    Get audio features for a song by searching tunebat
    
    - **title**: The song title
    - **artist**: The artist name
    """
    try:
        audio_features = await audio_service.get_audio_features(title, artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@audio_feature_router.post("/search", response_model=AudioFeature)
async def get_audio_features_post(request: AudioFeatureRequest):
    """
    Get audio features for a song by searching tunebat (POST method)
    """
    try:
        audio_features = await audio_service.get_audio_features(request.title, request.artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Health check endpoint
@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio_feature_router"}