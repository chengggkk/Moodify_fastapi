from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, Any
import asyncio
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
    def __init__(self, brave_api_key: str):
        self.brave_api_key = brave_api_key
        self.brave_search_url = "https://api.search.brave.com/res/v1/web/search"
    
    async def search_tunebat(self, title: str, artist: str) -> str:
        """Search for tunebat results using Brave Search API"""
        query = f"tunebat {title} {artist}"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": 1,  # We only need the first result
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
                
                if not search_results.get("web", {}).get("results"):
                    raise HTTPException(status_code=404, detail="No search results found")
                
                first_result = search_results["web"]["results"][0]
                return first_result.get("url", "")
                
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    
    async def scrape_tunebat_page(self, url: str) -> Dict[str, Any]:
        """Scrape audio features from tunebat page"""
        if not url:
            raise HTTPException(status_code=404, detail="No URL found in search results")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract audio features from tunebat page
                features = {}
                
                # Look for key musical attributes
                # These selectors might need adjustment based on tunebat's actual HTML structure
                
                # BPM
                bpm_element = soup.find(text=re.compile(r'BPM|Tempo'))
                if bpm_element:
                    bpm_parent = bpm_element.parent
                    bpm_match = re.search(r'(\d+)', bpm_parent.get_text())
                    if bpm_match:
                        features['bpm'] = int(bpm_match.group(1))
                
                # Key
                key_element = soup.find(text=re.compile(r'Key'))
                if key_element:
                    key_parent = key_element.parent
                    key_match = re.search(r'([A-G][#b]?\s*(?:major|minor|maj|min)?)', key_parent.get_text())
                    if key_match:
                        features['key'] = key_match.group(1).strip()
                
                # Time Signature
                time_sig_element = soup.find(text=re.compile(r'Time Signature'))
                if time_sig_element:
                    time_sig_parent = time_sig_element.parent
                    time_sig_match = re.search(r'(\d+/\d+)', time_sig_parent.get_text())
                    if time_sig_match:
                        features['time_signature'] = time_sig_match.group(1)
                
                # Camelot Key
                camelot_element = soup.find(text=re.compile(r'Camelot'))
                if camelot_element:
                    camelot_parent = camelot_element.parent
                    camelot_match = re.search(r'(\d+[AB])', camelot_parent.get_text())
                    if camelot_match:
                        features['camelot'] = camelot_match.group(1)
                
                # Additional attributes (these might be percentages or ratings)
                attribute_patterns = {
                    'energy': r'Energy.*?(\d+)',
                    'danceability': r'Danceability.*?(\d+)',
                    'happiness': r'Happiness.*?(\d+)',
                    'acousticness': r'Acousticness.*?(\d+)',
                    'instrumentalness': r'Instrumentalness.*?(\d+)',
                    'liveness': r'Liveness.*?(\d+)',
                    'speechiness': r'Speechiness.*?(\d+)',
                    'popularity': r'Popularity.*?(\d+)'
                }
                
                page_text = soup.get_text()
                for attr, pattern in attribute_patterns.items():
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match:
                        features[attr] = int(match.group(1))
                
                # Loudness (might be in dB)
                loudness_match = re.search(r'Loudness.*?(-?\d+(?:\.\d+)?\s*dB)', page_text, re.IGNORECASE)
                if loudness_match:
                    features['loudness'] = loudness_match.group(1)
                
                return features
                
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Failed to scrape page: {str(e)}")
    
    async def get_audio_features(self, title: str, artist: str) -> AudioFeature:
        """Main method to get audio features"""
        # Step 1: Search for tunebat URL
        url = await self.search_tunebat(title, artist)
        
        # Step 2: Scrape the page for audio features
        features = await self.scrape_tunebat_page(url)
        
        # Step 3: Return structured audio features
        return AudioFeature(
            title=title,
            artist=artist,
            **features
        )

# Initialize the service (you'll need to provide your Brave API key)
# You can get this from environment variables or configuration
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