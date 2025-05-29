from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from urllib.parse import quote, urljoin
import logging
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

# Pydantic models for request/response
class SpotifyUrlRequest(BaseModel):
    spotify_url: str
    
    @validator('spotify_url')
    def validate_spotify_url(cls, v):
        patterns = [
            r'spotify\.com/track/([a-zA-Z0-9]+)',
            r'spotify:track:([a-zA-Z0-9]+)',
            r'open\.spotify\.com/track/([a-zA-Z0-9]+)',
        ]
        
        if not any(re.search(pattern, v) for pattern in patterns):
            raise ValueError('Invalid Spotify URL format')
        return v

class BatchAnalysisRequest(BaseModel):
    spotify_urls: List[str]
    
    @validator('spotify_urls')
    def validate_urls(cls, v):
        if len(v) > 50:  # Limit batch size
            raise ValueError('Maximum 50 URLs allowed per batch')
        
        patterns = [
            r'spotify\.com/track/([a-zA-Z0-9]+)',
            r'spotify:track:([a-zA-Z0-9]+)',
            r'open\.spotify\.com/track/([a-zA-Z0-9]+)',
        ]
        
        for url in v:
            if not any(re.search(pattern, url) for pattern in patterns):
                raise ValueError(f'Invalid Spotify URL format: {url}')
        return v

class AudioFeaturesResponse(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    key: Optional[str] = None
    bpm: Optional[int] = None
    camelot: Optional[str] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None
    popularity: Optional[int] = None
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    loudness: Optional[int] = None
    duration_ms: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Shape of You",
                "artist": "Ed Sheeran",
                "key": "C# Minor",
                "bpm": 96,
                "camelot": "4A",
                "energy": 65,
                "danceability": 83,
                "happiness": 93,
                "popularity": 74,
                "acousticness": 58,
                "instrumentalness": 0,
                "liveness": 9,
                "speechiness": 8,
                "loudness": -3,
                "duration_ms": 233713
            }
        }

class AnalysisResult(BaseModel):
    spotify_url: str
    success: bool
    features: Optional[AudioFeaturesResponse] = None
    error: Optional[str] = None
    analyzed_at: str

class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_urls: int
    successful: int
    failed: int
    results: List[AnalysisResult]

# Enhanced TuneBat Scraper (optimized for async usage)
@dataclass
class AudioFeatures:
    title: Optional[str] = None
    artist: Optional[str] = None
    key: Optional[str] = None
    bmp: Optional[int] = None
    camelot: Optional[str] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None
    popularity: Optional[int] = None
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    loudness: Optional[int] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class TuneBatScraper:
    def __init__(self, delay_between_requests: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.base_url = "https://tunebat.com"
        self.delay = delay_between_requests
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_spotify_id(self, spotify_url: str) -> str:
        """Extract Spotify track ID from various URL formats"""
        patterns = [
            r'spotify\.com/track/([a-zA-Z0-9]+)',
            r'spotify:track:([a-zA-Z0-9]+)',
            r'open\.spotify\.com/track/([a-zA-Z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, spotify_url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Invalid Spotify URL format: {spotify_url}")
    
    def get_track_by_direct_url(self, spotify_id: str) -> Optional[AudioFeatures]:
        """Try to get track info using direct TuneBat URL"""
        try:
            track_url = f"{self.base_url}/Info/{spotify_id}"
            self.logger.info(f"Trying direct URL: {track_url}")
            
            response = self.session.get(track_url, timeout=15)
            response.raise_for_status()
            
            if "404" in response.text or "not found" in response.text.lower():
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._parse_track_page(soup)
            
        except Exception as e:
            self.logger.warning(f"Direct URL failed: {e}")
            return None
    
    def _parse_track_page(self, soup: BeautifulSoup) -> AudioFeatures:
        """Parse track page and extract audio features"""
        features = AudioFeatures()
        
        try:
            # Title and artist from headers
            h1 = soup.find('h1')
            if h1:
                features.title = h1.get_text().strip()
            
            h2 = soup.find('h2') 
            if h2:
                features.artist = h2.get_text().strip()
            
            # Parse page text for features
            page_text = soup.get_text()
            
            # Key patterns
            key_patterns = [
                re.compile(r'([A-G]#?)\s*(Major|Minor)', re.I),
                re.compile(r'Key[:\s]+([A-G]#?\s*(?:Major|Minor|maj|min))', re.I)
            ]
            
            for pattern in key_patterns:
                if not features.key:
                    match = pattern.search(page_text)
                    if match:
                        if len(match.groups()) == 2:
                            key = match.group(1)
                            scale = match.group(2)
                            features.key = f"{key} {scale.title()}"
                        else:
                            features.key = match.group(1).strip()
                        break
            
            # BPM patterns
            bpm_patterns = [
                re.compile(r'(\d+)\s*BPM', re.I),
                re.compile(r'BPM[:\s]+(\d+)', re.I),
                re.compile(r'Tempo[:\s]+(\d+)', re.I)
            ]
            
            for pattern in bpm_patterns:
                if not features.bpm:
                    match = pattern.search(page_text)
                    if match:
                        features.bpm = int(match.group(1))
                        break
            
            # Audio feature patterns
            feature_patterns = {
                'energy': re.compile(r'Energy[:\s]+(\d+)', re.I),
                'danceability': re.compile(r'Danceability[:\s]+(\d+)', re.I),
                'happiness': re.compile(r'(?:Happiness|Valence)[:\s]+(\d+)', re.I),
                'popularity': re.compile(r'Popularity[:\s]+(\d+)', re.I),
                'acousticness': re.compile(r'Acousticness[:\s]+(\d+)', re.I),
                'instrumentalness': re.compile(r'Instrumentalness[:\s]+(\d+)', re.I),
                'liveness': re.compile(r'Liveness[:\s]+(\d+)', re.I),
                'speechiness': re.compile(r'Speechiness[:\s]+(\d+)', re.I),
                'loudness': re.compile(r'Loudness[:\s]+(-?\d+)', re.I)
            }
            
            for feature, pattern in feature_patterns.items():
                match = pattern.search(page_text)
                if match:
                    setattr(features, feature, int(match.group(1)))
            
            # Camelot notation
            camelot_match = re.search(r'(\d+[AB])', page_text)
            if camelot_match:
                features.camelot = camelot_match.group(1)
            
            # Duration pattern (MM:SS)
            duration_match = re.search(r'(\d+):(\d+)', page_text)
            if duration_match:
                minutes = int(duration_match.group(1))
                seconds = int(duration_match.group(2))
                features.duration_ms = (minutes * 60 + seconds) * 1000
                
        except Exception as e:
            self.logger.error(f"Error parsing track page: {e}")
        
        return features
    
    def analyze_spotify_url(self, spotify_url: str) -> Optional[AudioFeatures]:
        """Main method to analyze a Spotify URL"""
        try:
            self.logger.info(f"Analyzing: {spotify_url}")
            
            # Extract Spotify ID
            spotify_id = self.extract_spotify_id(spotify_url)
            
            # Try direct URL
            features = self.get_track_by_direct_url(spotify_id)
            
            if features and features.title:
                self.logger.info("Success with direct URL")
                return features
            
            self.logger.warning(f"Could not analyze: {spotify_url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {spotify_url}: {e}")
            return None
        
        finally:
            time.sleep(self.delay)

# Global scraper instance
scraper = TuneBatScraper(delay_between_requests=1.5)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)

# Router definition
audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

@audio_feature_router.get("/")
async def get_audio_features_root():
    """Root endpoint with API information"""
    return {
        "message": "TuneBat Audio Features API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze single Spotify URL",
            "batch": "POST /batch - Analyze multiple Spotify URLs",
            "health": "GET /health - Health check"
        }
    }

@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tunebat-scraper"}

@audio_feature_router.post("/analyze", response_model=AnalysisResult)
async def analyze_single_track(request: SpotifyUrlRequest):
    """
    Analyze a single Spotify URL and return audio features from TuneBat
    
    - **spotify_url**: Valid Spotify track URL
    """
    try:
        # Run scraping in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            executor, 
            scraper.analyze_spotify_url, 
            request.spotify_url
        )
        
        if features:
            return AnalysisResult(
                spotify_url=request.spotify_url,
                success=True,
                features=AudioFeaturesResponse(**features.to_dict()),
                analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            return AnalysisResult(
                spotify_url=request.spotify_url,
                success=False,
                error="Track not found in TuneBat database",
                analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@audio_feature_router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_batch_tracks(request: BatchAnalysisRequest):
    """
    Analyze multiple Spotify URLs in batch
    
    - **spotify_urls**: List of valid Spotify track URLs (max 50)
    """
    try:
        batch_id = str(uuid.uuid4())
        
        # Run batch analysis in thread pool
        loop = asyncio.get_event_loop()
        
        async def analyze_single(url: str) -> AnalysisResult:
            try:
                features = await loop.run_in_executor(
                    executor,
                    scraper.analyze_spotify_url,
                    url
                )
                
                if features:
                    return AnalysisResult(
                        spotify_url=url,
                        success=True,
                        features=AudioFeaturesResponse(**features.to_dict()),
                        analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                else:
                    return AnalysisResult(
                        spotify_url=url,
                        success=False,
                        error="Track not found in TuneBat database",
                        analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
            except Exception as e:
                return AnalysisResult(
                    spotify_url=url,
                    success=False,
                    error=str(e),
                    analyzed_at=time.strftime("%Y-%m-%d %H:%M:%S")
                )
        
        # Process all URLs concurrently (with some throttling)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
        
        async def throttled_analyze(url: str):
            async with semaphore:
                return await analyze_single(url)
        
        results = await asyncio.gather(*[
            throttled_analyze(url) for url in request.spotify_urls
        ])
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchAnalysisResponse(
            batch_id=batch_id,
            total_urls=len(request.spotify_urls),
            successful=successful,
            failed=failed,
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

@audio_feature_router.get("/analyze")
async def analyze_single_track_get(
    spotify_url: str = Query(..., description="Spotify track URL to analyze")
):
    """
    Alternative GET endpoint for single track analysis
    
    - **spotify_url**: Valid Spotify track URL as query parameter
    """
    request = SpotifyUrlRequest(spotify_url=spotify_url)
    return await analyze_single_track(request)

@audio_feature_router.get("/features/{spotify_id}")
async def get_features_by_id(spotify_id: str):
    """
    Get audio features by Spotify track ID
    
    - **spotify_id**: Spotify track ID (without spotify:track: prefix)
    """
    spotify_url = f"https://open.spotify.com/track/{spotify_id}"
    request = SpotifyUrlRequest(spotify_url=spotify_url)
    return await analyze_single_track(request)

# Example usage and testing endpoints (optional)
@audio_feature_router.get("/examples")
async def get_examples():
    """Get example Spotify URLs for testing"""
    return {
        "example_urls": [
            "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh",  # Shape of You
            "https://open.spotify.com/track/0tgVpDi06FyKpA1z0VMD4v",  # Perfect
            "https://open.spotify.com/track/6habFhsOp2NvshLv26DqMb",  # Blinding Lights
        ],
        "supported_formats": [
            "https://open.spotify.com/track/{id}",
            "https://spotify.com/track/{id}",
            "spotify:track:{id}"
        ]
    }

# Optional: Statistics endpoint
@audio_feature_router.get("/stats")
async def get_stats():
    """Get API usage statistics (placeholder)"""
    return {
        "status": "operational",
        "source": "TuneBat.com",
        "features_available": [
            "title", "artist", "key", "bpm", "camelot",
            "energy", "danceability", "happiness", "popularity",
            "acousticness", "instrumentalness", "liveness", 
            "speechiness", "loudness", "duration_ms"
        ],
        "rate_limit": "1.5 seconds between requests",
        "batch_limit": "50 URLs per batch"
    }