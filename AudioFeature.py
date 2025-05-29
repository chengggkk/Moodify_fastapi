# audio_feature_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import aiohttp
import subprocess
import librosa
import numpy as np
import tempfile
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import glob
import shutil


logger = logging.getLogger(__name__)

audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

# Pydantic models
class TrackRequest(BaseModel):
    title: str
    artist: str

class TrackBatch(BaseModel):
    tracks: List[TrackRequest]

class AudioFeatures(BaseModel):
    tempo: float
    energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    mfcc_mean: List[float]
    chroma_mean: List[float]
    rolloff: float
    rms_energy: float

class TrackResult(BaseModel):
    title: str
    artist: str
    youtube_url: Optional[str]
    audio_features: Optional[AudioFeatures]
    error: Optional[str]

class BatchResponse(BaseModel):
    results: List[TrackResult]
    total_processed: int
    successful: int
    failed: int

# Configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Thread pool for CPU-intensive audio processing
executor = ThreadPoolExecutor(max_workers=3)

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'yt-dlp': False,
        'ffmpeg': False
    }
    
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        dependencies['yt-dlp'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("yt-dlp not found or not working")
    
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        dependencies['ffmpeg'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not found or not working")
    
    logger.info(f"Dependencies check: {dependencies}")
    return dependencies

async def search_youtube_url(session: aiohttp.ClientSession, title: str, artist: str) -> Optional[str]:
    """Search for YouTube URL using Brave Search API"""
    try:
        query = f"{title} {artist} site:youtube.com"
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": 3,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
            "freshness": "all"
        }
        
        async with session.get(BRAVE_SEARCH_URL, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                # Look for YouTube URLs in search results
                if "web" in data and "results" in data["web"]:
                    for result in data["web"]["results"]:
                        url = result.get("url", "")
                        if "youtube.com/watch" in url or "youtu.be/" in url:
                            logger.info(f"Found YouTube URL for {title} by {artist}: {url}")
                            return url
                
                logger.warning(f"No YouTube URL found for {title} by {artist}")
                return None
            else:
                logger.error(f"Brave Search API error: {response.status}")
                return None
                
    except Exception as e:
        logger.error(f"Error searching for {title} by {artist}: {str(e)}")
        return None

def download_with_cookies_from_browser(youtube_url: str, output_dir: str, browser: str = "chrome") -> Optional[str]:
    """Download using cookies extracted directly from browser"""
    try:
        command = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "--output", f"{output_dir}/%(title)s.%(ext)s",
            "--ffmpeg-location", shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
            "--prefer-ffmpeg",
            "--no-warnings",
            youtube_url
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info(f"Successfully downloaded with {browser} cookies: {youtube_url}")
            return find_audio_file(output_dir)
        else:
            logger.warning(f"Failed to download with {browser} cookies: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Download timeout for {youtube_url}")
        return None
    except Exception as e:
        logger.error(f"Error downloading with {browser} cookies: {str(e)}")
        return None

def download_without_cookies(youtube_url: str, output_dir: str) -> Optional[str]:
    """Download without cookies using various user agents and methods"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ]
    
    for i, user_agent in enumerate(user_agents):
        try:
            command = [
                "yt-dlp",
                "--user-agent", user_agent,
                "-x", "--audio-format", "mp3",
                "--audio-quality", "0",
                "--output", f"{output_dir}/%(title)s.%(ext)s",
                "--ffmpeg-location", shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
                "--prefer-ffmpeg",
                "--no-warnings",
                "--extractor-retries", "3",
                "--fragment-retries", "3",
                youtube_url
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded without cookies (attempt {i+1}): {youtube_url}")
                return find_audio_file(output_dir)
            else:
                logger.warning(f"Attempt {i+1} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout on attempt {i+1} for {youtube_url}")
            continue
        except Exception as e:
            logger.warning(f"Attempt {i+1} error: {str(e)}")
            continue
    
    return None

def download_best_audio_only(youtube_url: str, output_dir: str) -> Optional[str]:
    """Download best audio without conversion, then convert manually with ffmpeg"""
    try:
        # First, download the best audio format available
        command = [
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
            "--output", f"{output_dir}/%(title)s.%(ext)s",
            "--no-warnings",
            "--extractor-retries", "5",
            youtube_url
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"Failed to download audio: {result.stderr}")
            return None
        
        # Find the downloaded file
        audio_file = find_audio_file(output_dir)
        if not audio_file:
            return None
        
        # Convert to MP3 using ffmpeg directly
        output_mp3 = os.path.join(output_dir, "converted_audio.mp3")
        ffmpeg_command = [
            "ffmpeg",
            "-i", audio_file,
            "-acodec", "libmp3lame",
            "-ab", "192k",
            "-ar", "44100",
            "-y",  # Overwrite output file
            output_mp3
        ]
        
        ffmpeg_result = subprocess.run(ffmpeg_command, capture_output=True, text=True, timeout=60)
        
        if ffmpeg_result.returncode == 0 and os.path.exists(output_mp3):
            logger.info(f"Successfully converted to MP3: {output_mp3}")
            # Remove original file to save space
            try:
                os.remove(audio_file)
            except:
                pass
            return output_mp3
        else:
            logger.warning(f"FFmpeg conversion failed, using original file: {audio_file}")
            return audio_file
            
    except subprocess.TimeoutExpired:
        logger.error(f"Download timeout for {youtube_url}")
        return None
    except Exception as e:
        logger.error(f"Error in best audio download: {str(e)}")
        return None

def find_audio_file(directory: str) -> Optional[str]:
    """Find any audio file in the directory"""
    audio_extensions = ["*.mp3", "*.webm", "*.m4a", "*.wav", "*.ogg", "*.aac"]
    
    for ext in audio_extensions:
        files = glob.glob(os.path.join(directory, ext))
        if files:
            return files[0]
    
    return None

def download_audio_comprehensive(youtube_url: str, output_dir: str) -> Optional[str]:
    """Comprehensive download strategy with multiple fallbacks"""
    logger.info(f"Starting comprehensive download for: {youtube_url}")
    
    # Strategy 1: Try with browser cookies (Chrome first, then Firefox)
    for browser in ["chrome", "firefox", "edge", "safari"]:
        logger.info(f"Trying {browser} cookies...")
        result = download_with_cookies_from_browser(youtube_url, output_dir, browser)
        if result:
            return result
        # Small delay between attempts
        import time
        time.sleep(1)
    
    # Strategy 2: Try without cookies with different user agents
    logger.info("Trying without cookies...")
    result = download_without_cookies(youtube_url, output_dir)
    if result:
        return result
    
    # Strategy 3: Download best audio and convert manually
    logger.info("Trying best audio download with manual conversion...")
    result = download_best_audio_only(youtube_url, output_dir)
    if result:
        return result
    
    logger.error(f"All download strategies failed for: {youtube_url}")
    return None

def extract_features_from_audio_file(audio_path: str) -> Dict:
    """Extract audio features from local audio file using librosa"""
    try:
        # Load audio with librosa (supports many formats)
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        # Extract comprehensive audio features
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        rms_energy = np.sqrt(np.mean(y**2))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = np.mean(spectral_centroids)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zero_crossing_rate = np.mean(zcr)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff = np.mean(rolloff)
        
        return {
            "tempo": float(tempo),
            "energy": float(energy),
            "spectral_centroid": float(spectral_centroid),
            "zero_crossing_rate": float(zero_crossing_rate),
            "mfcc_mean": mfcc_mean,
            "chroma_mean": chroma_mean,
            "rolloff": float(spectral_rolloff),
            "rms_energy": float(rms_energy)
        }
        
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {str(e)}")
        raise

def extract_features_from_youtube(url: str) -> Dict:
    """Extract audio features from YouTube URL using comprehensive download strategy"""
    temp_dir = None
    
    try:
        # Check dependencies first
        deps = check_dependencies()
        if not deps['yt-dlp']:
            raise ValueError("yt-dlp is not available")
        if not deps['ffmpeg']:
            logger.warning("ffmpeg not found - audio conversion may fail")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Use comprehensive download strategy
        audio_path = download_audio_comprehensive(url, temp_dir)
        
        if not audio_path:
            raise ValueError("Failed to download audio from YouTube using all available methods")
        
        logger.info(f"Processing audio file: {audio_path}")
        
        # Extract features from the downloaded file
        features = extract_features_from_audio_file(audio_path)
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing YouTube URL {url}: {str(e)}")
        raise
    
    finally:
        # Cleanup temporary files and directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {cleanup_error}")

async def process_track_batch(tracks: List[TrackRequest]) -> List[TrackResult]:
    """Process a batch of tracks synchronously"""
    results = []
    
    # Step 1: Search for YouTube URLs for all tracks in batch
    async with aiohttp.ClientSession() as session:
        search_tasks = [
            search_youtube_url(session, track.title, track.artist) 
            for track in tracks
        ]
        youtube_urls = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Step 2: Process each track with its YouTube URL
    for i, track in enumerate(tracks):
        result = TrackResult(
            title=track.title,
            artist=track.artist,
            youtube_url=None,
            audio_features=None,
            error=None
        )
        
        try:
            # Handle search result
            if isinstance(youtube_urls[i], Exception):
                result.error = f"Search error: {str(youtube_urls[i])}"
                results.append(result)
                continue
            
            youtube_url = youtube_urls[i]
            if not youtube_url:
                result.error = "No YouTube URL found"
                results.append(result)
                continue
            
            result.youtube_url = youtube_url
            
            # Extract audio features using thread pool
            loop = asyncio.get_event_loop()
            features_dict = await loop.run_in_executor(
                executor, 
                extract_features_from_youtube, 
                youtube_url
            )
            
            result.audio_features = AudioFeatures(**features_dict)
            logger.info(f"Successfully processed {track.title} by {track.artist}")
            
        except Exception as e:
            result.error = f"Processing error: {str(e)}"
            logger.error(f"Error processing {track.title} by {track.artist}: {str(e)}")
        
        results.append(result)
    
    return results

@audio_feature_router.post("/analyze", response_model=BatchResponse)
async def analyze_audio_features(batch: TrackBatch):
    """
    Analyze audio features for a batch of tracks.
    Processes tracks in groups of 3 synchronously.
    """
    try:
        all_results = []
        total_tracks = len(batch.tracks)
        
        # Check dependencies at startup
        deps = check_dependencies()
        if not deps['yt-dlp']:
            raise HTTPException(status_code=500, detail="yt-dlp is not available")
        
        # Process tracks in batches of 3
        for i in range(0, total_tracks, 3):
            batch_tracks = batch.tracks[i:i+3]
            logger.info(f"Processing batch {i//3 + 1}: tracks {i+1}-{min(i+3, total_tracks)} of {total_tracks}")
            
            batch_results = await process_track_batch(batch_tracks)
            all_results.extend(batch_results)
            
            # Delay between batches to avoid rate limiting
            if i + 3 < total_tracks:
                await asyncio.sleep(2)
        
        # Calculate statistics
        successful = len([r for r in all_results if r.audio_features is not None])
        failed = len([r for r in all_results if r.error is not None])
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed out of {total_tracks}")
        
        return BatchResponse(
            results=all_results,
            total_processed=total_tracks,
            successful=successful,
            failed=failed
        )
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@audio_feature_router.post("/analyze_single", response_model=TrackResult)
async def analyze_single_track(track: TrackRequest):
    """Analyze audio features for a single track"""
    try:
        batch_results = await process_track_batch([track])
        return batch_results[0]
    except Exception as e:
        logger.error(f"Error processing single track: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Single track processing failed: {str(e)}")

@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    deps = check_dependencies()
    return {
        "status": "healthy" if deps['yt-dlp'] else "degraded",
        "service": "audio_feature_analyzer",
        "dependencies": deps
    }