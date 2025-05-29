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

def download_audio_fallback(youtube_url: str, output_dir: str) -> Optional[str]:
    """Fallback download method - just get the best audio without conversion"""
    try:
        command = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-f", "bestaudio",
            "--output", f"{output_dir}/%(title)s.%(ext)s",
            youtube_url
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Fallback download successful for {youtube_url}")
        
        # Look for any audio files
        audio_files = []
        for ext in ["*.webm", "*.m4a", "*.mp3", "*.wav", "*.ogg"]:
            audio_files.extend(glob.glob(os.path.join(output_dir, ext)))
        
        if audio_files:
            return audio_files[0]
        else:
            logger.error(f"No audio file found after fallback download in {output_dir}")
            return None
            
    except Exception as e:
        logger.error(f"Fallback download failed for {youtube_url}: {str(e)}")
        return None
    """Download audio from YouTube URL using subprocess"""
    try:
        command = [
            "yt-dlp",
            "--cookies", "cookies.txt",
            "-x", "--audio-format", "mp3",
            "--output", f"{output_dir}/%(title)s.%(ext)s",
            "--prefer-ffmpeg",
            "--ffmpeg-location", "/usr/bin/ffmpeg",  # Adjust path as needed
            youtube_url
        ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Download successful for {youtube_url}")
        logger.debug(f"yt-dlp output: {result.stdout}")
        
        # Look for both mp3 and other audio files
        audio_files = []
        for ext in ["*.mp3", "*.webm", "*.m4a", "*.wav", "*.ogg"]:
            audio_files.extend(glob.glob(os.path.join(output_dir, ext)))
        
        if audio_files:
            audio_file = audio_files[0]
            logger.info(f"Found audio file: {audio_file}")
            return audio_file
        else:
            logger.error(f"No audio file found after download in {output_dir}")
            # List all files in directory for debugging
            all_files = os.listdir(output_dir)
            logger.error(f"Files in directory: {all_files}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed for {youtube_url}: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        logger.error(f"Command stdout: {e.stdout}")
        
        # Try to find any downloaded files even if conversion failed
        audio_files = []
        for ext in ["*.mp3", "*.webm", "*.m4a", "*.wav", "*.ogg"]:
            audio_files.extend(glob.glob(os.path.join(output_dir, ext)))
        
        if audio_files:
            logger.info(f"Found audio file despite error: {audio_files[0]}")
            return audio_files[0]
        
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {youtube_url}: {str(e)}")
        return None

def extract_features_from_audio_file(audio_path: str) -> Dict:
    """Extract audio features from local audio file using librosa"""
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=22050, duration=30)  # Limit to 30 seconds
        
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
    """Extract audio features from YouTube URL by downloading and processing"""
    temp_dir = None
    audio_path = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download audio using subprocess
        audio_path = download_audio(url, temp_dir)
        if not audio_path:
            # Try fallback method
            logger.info(f"Trying fallback download method for {url}")
            audio_path = download_audio_fallback(url, temp_dir)
            
        if not audio_path:
            raise ValueError("Failed to download audio from YouTube")
        
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
                # Remove all files in temp directory
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                # Remove the directory
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {cleanup_error}")

async def process_track_batch(tracks: List[TrackRequest]) -> List[TrackResult]:
    """Process a batch of 3 tracks synchronously"""
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
            
            # Step 3: Extract audio features using thread pool
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
        
        # Process tracks in batches of 3
        for i in range(0, total_tracks, 3):
            batch_tracks = batch.tracks[i:i+3]
            logger.info(f"Processing batch {i//3 + 1}: tracks {i+1}-{min(i+3, total_tracks)} of {total_tracks}")
            
            batch_results = await process_track_batch(batch_tracks)
            all_results.extend(batch_results)
            
            # Small delay between batches to be respectful to APIs
            if i + 3 < total_tracks:
                await asyncio.sleep(1)
        
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
    return {"status": "healthy", "service": "audio_feature_analyzer"}