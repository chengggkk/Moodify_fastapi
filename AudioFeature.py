# audio_feature_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
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
from ytmusicapi import YTMusic


logger = logging.getLogger(__name__)

audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])
ytmusic = YTMusic()  # Anonymous mode (no cookies)

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
    youtube_music_url: Optional[str] = None
    video_id: Optional[str] = None
    duration: Optional[str] = None
    audio_features: Optional[AudioFeatures]
    error: Optional[str]

class BatchResponse(BaseModel):
    results: List[TrackResult]
    total_processed: int
    successful: int
    failed: int

# Thread pool for CPU-intensive audio processing
executor = ThreadPoolExecutor(max_workers=3)

def download_song_from_youtube_music(title: str, artist: str):
    """Download song from YouTube Music using ytmusicapi + yt-dlp - YouTube Music ONLY"""
    try:
        # 1. Search on YouTube Music specifically
        results = ytmusic.search(query=f"{title} {artist}", filter="songs")
        if not results:
            raise Exception("Song not found on YouTube Music")
        
        song = results[0]
        video_id = song["videoId"]
        duration = song["duration"]
        # Use YouTube Music URL format
        yt_music_url = f"https://music.youtube.com/watch?v={video_id}"
        
        # 2. Prepare download directory
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, "%(title)s.%(ext)s")
        
        try:
            # 3. Run yt-dlp to download audio from YouTube Music URL
            # Note: We use the YouTube Music URL but yt-dlp internally redirects to YouTube
            # This ensures we're getting the Music version of the track
            result = subprocess.run(
                [
                    "yt-dlp",
                    "-x", "--audio-format", "mp3",
                    "--output", output_path,
                    # Use YouTube Music URL to ensure we get the Music version
                    yt_music_url
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"yt-dlp error: {result.stderr}")
            
            # Find the downloaded file
            downloaded_file = None
            for file in os.listdir(tmpdir):
                if file.endswith(('.mp3', '.webm', '.m4a')):
                    downloaded_file = os.path.join(tmpdir, file)
                    break
            
            if not downloaded_file:
                raise Exception("Downloaded file not found")
            
            # 4. Return info with file path
            return {
                "title": title,
                "artist": artist,
                "youtube_music_url": yt_music_url,
                "video_id": video_id,
                "duration": duration,
                "audio_path": downloaded_file,
                "temp_dir": tmpdir,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            # Cleanup on error
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
            raise e
            
    except Exception as e:
        return {
            "title": title,
            "artist": artist,
            "youtube_music_url": None,
            "video_id": None,
            "duration": None,
            "audio_path": None,
            "temp_dir": None,
            "success": False,
            "error": str(e)
        }

def extract_features_from_audio_file(audio_path: str) -> Dict:
    """Extract audio features from local audio file using librosa"""
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=22050, duration=30)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        # Extract comprehensive audio features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        rms_energy = np.sqrt(np.mean(y**2))
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = np.mean(spectral_centroids)
        
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zero_crossing_rate = np.mean(zcr)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()
        
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

def get_audio_features_simple(title: str, artist: str) -> Dict:
    """
    Simple function: input title and artist, get audio features
    Uses YouTube Music ONLY - no regular YouTube downloads
    """
    download_result = None
    
    try:
        # Step 1: Download from YouTube Music using ytmusicapi search + yt-dlp
        download_result = download_song_from_youtube_music(title, artist)
        
        if not download_result["success"]:
            return {
                "title": title,
                "artist": artist,
                "youtube_music_url": None,
                "video_id": None,
                "duration": None,
                "audio_features": None,
                "error": download_result["error"]
            }
        
        # Step 2: Extract audio features
        audio_features = extract_features_from_audio_file(download_result["audio_path"])
        
        # Step 3: Return complete result
        return {
            "title": download_result["title"],
            "artist": download_result["artist"],
            "youtube_music_url": download_result["youtube_music_url"],
            "video_id": download_result["video_id"],
            "duration": download_result["duration"],
            "audio_features": audio_features,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error in get_audio_features_simple: {str(e)}")
        return {
            "title": title,
            "artist": artist,
            "youtube_music_url": download_result["youtube_music_url"] if download_result else None,
            "video_id": download_result["video_id"] if download_result else None,
            "duration": download_result["duration"] if download_result else None,
            "audio_features": None,
            "error": str(e)
        }
    
    finally:
        # Cleanup temporary directory
        if download_result and download_result.get("temp_dir") and os.path.exists(download_result["temp_dir"]):
            try:
                shutil.rmtree(download_result["temp_dir"])
                logger.debug(f"Cleaned up temp directory: {download_result['temp_dir']}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up: {cleanup_error}")

async def process_track_batch(tracks: List[TrackRequest]) -> List[TrackResult]:
    """Process a batch of tracks asynchronously"""
    results = []
    
    for track in tracks:
        try:
            loop = asyncio.get_event_loop()
            result_dict = await loop.run_in_executor(
                executor,
                get_audio_features_simple,
                track.title,
                track.artist
            )
            
            result = TrackResult(
                title=result_dict["title"],
                artist=result_dict["artist"],
                youtube_music_url=result_dict["youtube_music_url"],
                video_id=result_dict["video_id"],
                duration=result_dict["duration"],
                audio_features=AudioFeatures(**result_dict["audio_features"]) if result_dict["audio_features"] else None,
                error=result_dict["error"]
            )
            
            results.append(result)
            
        except Exception as e:
            result = TrackResult(
                title=track.title,
                artist=track.artist,
                error=f"Processing error: {str(e)}"
            )
            results.append(result)
            logger.error(f"Error processing {track.title} by {track.artist}: {str(e)}")
    
    return results

@audio_feature_router.get("/download")
def download_song(title: str, artist: str):
    """
    Download endpoint - Downloads from YouTube Music ONLY
    """
    try:
        # 1. Search on YouTube Music
        results = ytmusic.search(query=f"{title} {artist}", filter="songs")
        if not results:
            raise HTTPException(status_code=404, detail="Song not found on YouTube Music")
        
        song = results[0]
        video_id = song["videoId"]
        duration = song["duration"]
        # Use YouTube Music URL
        yt_music_url = f"https://music.youtube.com/watch?v={video_id}"
        
        # 2. Prepare download directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "%(title)s.%(ext)s")
            
            # 3. Run yt-dlp to download audio from YouTube Music
            result = subprocess.run(
                [
                    "yt-dlp",
                    "-x", "--audio-format", "mp3",
                    "--output", output_path,
                    # Use YouTube Music URL to ensure Music version
                    yt_music_url
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"yt-dlp error: {result.stderr}")
            
            # 4. Return basic info
            return {
                "title": title,
                "artist": artist,
                "youtube_music_url": yt_music_url,
                "video_id": video_id,
                "duration": duration,
                "audio_path": output_path,
                "source": "YouTube Music",
                "audio_features": None  # You can extract using librosa later
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download from YouTube Music: {str(e)}")

@audio_feature_router.get("/simple_analyze")
def simple_analyze(title: str, artist: str):
    """
    Simple endpoint: input title and artist, get audio features
    Downloads from YouTube Music ONLY using ytmusicapi search
    """
    try:
        result = get_audio_features_simple(title, artist)
        
        if result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Add source indicator
        result["source"] = "YouTube Music"
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple_analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@audio_feature_router.post("/analyze", response_model=BatchResponse)
async def analyze_audio_features(batch: TrackBatch):
    """Analyze audio features for a batch of tracks using YouTube Music ONLY"""
    try:
        all_results = []
        total_tracks = len(batch.tracks)
        
        # Process tracks in batches of 3
        for i in range(0, total_tracks, 3):
            batch_tracks = batch.tracks[i:i+3]
            logger.info(f"Processing batch {i//3 + 1}: tracks {i+1}-{min(i+3, total_tracks)} of {total_tracks}")
            
            batch_results = await process_track_batch(batch_tracks)
            all_results.extend(batch_results)
            
            # Delay between batches to be respectful to YouTube Music
            if i + 3 < total_tracks:
                await asyncio.sleep(2)
        
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
    """Analyze audio features for a single track using YouTube Music ONLY"""
    try:
        batch_results = await process_track_batch([track])
        return batch_results[0]
    except Exception as e:
        logger.error(f"Error processing single track: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Single track processing failed: {str(e)}")

@audio_feature_router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "audio_feature_analyzer",
        "source": "YouTube Music ONLY",
        "note": "Uses ytmusicapi for search + yt-dlp for download from YouTube Music URLs",
        "endpoints": [
            "/download - Download only from YouTube Music",
            "/simple_analyze - Download from YouTube Music + extract features", 
            "/analyze - Batch processing (YouTube Music only)",
            "/analyze_single - Single track processing (YouTube Music only)"
        ]
    }