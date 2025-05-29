# audio_feature_router.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
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

# Initialize YTMusic
ytmusic = YTMusic()

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

def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'yt-dlp': False,
        'ffmpeg': False,
        'ytmusicapi': False
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
    
    try:
        ytmusic.search("test", filter="songs", limit=1)
        dependencies['ytmusicapi'] = True
    except Exception as e:
        logger.warning(f"ytmusicapi not working: {e}")
    
    logger.info(f"Dependencies check: {dependencies}")
    return dependencies

def search_youtube_music(title: str, artist: str) -> Optional[Dict]:
    """Search for track using YouTube Music API and return YouTube Music URL"""
    try:
        query = f"{title} {artist}"
        logger.info(f"Searching YouTube Music for: {query}")
        
        results = ytmusic.search(query, filter="songs", limit=5)
        
        if not results:
            logger.warning(f"No results found for: {query}")
            return None
        
        # Find the best match
        best_match = None
        for result in results:
            try:
                result_title = result.get('title', '').lower()
                result_artists = [artist['name'].lower() for artist in result.get('artists', [])]
                
                title_match = title.lower() in result_title or result_title in title.lower()
                artist_match = any(artist.lower() in ra or ra in artist.lower() for ra in result_artists)
                
                if title_match and artist_match:
                    best_match = result
                    break
                elif title_match or artist_match:
                    if not best_match:
                        best_match = result
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        if not best_match:
            best_match = results[0]
        
        video_id = best_match.get('videoId')
        if not video_id:
            logger.error(f"No video ID found in result: {best_match}")
            return None
        
        track_info = {
            'title': best_match.get('title', title),
            'artists': [artist['name'] for artist in best_match.get('artists', [])],
            'duration': best_match.get('duration', 'Unknown'),
            'video_id': video_id,
            'youtube_music_url': f"https://music.youtube.com/watch?v={video_id}",
            'youtube_url': f"https://www.youtube.com/watch?v={video_id}"  # Fallback URL
        }
        
        logger.info(f"Found match: {track_info['title']} by {', '.join(track_info['artists'])}")
        return track_info
        
    except Exception as e:
        logger.error(f"Error searching YouTube Music for {title} by {artist}: {str(e)}")
        return None

def download_from_youtube_music(youtube_music_url: str, output_dir: str) -> Optional[str]:
    """Download audio from YouTube Music URL using yt-dlp"""
    try:
        # Extract video ID from YouTube Music URL
        if "music.youtube.com" in youtube_music_url:
            video_id = youtube_music_url.split("v=")[1].split("&")[0]
            # Use both YouTube Music URL and regular YouTube URL as fallbacks
            urls_to_try = [
                youtube_music_url,
                f"https://www.youtube.com/watch?v={video_id}"
            ]
        else:
            urls_to_try = [youtube_music_url]
        
        for url in urls_to_try:
            logger.info(f"Trying to download from: {url}")
            
            # Strategy 1: Try with browser cookies
            for browser in ["chrome", "firefox", "edge"]:
                try:
                    command = [
                        "yt-dlp",
                        "--cookies-from-browser", browser,
                        "-x", "--audio-format", "mp3",
                        "--audio-quality", "0",
                        "--output", f"{output_dir}/%(title)s.%(ext)s",
                        "--prefer-ffmpeg",
                        "--no-warnings",
                        url
                    ]
                    
                    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        audio_file = find_audio_file(output_dir)
                        if audio_file:
                            logger.info(f"Successfully downloaded with {browser} cookies from {url}")
                            return audio_file
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout with {browser} cookies")
                    continue
                except Exception as e:
                    logger.warning(f"Failed with {browser} cookies: {str(e)}")
                    continue
            
            # Strategy 2: Try without cookies
            try:
                command = [
                    "yt-dlp",
                    "-x", "--audio-format", "mp3",
                    "--audio-quality", "0",
                    "--output", f"{output_dir}/%(title)s.%(ext)s",
                    "--prefer-ffmpeg",
                    "--no-warnings",
                    "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    url
                ]
                
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    audio_file = find_audio_file(output_dir)
                    if audio_file:
                        logger.info(f"Successfully downloaded without cookies from {url}")
                        return audio_file
            except Exception as e:
                logger.warning(f"Failed without cookies: {str(e)}")
                continue
            
            # Strategy 3: Download and convert manually
            try:
                command = [
                    "yt-dlp",
                    "-f", "bestaudio",
                    "--output", f"{output_dir}/%(title)s.%(ext)s",
                    "--no-warnings",
                    url
                ]
                
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    # Find downloaded file and convert with ffmpeg
                    downloaded_file = find_audio_file(output_dir)
                    if downloaded_file:
                        output_mp3 = os.path.join(output_dir, "converted.mp3")
                        ffmpeg_command = [
                            "ffmpeg", "-i", downloaded_file,
                            "-acodec", "libmp3lame", "-ab", "192k",
                            "-y", output_mp3
                        ]
                        
                        ffmpeg_result = subprocess.run(ffmpeg_command, capture_output=True, timeout=60)
                        if ffmpeg_result.returncode == 0 and os.path.exists(output_mp3):
                            os.remove(downloaded_file)  # Remove original
                            logger.info(f"Successfully converted from {url}")
                            return output_mp3
                        else:
                            logger.info(f"Using original format from {url}")
                            return downloaded_file
            except Exception as e:
                logger.warning(f"Manual conversion failed: {str(e)}")
                continue
        
        logger.error(f"All download strategies failed for {youtube_music_url}")
        return None
        
    except Exception as e:
        logger.error(f"Error downloading from YouTube Music: {str(e)}")
        return None

def find_audio_file(directory: str) -> Optional[str]:
    """Find any audio file in the directory"""
    audio_extensions = ["*.mp3", "*.webm", "*.m4a", "*.wav", "*.ogg", "*.aac"]
    
    for ext in audio_extensions:
        files = glob.glob(os.path.join(directory, ext))
        if files:
            return files[0]
    
    return None

def extract_features_from_audio_file(audio_path: str) -> Dict:
    """Extract audio features from local audio file using librosa"""
    try:
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
    Uses YouTube Music for downloading
    """
    temp_dir = None
    
    try:
        result = {
            "title": title,
            "artist": artist,
            "youtube_music_url": None,
            "video_id": None,
            "duration": None,
            "audio_features": None,
            "error": None
        }
        
        # Step 1: Search YouTube Music
        track_info = search_youtube_music(title, artist)
        if not track_info:
            result["error"] = "Track not found on YouTube Music"
            return result
        
        # Set track info
        result["youtube_music_url"] = track_info["youtube_music_url"]
        result["video_id"] = track_info["video_id"]
        result["duration"] = track_info["duration"]
        
        # Step 2: Download from YouTube Music
        temp_dir = tempfile.mkdtemp()
        audio_path = download_from_youtube_music(track_info["youtube_music_url"], temp_dir)
        
        if not audio_path:
            result["error"] = "Failed to download from YouTube Music"
            return result
        
        # Step 3: Extract audio features
        features_dict = extract_features_from_audio_file(audio_path)
        result["audio_features"] = features_dict
        
        logger.info(f"Successfully processed {title} by {artist}")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_audio_features_simple: {str(e)}")
        return {
            "title": title,
            "artist": artist,
            "youtube_music_url": None,
            "video_id": None,
            "duration": None,
            "audio_features": None,
            "error": str(e)
        }
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
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

@audio_feature_router.post("/simple_analyze")
async def simple_analyze(title: str, artist: str):
    """
    Simple endpoint: just input title and artist, get audio features
    Downloads from YouTube Music automatically
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            get_audio_features_simple,
            title,
            artist
        )
        return result
        
    except Exception as e:
        logger.error(f"Error in simple_analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@audio_feature_router.post("/analyze", response_model=BatchResponse)
async def analyze_audio_features(batch: TrackBatch):
    """Analyze audio features for a batch of tracks using YouTube Music"""
    try:
        all_results = []
        total_tracks = len(batch.tracks)
        
        # Check dependencies
        deps = check_dependencies()
        if not deps['yt-dlp']:
            raise HTTPException(status_code=500, detail="yt-dlp is not available")
        if not deps['ytmusicapi']:
            raise HTTPException(status_code=500, detail="ytmusicapi is not available")
        
        # Process tracks in batches of 3
        for i in range(0, total_tracks, 3):
            batch_tracks = batch.tracks[i:i+3]
            logger.info(f"Processing batch {i//3 + 1}: tracks {i+1}-{min(i+3, total_tracks)} of {total_tracks}")
            
            batch_results = await process_track_batch(batch_tracks)
            all_results.extend(batch_results)
            
            # Delay between batches
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
        "status": "healthy" if all([deps['yt-dlp'], deps['ytmusicapi']]) else "degraded",
        "service": "audio_feature_analyzer",
        "dependencies": deps,
        "source": "YouTube Music"
    }