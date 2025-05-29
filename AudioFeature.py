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
import requests
import base64


logger = logging.getLogger(__name__)

audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

# Initialize YTMusic for metadata only
ytmusic = YTMusic()

# Pydantic models
class TrackRequest(BaseModel):
    title: str
    artist: str
    audio_url: Optional[str] = None  # Direct audio URL if available
    audio_base64: Optional[str] = None  # Base64 encoded audio data

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
    metadata: Optional[Dict] = None  # Track metadata from YouTube Music
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
        'librosa': False,
        'ytmusicapi': False,
        'requests': False
    }
    
    try:
        import librosa
        dependencies['librosa'] = True
    except ImportError:
        logger.warning("librosa not found")
    
    try:
        ytmusic.search("test", filter="songs", limit=1)
        dependencies['ytmusicapi'] = True
    except Exception as e:
        logger.warning(f"ytmusicapi not working: {e}")
    
    try:
        import requests
        dependencies['requests'] = True
    except ImportError:
        logger.warning("requests not found")
    
    logger.info(f"Dependencies check: {dependencies}")
    return dependencies

def get_track_metadata(title: str, artist: str) -> Optional[Dict]:
    """Get track metadata using YouTube Music API (metadata only, no download)"""
    try:
        query = f"{title} {artist}"
        logger.info(f"Getting metadata for: {query}")
        
        results = ytmusic.search(query, filter="songs", limit=5)
        
        if not results:
            logger.warning(f"No metadata found for: {query}")
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
        
        metadata = {
            'title': best_match.get('title', title),
            'artists': [artist['name'] for artist in best_match.get('artists', [])],
            'album': best_match.get('album', {}).get('name', 'Unknown') if best_match.get('album') else 'Unknown',
            'duration': best_match.get('duration', 'Unknown'),
            'year': best_match.get('year', 'Unknown'),
            'thumbnails': best_match.get('thumbnails', []),
            'video_id': best_match.get('videoId', '')
        }
        
        logger.info(f"Found metadata: {metadata['title']} by {', '.join(metadata['artists'])}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error getting metadata for {title} by {artist}: {str(e)}")
        return None

def download_audio_from_url(audio_url: str, output_path: str) -> bool:
    """Download audio from a direct URL"""
    try:
        logger.info(f"Downloading audio from URL: {audio_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(audio_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from URL {audio_url}: {str(e)}")
        return False

def save_base64_audio(audio_base64: str, output_path: str) -> bool:
    """Save base64 encoded audio to file"""
    try:
        logger.info("Saving base64 audio data")
        
        # Remove data URL prefix if present
        if audio_base64.startswith('data:'):
            audio_base64 = audio_base64.split(',')[1]
        
        audio_data = base64.b64decode(audio_base64)
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Successfully saved base64 audio to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving base64 audio: {str(e)}")
        return False

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

def process_audio_from_sources(track: TrackRequest) -> Dict:
    """Process audio from various sources (URL, base64, or metadata only)"""
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory: {temp_dir}")
        
        audio_path = None
        
        # Try different audio sources
        if track.audio_url:
            # Download from direct URL
            audio_path = os.path.join(temp_dir, "audio_from_url")
            success = download_audio_from_url(track.audio_url, audio_path)
            if not success:
                audio_path = None
        
        elif track.audio_base64:
            # Save from base64 data
            audio_path = os.path.join(temp_dir, "audio_from_base64")
            success = save_base64_audio(track.audio_base64, audio_path)
            if not success:
                audio_path = None
        
        if not audio_path:
            raise ValueError("No valid audio source provided. Please provide either audio_url or audio_base64.")
        
        logger.info(f"Processing audio file: {audio_path}")
        
        # Extract features from the audio file
        features = extract_features_from_audio_file(audio_path)
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise
    
    finally:
        # Cleanup temporary files and directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp directory {temp_dir}: {cleanup_error}")

def process_track_batch_sync(tracks: List[TrackRequest]) -> List[TrackResult]:
    """Process a batch of tracks synchronously"""
    results = []
    
    for track in tracks:
        result = TrackResult(
            title=track.title,
            artist=track.artist,
            metadata=None,
            audio_features=None,
            error=None
        )
        
        try:
            # Step 1: Get metadata (always available)
            metadata = get_track_metadata(track.title, track.artist)
            result.metadata = metadata
            
            # Step 2: Process audio if available
            if track.audio_url or track.audio_base64:
                features_dict = process_audio_from_sources(track)
                result.audio_features = AudioFeatures(**features_dict)
                logger.info(f"Successfully processed {track.title} by {track.artist}")
            else:
                logger.info(f"No audio source provided for {track.title} by {track.artist}, metadata only")
            
        except Exception as e:
            result.error = f"Processing error: {str(e)}"
            logger.error(f"Error processing {track.title} by {track.artist}: {str(e)}")
        
        results.append(result)
    
    return results

async def process_track_batch(tracks: List[TrackRequest]) -> List[TrackResult]:
    """Process a batch of tracks asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_track_batch_sync, tracks)

@audio_feature_router.post("/analyze", response_model=BatchResponse)
async def analyze_audio_features(batch: TrackBatch):
    """
    Analyze audio features for a batch of tracks.
    Supports direct audio URLs or base64 encoded audio data.
    """
    try:
        all_results = []
        total_tracks = len(batch.tracks)
        
        # Check dependencies
        deps = check_dependencies()
        if not deps['librosa']:
            raise HTTPException(status_code=500, detail="librosa is not available")
        
        # Process tracks in batches of 3
        for i in range(0, total_tracks, 3):
            batch_tracks = batch.tracks[i:i+3]
            logger.info(f"Processing batch {i//3 + 1}: tracks {i+1}-{min(i+3, total_tracks)} of {total_tracks}")
            
            batch_results = await process_track_batch(batch_tracks)
            all_results.extend(batch_results)
            
            # Small delay between batches
            if i + 3 < total_tracks:
                await asyncio.sleep(1)
        
        # Calculate statistics
        successful = len([r for r in all_results if r.audio_features is not None])
        failed = len([r for r in all_results if r.error is not None])
        metadata_only = len([r for r in all_results if r.metadata is not None and r.audio_features is None and r.error is None])
        
        logger.info(f"Batch processing complete: {successful} with features, {metadata_only} metadata only, {failed} failed out of {total_tracks}")
        
        return BatchResponse(
            results=all_results,
            total_processed=total_tracks,
            successful=successful + metadata_only,  # Count metadata-only as successful
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

@audio_feature_router.post("/analyze_upload")
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    title: str = "Unknown",
    artist: str = "Unknown"
):
    """Analyze audio features from uploaded file"""
    temp_dir = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded file
        file_path = os.path.join(temp_dir, f"uploaded_{file.filename}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved to: {file_path}")
        
        # Extract features
        features_dict = extract_features_from_audio_file(file_path)
        
        # Get metadata
        metadata = get_track_metadata(title, artist)
        
        return TrackResult(
            title=title,
            artist=artist,
            metadata=metadata,
            audio_features=AudioFeatures(**features_dict),
            error=None
        )
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up: {cleanup_error}")

@audio_feature_router.get("/metadata/{artist}/{title}")
async def get_track_metadata_endpoint(artist: str, title: str):
    """Get track metadata only (no audio processing)"""
    try:
        metadata = get_track_metadata(title, artist)
        if metadata:
            return {
                "found": True,
                "metadata": metadata
            }
        else:
            return {
                "found": False,
                "message": "No metadata found"
            }
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metadata retrieval failed: {str(e)}")

@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    deps = check_dependencies()
    return {
        "status": "healthy" if deps['librosa'] else "degraded",
        "service": "audio_feature_analyzer",
        "dependencies": deps,
        "supported_sources": [
            "Direct audio URLs",
            "Base64 encoded audio",
            "File uploads",
            "Metadata-only queries"
        ]
    }