import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import re
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Lyrics-to-Story Generator API")

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORY_MODEL = "gpt2-medium"

# Global variables for models
embedding_model = None
story_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pydantic models
class SongRequest(BaseModel):
    title: str
    artist: str

class StoryParams(BaseModel):
    max_length: int = 400
    temperature: float = 0.8
    top_p: float = 0.9

class LyricsAnalysisResult(BaseModel):
    themes: List[str]
    emotions: List[str]
    story_elements: Dict[str, List[str]]
    embeddings: List[List[float]]
    story_prompt: str

# Initialize models on startup
@app.on_event("startup")
async def load_models():
    global embedding_model, story_pipeline
    
    try:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        print("Loading story generation model...")
        story_tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL)
        story_model = AutoModelForCausalLM.from_pretrained(
            STORY_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        story_tokenizer.pad_token = story_tokenizer.eos_token
        
        story_pipeline = pipeline(
            "text-generation",
            model=story_model,
            tokenizer=story_tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Function 1: Enhanced lyrics fetching with multiple sources
class LyricsFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def search_genius(self, title: str, artist: str) -> Optional[str]:
        """Search Genius.com for lyrics"""
        try:
            search_url = "https://genius.com/api/search/multi"
            params = {
                'per_page': 5,
                'q': f"{title} {artist}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Look for song matches
                        if 'response' in data and 'sections' in data['response']:
                            for section in data['response']['sections']:
                                if section['type'] == 'song':
                                    for hit in section['hits']:
                                        song_url = hit['result']['url']
                                        lyrics = await self._extract_genius_lyrics(session, song_url)
                                        if lyrics:
                                            return lyrics
        except Exception as e:
            print(f"Genius search error: {e}")
        return None
    
    async def _extract_genius_lyrics(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Extract lyrics from Genius song page"""
        try:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find lyrics container (Genius uses various selectors)
                    lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
                    if lyrics_divs:
                        lyrics_parts = []
                        for div in lyrics_divs:
                            lyrics_parts.append(div.get_text(separator='\n'))
                        return '\n'.join(lyrics_parts).strip()
        except Exception as e:
            print(f"Genius extraction error: {e}")
        return None
    
    async def search_azlyrics(self, title: str, artist: str) -> Optional[str]:
        """Search AZLyrics as fallback"""
        try:
            # Format for AZLyrics URL structure
            artist_clean = re.sub(r'[^a-zA-Z0-9]', '', artist.lower())
            title_clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
            
            url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{title_clean}.html"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # AZLyrics lyrics are in a specific div
                        lyrics_div = soup.find('div', class_='')
                        if lyrics_div and not lyrics_div.get('class'):
                            lyrics_text = lyrics_div.get_text(separator='\n').strip()
                            if len(lyrics_text) > 50:  # Basic validation
                                return lyrics_text
        except Exception as e:
            print(f"AZLyrics search error: {e}")
        return None
    
    async def fetch_lyrics(self, title: str, artist: str) -> Dict[str, Any]:
        """Try multiple sources to fetch lyrics"""
        lyrics = None
        source = "none"
        
        # Try Genius first
        lyrics = await self.search_genius(title, artist)
        if lyrics:
            source = "genius"
        else:
            # Try AZLyrics as fallback
            lyrics = await self.search_azlyrics(title, artist)
            if lyrics:
                source = "azlyrics"
        
        return {
            "title": title,
            "artist": artist,
            "lyrics": lyrics,
            "source": source,
            "success": lyrics is not None
        }

# Function 2 & 3: Combined Embedding and Analysis
class LyricsAnalyzer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.theme_patterns = {
            "love_romance": ["love", "heart", "kiss", "romance", "together", "forever", "baby", "darling", "beloved"],
            "loss_heartbreak": ["lost", "gone", "cry", "tears", "miss", "lonely", "hurt", "pain", "goodbye", "broken"],
            "freedom_rebellion": ["free", "fly", "run", "escape", "break", "rebel", "wild", "chains", "prison"],
            "hope_dreams": ["hope", "dream", "wish", "believe", "future", "tomorrow", "light", "shine", "faith"],
            "struggle_perseverance": ["fight", "battle", "struggle", "war", "overcome", "strength", "survive", "endure"],
            "celebration_joy": ["dance", "party", "celebrate", "joy", "happy", "music", "sing", "laugh", "smile"],
            "nostalgia_memory": ["remember", "past", "yesterday", "childhood", "used to", "old days", "memory", "time"],
            "spirituality_transcendence": ["soul", "spirit", "heaven", "divine", "prayer", "angel", "sacred", "eternal"],
            "nature_journey": ["mountain", "ocean", "river", "sky", "road", "path", "journey", "travel", "adventure"],
            "urban_nightlife": ["city", "street", "neon", "night", "club", "bar", "downtown", "lights", "crowd"]
        }
        
        self.emotion_patterns = {
            "passionate": ["fire", "burn", "intense", "wild", "crazy", "mad", "fever", "flame"],
            "melancholic": ["blue", "rain", "grey", "shadow", "dark", "cold", "empty", "hollow"],
            "euphoric": ["high", "fly", "sky", "up", "rise", "soar", "electric", "alive"],
            "rebellious": ["against", "system", "rules", "authority", "conform", "different"],
            "romantic": ["tender", "gentle", "soft", "sweet", "warm", "embrace", "whisper"],
            "empowering": ["strong", "power", "rise", "stand", "voice", "courage", "brave"]
        }
    
    def embed_and_analyze(self, lyrics: str) -> LyricsAnalysisResult:
        """Combined embedding generation and analysis"""
        # Split into meaningful segments
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Generate embeddings
        line_embeddings = self.embedding_model.encode(lines, convert_to_numpy=True)
        overall_embedding = self.embedding_model.encode([lyrics], convert_to_numpy=True)[0]
        
        # Cluster analysis for thematic grouping
        themes = self._extract_themes(lyrics)
        emotions = self._extract_emotions(lyrics)
        story_elements = self._extract_story_elements(lyrics, lines, line_embeddings)
        
        # Generate sophisticated story prompt
        story_prompt = self._generate_story_prompt(themes, emotions, story_elements)
        
        return LyricsAnalysisResult(
            themes=themes,
            emotions=emotions,
            story_elements=story_elements,
            embeddings=line_embeddings.tolist(),
            story_prompt=story_prompt
        )
    
    def _extract_themes(self, lyrics: str) -> List[str]:
        """Extract themes using keyword matching and semantic analysis"""
        lyrics_lower = lyrics.lower()
        detected_themes = []
        
        for theme, keywords in self.theme_patterns.items():
            score = sum(1 for keyword in keywords if keyword in lyrics_lower)
            if score >= 2:  # Require multiple keyword matches
                detected_themes.append(theme.replace("_", " "))
        
        return detected_themes or ["personal journey"]
    
    def _extract_emotions(self, lyrics: str) -> List[str]:
        """Extract emotional tone"""
        lyrics_lower = lyrics.lower()
        detected_emotions = []
        
        for emotion, keywords in self.emotion_patterns.items():
            if any(keyword in lyrics_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions or ["reflective"]
    
    def _extract_story_elements(self, lyrics: str, lines: List[str], embeddings: np.ndarray) -> Dict[str, List[str]]:
        """Extract story elements using clustering and pattern matching"""
        # Character detection
        characters = []
        pronouns = ["i", "you", "he", "she", "we", "they"]
        roles = ["girl", "boy", "man", "woman", "child", "friend", "lover", "stranger"]
        
        lyrics_lower = lyrics.lower()
        for char in pronouns + roles:
            if char in lyrics_lower:
                characters.append(char)
        
        # Setting detection
        settings = []
        locations = ["city", "town", "home", "street", "car", "beach", "mountain", "room", "stage", "bar"]
        for location in locations:
            if location in lyrics_lower:
                settings.append(location)
        
        # Temporal elements
        time_elements = []
        time_words = ["night", "day", "morning", "evening", "summer", "winter", "tonight", "yesterday"]
        for time_word in time_words:
            if time_word in lyrics_lower:
                time_elements.append(time_word)
        
        # Conflict/resolution patterns
        conflicts = []
        if any(word in lyrics_lower for word in ["fight", "struggle", "problem", "trouble", "conflict"]):
            conflicts.append("internal struggle")
        if any(word in lyrics_lower for word in ["against", "versus", "enemy", "opposition"]):
            conflicts.append("external conflict")
        
        return {
            "characters": list(set(characters)),
            "settings": list(set(settings)),
            "time_elements": list(set(time_elements)),
            "conflicts": conflicts
        }
    
    def _generate_story_prompt(self, themes: List[str], emotions: List[str], story_elements: Dict[str, List[str]]) -> str:
        """Generate sophisticated story prompt"""
        # Select primary theme and emotion
        primary_theme = themes[0] if themes else "personal journey"
        primary_emotion = emotions[0] if emotions else "reflective"
        
        # Build story elements
        character_desc = "a person"
        if "lover" in story_elements.get("characters", []):
            character_desc = "two lovers"
        elif "friend" in story_elements.get("characters", []):
            character_desc = "close friends"
        
        setting_desc = ""
        if story_elements.get("settings"):
            setting_desc = f" in {story_elements['settings'][0]}"
        
        time_desc = ""
        if story_elements.get("time_elements"):
            time_desc = f" during {story_elements['time_elements'][0]}"
        
        # Construct prompt based on theme
        if "love" in primary_theme:
            base_story = f"{character_desc} discover unexpected love"
        elif "loss" in primary_theme:
            base_story = f"{character_desc} learn to heal from profound loss"
        elif "freedom" in primary_theme:
            base_story = f"{character_desc} break free from constraints"
        elif "hope" in primary_theme:
            base_story = f"{character_desc} find hope in darkness"
        else:
            base_story = f"{character_desc} face a life-changing moment"
        
        # Add emotional modifier
        if primary_emotion == "passionate":
            emotion_modifier = "with intense emotions and dramatic revelations"
        elif primary_emotion == "melancholic":
            emotion_modifier = "with bittersweet reflection and quiet wisdom"
        elif primary_emotion == "euphoric":
            emotion_modifier = "filled with joy and triumphant moments"
        else:
            emotion_modifier = "with deep personal transformation"
        
        return f"{base_story}{setting_desc}{time_desc}, {emotion_modifier}"

# Function 4: Enhanced Story Generation
class StoryGenerator:
    def __init__(self, story_pipeline):
        self.story_pipeline = story_pipeline
    
    def generate_story(self, analysis: LyricsAnalysisResult, params: StoryParams) -> Dict[str, Any]:
        """Generate story based on lyrics analysis"""
        # Create rich narrative prompt
        formatted_prompt = f"Once upon a time, {analysis.story_prompt}. "
        
        # Generate story with optimized parameters
        with torch.no_grad():
            result = self.story_pipeline(
                formatted_prompt,
                max_length=params.max_length,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.story_pipeline.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        generated_text = result[0]["generated_text"]
        
        # Post-process story
        story_text = self._clean_story(generated_text)
        
        return {
            "story": story_text,
            "inspiration": {
                "themes": analysis.themes,
                "emotions": analysis.emotions,
                "story_elements": analysis.story_elements
            },
            "generation_params": {
                "prompt": analysis.story_prompt,
                "max_length": params.max_length,
                "temperature": params.temperature
            }
        }
    
    def _clean_story(self, text: str) -> str:
        """Clean and format the generated story"""
        # Remove prompt if included
        if text.startswith("Once upon a time, "):
            story_start = text.find(". ") + 2
            if story_start < len(text):
                text = "Once upon a time, " + text[story_start:]
        
        # Ensure proper sentence endings
        sentences = text.split('. ')
        if len(sentences) > 1 and not text.rstrip().endswith('.'):
            text = '. '.join(sentences[:-1]) + '.'
        
        # Basic formatting
        text = re.sub(r'\n+', ' ', text)  # Remove excessive newlines
        text = re.sub(r' +', ' ', text)   # Remove excessive spaces
        
        return text.strip()

# Initialize components
lyrics_fetcher = LyricsFetcher()
lyrics_analyzer = None
story_generator = None

@app.on_event("startup")
async def initialize_components():
    global lyrics_analyzer, story_generator
    await load_models()
    lyrics_analyzer = LyricsAnalyzer(embedding_model)
    story_generator = StoryGenerator(story_pipeline)

# API Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Advanced Lyrics-to-Story Generator API",
        "pipeline": [
            "1. Fetch lyrics from multiple sources (Genius, AZLyrics)",
            "2. Generate embeddings and analyze themes/emotions",
            "3. Extract story elements and create narrative prompt",
            "4. Generate compelling story based on musical inspiration"
        ],
        "endpoints": {
            "/fetch-lyrics/": "Fetch lyrics from web sources",
            "/analyze-lyrics/": "Combined embedding and analysis",
            "/generate-story-from-lyrics/": "Generate story from lyrics text",
            "/song-to-story/": "Complete pipeline: song info to story"
        }
    }

@app.post("/fetch-lyrics/")
async def fetch_lyrics_endpoint(song_request: SongRequest):
    """Fetch lyrics using multiple web sources"""
    result = await lyrics_fetcher.fetch_lyrics(song_request.title, song_request.artist)
    
    if not result["success"]:
        raise HTTPException(
            status_code=404, 
            detail=f"Could not find lyrics for '{song_request.title}' by {song_request.artist}"
        )
    
    return result

@app.post("/analyze-lyrics/")
def analyze_lyrics_endpoint(lyrics: str):
    """Combined embedding generation and analysis"""
    if not lyrics_analyzer:
        raise HTTPException(status_code=500, detail="Analyzer not initialized")
    
    try:
        analysis = lyrics_analyzer.embed_and_analyze(lyrics)
        return {
            "analysis": analysis.dict(),
            "embedding_info": {
                "dimension": len(analysis.embeddings[0]) if analysis.embeddings else 0,
                "num_segments": len(analysis.embeddings)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-story-from-lyrics/")
def generate_story_from_lyrics(lyrics: str, params: StoryParams = StoryParams()):
    """Generate story from lyrics text"""
    if not lyrics_analyzer or not story_generator:
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    try:
        analysis = lyrics_analyzer.embed_and_analyze(lyrics)
        story_result = story_generator.generate_story(analysis, params)
        return story_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.post("/song-to-story/")
async def song_to_story_complete_pipeline(
    song_request: SongRequest, 
    params: StoryParams = StoryParams()
):
    """Complete pipeline: Song information to generated story"""
    try:
        # Step 1: Fetch lyrics
        lyrics_result = await lyrics_fetcher.fetch_lyrics(song_request.title, song_request.artist)
        
        if not lyrics_result["success"]:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not fetch lyrics for '{song_request.title}' by {song_request.artist}"
            )
        
        # Step 2: Analyze lyrics
        analysis = lyrics_analyzer.embed_and_analyze(lyrics_result["lyrics"])
        
        # Step 3: Generate story
        story_result = story_generator.generate_story(analysis, params)
        
        return {
            "song_info": {
                "title": song_request.title,
                "artist": song_request.artist,
                "lyrics_source": lyrics_result["source"]
            },
            "analysis": analysis.dict(),
            "story": story_result["story"],
            "inspiration": story_result["inspiration"],
            "generation_params": story_result["generation_params"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complete pipeline failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "components": {
            "embedding_model": embedding_model is not None,
            "story_pipeline": story_pipeline is not None,
            "lyrics_analyzer": lyrics_analyzer is not None,
            "story_generator": story_generator is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)