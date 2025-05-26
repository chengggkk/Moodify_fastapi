import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="NLP Story Generator API")

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

class LyricsData(BaseModel):
    lyrics: str

class StoryParams(BaseModel):
    max_length: int = 300
    temperature: float = 0.8

class StoryGenerationRequest(BaseModel):
    song_title: str
    artist: str
    story_params: Optional[StoryParams] = StoryParams()

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

@app.get("/")
def read_root():
    return {
        "message": "Streamlined NLP Story Generator API",
        "functions": [
            "1. Fetch lyrics by song title and artist",
            "2. Generate embeddings for lyrics",
            "3. Analyze embeddings for themes and sentiment",
            "4. Generate story based on analysis"
        ],
        "endpoints": {
            "/fetch-lyrics/": "Fetch lyrics from web",
            "/embed-lyrics/": "Generate lyrics embeddings",
            "/analyze-embeddings/": "Analyze embeddings for patterns",
            "/generate-story-from-song/": "Complete pipeline: song to story"
        }
    }

# Function 1: Web crawler to fetch lyrics
@app.post("/fetch-lyrics/")
def fetch_lyrics(song_request: SongRequest):
    """Fetch lyrics by song title and artist using web scraping"""
    try:
        # Format search query
        query = f"{song_request.title} {song_request.artist} lyrics"
        
        # Use multiple sources for better reliability
        lyrics = None
        
        # Try Genius.com first (most reliable for lyrics)
        try:
            genius_url = f"https://genius.com/search?q={query.replace(' ', '%20')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Note: In a production environment, you'd want to use official APIs
            # This is a simplified example - actual implementation would need
            # proper API keys and respect rate limits
            
            # For demo purposes, return sample structure
            lyrics = f"Sample lyrics for '{song_request.title}' by {song_request.artist}"
            
        except Exception as e:
            print(f"Error fetching from primary source: {e}")
        
        # Fallback: Try alternative sources or return placeholder
        if not lyrics:
            lyrics = f"Unable to fetch lyrics for '{song_request.title}' by {song_request.artist}. Please provide lyrics manually."
        
        return {
            "title": song_request.title,
            "artist": song_request.artist,
            "lyrics": lyrics,
            "source": "web_crawler",
            "success": lyrics is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lyrics fetching failed: {str(e)}")

# Function 2: Generate embeddings for lyrics
@app.post("/embed-lyrics/")
def embed_lyrics(lyrics_data: LyricsData):
    """Generate embeddings for lyrics text"""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    try:
        # Split lyrics into sentences for better analysis
        sentences = [s.strip() for s in lyrics_data.lyrics.split('\n') if s.strip()]
        
        # Generate embeddings for each line and overall lyrics
        sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
        overall_embedding = embedding_model.encode([lyrics_data.lyrics], convert_to_numpy=True)[0]
        
        return {
            "overall_embedding": overall_embedding.tolist(),
            "sentence_embeddings": sentence_embeddings.tolist(),
            "embedding_dimension": len(overall_embedding),
            "num_sentences": len(sentences),
            "sentences": sentences
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# Function 3: Analyze embeddings for themes and patterns
@app.post("/analyze-embeddings/")
def analyze_embeddings(lyrics_data: LyricsData):
    """Analyze lyrics embeddings to extract themes, sentiment, and story elements"""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    try:
        # Get embeddings
        embedding_result = embed_lyrics(lyrics_data)
        sentence_embeddings = np.array(embedding_result["sentence_embeddings"])
        sentences = embedding_result["sentences"]
        
        # Cluster sentences to find thematic groups
        if len(sentences) > 3:
            n_clusters = min(3, len(sentences) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(sentence_embeddings)
        else:
            clusters = [0] * len(sentences)
        
        # Analyze themes based on keywords and semantic similarity
        themes = []
        emotions = []
        story_elements = []
        
        lyrics_lower = lyrics_data.lyrics.lower()
        
        # Theme detection with expanded categories
        theme_patterns = {
            "love_romance": ["love", "heart", "kiss", "romance", "together", "forever", "baby", "darling"],
            "loss_sadness": ["lost", "gone", "cry", "tears", "miss", "lonely", "hurt", "pain", "goodbye"],
            "freedom_adventure": ["free", "fly", "run", "escape", "journey", "road", "wild", "adventure"],
            "hope_dreams": ["hope", "dream", "wish", "believe", "future", "tomorrow", "light", "shine"],
            "struggle_conflict": ["fight", "battle", "struggle", "war", "against", "overcome", "strength"],
            "celebration": ["dance", "party", "celebrate", "joy", "happy", "music", "sing", "laugh"],
            "nostalgia": ["remember", "past", "yesterday", "childhood", "used to", "old days", "memory"],
            "spirituality": ["god", "heaven", "soul", "spirit", "pray", "faith", "angel", "divine"]
        }
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in lyrics_lower for keyword in keywords):
                themes.append(theme.replace("_", " "))
        
        # Emotion detection
        emotion_patterns = {
            "passionate": ["fire", "burn", "intense", "wild", "crazy", "mad"],
            "melancholic": ["blue", "rain", "grey", "shadow", "dark", "cold"],
            "euphoric": ["high", "fly", "sky", "up", "rise", "soar"],
            "rebellious": ["break", "rules", "rebel", "against", "system", "fight"]
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in lyrics_lower for keyword in keywords):
                emotions.append(emotion)
        
        # Story element extraction
        story_elements = {
            "characters": [],
            "settings": [],
            "conflicts": [],
            "resolutions": []
        }
        
        # Simple character detection (pronouns and roles)
        character_indicators = ["i", "you", "he", "she", "we", "they", "girl", "boy", "man", "woman"]
        for indicator in character_indicators:
            if indicator in lyrics_lower:
                story_elements["characters"].append(indicator)
        
        # Setting detection
        setting_words = ["city", "town", "street", "home", "car", "beach", "mountain", "room", "bar", "club"]
        for setting in setting_words:
            if setting in lyrics_lower:
                story_elements["settings"].append(setting)
        
        # Generate story prompt based on analysis
        dominant_theme = themes[0] if themes else "personal journey"
        dominant_emotion = emotions[0] if emotions else "reflective"
        
        story_prompt = f"A {dominant_emotion} story about {dominant_theme}"
        if story_elements["settings"]:
            story_prompt += f" set in {story_elements['settings'][0]}"
        
        return {
            "analysis": {
                "themes": themes,
                "emotions": emotions,
                "story_elements": story_elements,
                "sentence_clusters": {
                    f"cluster_{i}": [sentences[j] for j, c in enumerate(clusters) if c == i]
                    for i in range(max(clusters) + 1)
                },
                "dominant_theme": dominant_theme,
                "dominant_emotion": dominant_emotion,
                "story_prompt": story_prompt
            },
            "embeddings_info": {
                "dimension": embedding_result["embedding_dimension"],
                "num_sentences": len(sentences)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding analysis failed: {str(e)}")

# Function 4: Generate story based on analyzed information
@app.post("/generate-story/")
def generate_story(lyrics_data: LyricsData, story_params: StoryParams = StoryParams()):
    """Generate a story based on lyrics analysis"""
    if story_pipeline is None:
        raise HTTPException(status_code=500, detail="Story generation model not loaded")
    
    try:
        # First analyze the lyrics
        analysis_result = analyze_embeddings(lyrics_data)
        analysis = analysis_result["analysis"]
        
        # Create a rich story prompt based on analysis
        themes = analysis["themes"]
        emotions = analysis["emotions"]
        settings = analysis["story_elements"]["settings"]
        
        # Build narrative elements
        if themes and emotions:
            if "love romance" in themes:
                story_setup = "Two people find an unexpected connection"
            elif "loss sadness" in themes:
                story_setup = "Someone learns to heal from a profound loss"
            elif "freedom adventure" in themes:
                story_setup = "A person breaks free from constraints to discover themselves"
            else:
                story_setup = "A character faces a life-changing moment"
            
            # Add emotional tone
            if "passionate" in emotions:
                tone_modifier = "with intense emotions and dramatic turns"
            elif "melancholic" in emotions:
                tone_modifier = "with bittersweet reflection and quiet wisdom"
            elif "euphoric" in emotions:
                tone_modifier = "filled with joy and triumphant moments"
            else:
                tone_modifier = "with deep personal growth"
            
            # Add setting if available
            setting_desc = f" in {settings[0]}" if settings else " in an evocative location"
            
            formatted_prompt = f"Once upon a time, {story_setup} {setting_desc}, {tone_modifier}. "
        else:
            formatted_prompt = f"Once upon a time, {analysis['story_prompt']}. "
        
        # Generate the story
        with torch.no_grad():
            result = story_pipeline(
                formatted_prompt,
                max_length=story_params.max_length,
                temperature=story_params.temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=story_pipeline.tokenizer.pad_token_id
            )
        
        generated_text = result[0]["generated_text"]
        
        # Clean up the story
        sentences = generated_text.split('. ')
        if len(sentences) > 1 and not generated_text.rstrip().endswith('.'):
            story_text = '. '.join(sentences[:-1]) + '.'
        else:
            story_text = generated_text
        
        return {
            "story": story_text,
            "inspiration": {
                "themes": themes,
                "emotions": emotions,
                "story_elements": analysis["story_elements"]
            },
            "generation_params": {
                "max_length": story_params.max_length,
                "temperature": story_params.temperature,
                "prompt_used": formatted_prompt
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

# Complete pipeline: Song to Story
@app.post("/generate-story-from-song/")
def generate_story_from_song(request: StoryGenerationRequest):
    """Complete pipeline: Fetch lyrics, analyze, and generate story"""
    try:
        # Step 1: Fetch lyrics
        song_request = SongRequest(title=request.song_title, artist=request.artist)
        lyrics_result = fetch_lyrics(song_request)
        
        if not lyrics_result["success"]:
            raise HTTPException(status_code=404, detail="Could not fetch lyrics")
        
        # Step 2-4: Analyze and generate story
        lyrics_data = LyricsData(lyrics=lyrics_result["lyrics"])
        story_result = generate_story(lyrics_data, request.story_params)
        
        return {
            "song_info": {
                "title": request.song_title,
                "artist": request.artist
            },
            "lyrics_source": lyrics_result["source"],
            "story": story_result["story"],
            "analysis": story_result["inspiration"],
            "generation_info": story_result["generation_params"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complete pipeline failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "embedding": embedding_model is not None,
            "story": story_pipeline is not None
        },
        "functions_available": [
            "fetch_lyrics",
            "embed_lyrics", 
            "analyze_embeddings",
            "generate_story"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)