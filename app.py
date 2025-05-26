import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="NLP Story Generator API")

# Model configurations
# Using a smaller, faster embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Much faster, 384 dimensions
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # Fast sentiment analysis
STORY_MODEL = "gpt2-medium"  # Fallback to GPT-2 medium for local generation

# Global variables for models
embedding_model = None
sentiment_pipeline = None
story_pipeline = None
story_tokenizer = None
story_model = None

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Pydantic models
class TextInput(BaseModel):
    text: str
    
class TextEmbeddingInput(BaseModel):
    texts: List[str]
    
class StoryPrompt(BaseModel):
    prompt: str
    max_length: int = 300
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True
    
class LyricsAnalysisInput(BaseModel):
    lyrics: str
    story_themes: Optional[List[str]] = None

# Initialize models on startup
@app.on_event("startup")
async def load_models():
    global embedding_model, sentiment_pipeline, story_pipeline, story_tokenizer, story_model
    
    try:
        # Load embedding model (small and fast)
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load sentiment analysis pipeline (fast)
        print("Loading sentiment analysis model...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=0 if device == "cuda" else -1
        )
        
        # Load story generation model
        print("Loading story generation model...")
        # Try to load a better model first, fallback to GPT-2 if needed
        try:
            # Try loading a smaller story-focused model
            story_model_name = "roneneldan/TinyStories-33M"  # Very small model for stories
            story_tokenizer = AutoTokenizer.from_pretrained(story_model_name)
            story_model = AutoModelForCausalLM.from_pretrained(
                story_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            # Set pad token
            if story_tokenizer.pad_token is None:
                story_tokenizer.pad_token = story_tokenizer.eos_token
                
            print(f"Loaded story model: {story_model_name}")
        except Exception as e:
            print(f"Failed to load TinyStories, falling back to GPT-2: {e}")
            # Fallback to GPT-2
            story_model_name = STORY_MODEL
            story_tokenizer = AutoTokenizer.from_pretrained(story_model_name)
            story_model = AutoModelForCausalLM.from_pretrained(
                story_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            # Set pad token for GPT-2
            story_tokenizer.pad_token = story_tokenizer.eos_token
            
        # Create pipeline for easier use
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

# API endpoints
@app.get("/")
def read_root():
    return {
        "message": "NLP Story Generator API",
        "models": {
            "embedding": EMBEDDING_MODEL,
            "sentiment": SENTIMENT_MODEL,
            "story": "TinyStories-33M or GPT-2-medium"
        },
        "endpoints": {
            "/embed-text/": "Generate text embeddings (fast)",
            "/analyze-text/": "Analyze text sentiment and themes (fast)",
            "/analyze-lyrics/": "Analyze lyrics and generate story themes",
            "/generate-story/": "Generate a fictional story locally"
        }
    }

@app.post("/embed-text/")
def embed_text(input_data: TextEmbeddingInput):
    """Generate embeddings for text using a fast model"""
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    try:
        embeddings = embedding_model.encode(input_data.texts, convert_to_numpy=True)
        
        return {
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1],
            "num_texts": len(input_data.texts),
            "model": EMBEDDING_MODEL
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/analyze-text/")
def analyze_text(input_data: TextInput):
    """Fast text analysis for sentiment and basic themes"""
    if sentiment_pipeline is None:
        raise HTTPException(status_code=500, detail="Sentiment model not loaded")
    
    try:
        # Get sentiment analysis
        sentiment_result = sentiment_pipeline(input_data.text[:512])[0]  # Limit length for speed
        
        # Simple theme extraction based on keywords
        text_lower = input_data.text.lower()
        themes = []
        
        # Check for common themes
        theme_keywords = {
            "love": ["love", "heart", "romance", "kiss", "passion"],
            "adventure": ["journey", "quest", "explore", "discover", "adventure"],
            "mystery": ["secret", "hidden", "mystery", "unknown", "puzzle"],
            "friendship": ["friend", "companion", "together", "trust", "bond"],
            "loss": ["lost", "gone", "miss", "goodbye", "farewell"],
            "hope": ["hope", "dream", "wish", "future", "believe"],
            "fear": ["afraid", "fear", "scared", "terror", "nightmare"],
            "joy": ["happy", "joy", "celebrate", "laugh", "smile"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)
        
        # Generate story-relevant analysis
        analysis = {
            "sentiment": {
                "label": sentiment_result["label"],
                "score": sentiment_result["score"]
            },
            "detected_themes": themes,
            "story_mood": "uplifting" if sentiment_result["label"] == "POSITIVE" else "dramatic",
            "suggested_genre": "romance" if "love" in themes else "adventure" if "adventure" in themes else "drama"
        }
        
        return {"analysis": analysis}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze-lyrics/")
def analyze_lyrics(input_data: LyricsAnalysisInput):
    """Analyze lyrics and suggest story themes"""
    try:
        # Analyze the lyrics
        text_analysis = analyze_text(TextInput(text=input_data.lyrics))
        
        # Generate story suggestions based on themes
        story_suggestions = []
        detected_themes = text_analysis["analysis"]["detected_themes"]
        
        if "love" in detected_themes:
            story_suggestions.append("A tale of unexpected romance")
        if "adventure" in detected_themes:
            story_suggestions.append("An epic journey through unknown lands")
        if "loss" in detected_themes:
            story_suggestions.append("A story of redemption and finding hope")
        if "friendship" in detected_themes:
            story_suggestions.append("A heartwarming tale of unlikely friendships")
        
        # Add default suggestions if none found
        if not story_suggestions:
            if text_analysis["analysis"]["sentiment"]["label"] == "POSITIVE":
                story_suggestions.append("An uplifting story of personal triumph")
            else:
                story_suggestions.append("A dramatic tale of overcoming challenges")
        
        # Add user themes
        if input_data.story_themes:
            story_suggestions.extend(input_data.story_themes)
        
        # Create a story prompt
        mood = text_analysis["analysis"]["story_mood"]
        genre = text_analysis["analysis"]["suggested_genre"]
        
        recommended_prompt = f"Write a {mood} {genre} story about {story_suggestions[0].lower()}"
        
        return {
            "analysis": text_analysis["analysis"],
            "story_suggestions": story_suggestions,
            "recommended_prompt": recommended_prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lyrics analysis failed: {str(e)}")

@app.post("/generate-story/")
def generate_story(story_prompt: StoryPrompt):
    """Generate a fictional story using local model"""
    if story_pipeline is None:
        raise HTTPException(status_code=500, detail="Story generation model not loaded")
    
    try:
        # Format prompt for better story generation
        formatted_prompt = f"Once upon a time, {story_prompt.prompt.lower().rstrip('.')}. "
        
        # Generate story
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    result = story_pipeline(
                        formatted_prompt,
                        max_length=story_prompt.max_length,
                        temperature=story_prompt.temperature,
                        top_p=story_prompt.top_p,
                        do_sample=story_prompt.do_sample,
                        num_return_sequences=1,
                        pad_token_id=story_tokenizer.pad_token_id,
                        eos_token_id=story_tokenizer.eos_token_id
                    )
            else:
                result = story_pipeline(
                    formatted_prompt,
                    max_length=story_prompt.max_length,
                    temperature=story_prompt.temperature,
                    top_p=story_prompt.top_p,
                    do_sample=story_prompt.do_sample,
                    num_return_sequences=1,
                    pad_token_id=story_tokenizer.pad_token_id,
                    eos_token_id=story_tokenizer.eos_token_id
                )
        
        # Extract generated text
        generated_text = result[0]["generated_text"]
        
        # Clean up the text (remove the prompt if it's included)
        if generated_text.startswith(formatted_prompt):
            story_text = generated_text
        else:
            story_text = formatted_prompt + generated_text
            
        # Post-process to ensure it ends properly
        sentences = story_text.split('. ')
        if len(sentences) > 1 and not story_text.rstrip().endswith('.'):
            story_text = '. '.join(sentences[:-1]) + '.'
        
        return {
            "story": story_text,
            "prompt_used": story_prompt.prompt,
            "model": "TinyStories-33M" if "TinyStories" in str(story_model.__class__) else "GPT-2-medium",
            "parameters": {
                "max_length": story_prompt.max_length,
                "temperature": story_prompt.temperature,
                "top_p": story_prompt.top_p
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.post("/lyrics-to-story/")
def lyrics_to_story(input_data: LyricsAnalysisInput, max_length: int = 300):
    """Complete pipeline: analyze lyrics and generate a story"""
    try:
        # Step 1: Analyze lyrics
        lyrics_analysis = analyze_lyrics(input_data)
        
        # Step 2: Generate story based on analysis
        story_prompt = StoryPrompt(
            prompt=lyrics_analysis["recommended_prompt"],
            max_length=max_length
        )
        
        story_result = generate_story(story_prompt)
        
        return {
            "lyrics_analysis": lyrics_analysis,
            "generated_story": story_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lyrics to story pipeline failed: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "embedding": embedding_model is not None,
            "sentiment": sentiment_pipeline is not None,
            "story": story_pipeline is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)