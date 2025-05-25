import os
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load environment variables
HF_API_TOKEN = os.getenv("HF_API_KEY")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_KEY not found in environment variables.")

# Initialize FastAPI app
app = FastAPI(title="NLP Story Generator API")

# Model configurations
GTE_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
STORY_GENERATION_MODEL = "mosaicml/mpt-7b-storywriter"
HF_API_URL_TEMPLATE = "https://api-inference.huggingface.co/models/{}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Initialize models
print("Loading GTE-Qwen2-7B-instruct model...")
try:
    # Option 1: Using sentence-transformers (easier)
    gte_model = SentenceTransformer(GTE_MODEL_NAME, trust_remote_code=True)
    gte_model.max_seq_length = 8192
    USE_SENTENCE_TRANSFORMERS = True
except Exception as e:
    print(f"Failed to load with sentence-transformers: {e}")
    # Option 2: Using transformers directly
    try:
        gte_tokenizer = AutoTokenizer.from_pretrained(GTE_MODEL_NAME, trust_remote_code=True)
        gte_model = AutoModel.from_pretrained(GTE_MODEL_NAME, trust_remote_code=True)
        if torch.cuda.is_available():
            gte_model = gte_model.cuda()
        USE_SENTENCE_TRANSFORMERS = False
    except Exception as e:
        print(f"Failed to load GTE model: {e}")
        gte_model = None
        gte_tokenizer = None

# Pydantic models
class TextInput(BaseModel):
    text: str
    task_description: Optional[str] = "Given text, extract key themes and emotions for story generation"

class TextEmbeddingInput(BaseModel):
    texts: List[str]
    is_query: bool = True
    task_description: Optional[str] = "Given a web search query, retrieve relevant passages that answer the query"

class StoryPrompt(BaseModel):
    prompt: str
    max_length: int = 500
    temperature: float = 0.8
    top_p: float = 0.9

class LyricsAnalysisInput(BaseModel):
    lyrics: str
    story_themes: Optional[List[str]] = None

# Helper functions
def last_token_pool(last_hidden_states, attention_mask):
    """Pool the last token from the sequence"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction for the GTE model"""
    return f'Instruct: {task_description}\nQuery: {query}'

# API endpoints
@app.get("/")
def read_root():
    return {
        "message": "NLP Story Generator API",
        "endpoints": {
            "/embed-text/": "Generate text embeddings using GTE-Qwen2-7B-instruct",
            "/analyze-text/": "Analyze text for themes and emotions",
            "/analyze-lyrics/": "Analyze lyrics and generate story themes",
            "/generate-story/": "Generate a fictional story using MPT-7B-StoryWriter"
        }
    }

@app.post("/embed-text/")
def embed_text(input_data: TextEmbeddingInput):
    """Generate embeddings for text using GTE-Qwen2-7B-instruct"""
    if gte_model is None:
        raise HTTPException(status_code=500, detail="GTE model not loaded")
    
    try:
        if USE_SENTENCE_TRANSFORMERS:
            # Using sentence-transformers
            if input_data.is_query:
                embeddings = gte_model.encode(
                    [get_detailed_instruct(input_data.task_description, text) for text in input_data.texts],
                    prompt_name="query",
                    convert_to_numpy=True
                )
            else:
                embeddings = gte_model.encode(input_data.texts, convert_to_numpy=True)
        else:
            # Using transformers directly
            if input_data.is_query:
                formatted_texts = [get_detailed_instruct(input_data.task_description, text) for text in input_data.texts]
            else:
                formatted_texts = input_data.texts
            
            batch_dict = gte_tokenizer(
                formatted_texts,
                max_length=8192,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = gte_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.cpu().numpy()
        
        return {
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1],
            "num_texts": len(input_data.texts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/analyze-text/")
def analyze_text(input_data: TextInput):
    """Analyze text to extract themes and emotions for story generation"""
    if gte_model is None:
        # Fallback to Hugging Face API
        try:
            url = HF_API_URL_TEMPLATE.format("nlptown/bert-base-multilingual-uncased-sentiment")
            payload = {"inputs": input_data.text}
            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            return {"analysis": response.json()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")
    
    try:
        # Create embeddings for the text
        theme_queries = [
            "What are the main themes in this text?",
            "What emotions are expressed in this text?",
            "What is the mood and tone of this text?",
            "What story elements are present in this text?"
        ]
        
        # Generate embeddings for queries and text
        if USE_SENTENCE_TRANSFORMERS:
            query_embeddings = gte_model.encode(
                [get_detailed_instruct(input_data.task_description, q) for q in theme_queries],
                prompt_name="query"
            )
            text_embedding = gte_model.encode([input_data.text])
        else:
            # Use transformers directly
            all_texts = [get_detailed_instruct(input_data.task_description, q) for q in theme_queries] + [input_data.text]
            
            batch_dict = gte_tokenizer(
                all_texts,
                max_length=8192,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            if torch.cuda.is_available():
                batch_dict = {k: v.cuda() for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = gte_model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                query_embeddings = embeddings[:len(theme_queries)]
                text_embedding = embeddings[len(theme_queries):].unsqueeze(0)
        
        # Calculate similarity scores
        scores = (query_embeddings @ text_embedding.T) * 100
        scores = scores.squeeze().tolist()
        
        # Create analysis result
        analysis = {
            "themes_relevance": scores[0],
            "emotions_relevance": scores[1],
            "mood_tone_relevance": scores[2],
            "story_elements_relevance": scores[3],
            "suggested_themes": []
        }
        
        # Suggest themes based on high scores
        if scores[0] > 70:
            analysis["suggested_themes"].append("strong thematic content")
        if scores[1] > 70:
            analysis["suggested_themes"].append("emotional depth")
        if scores[2] > 70:
            analysis["suggested_themes"].append("distinctive mood")
        if scores[3] > 70:
            analysis["suggested_themes"].append("narrative elements")
            
        return {"analysis": analysis}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/analyze-lyrics/")
def analyze_lyrics(input_data: LyricsAnalysisInput):
    """Analyze lyrics and suggest story themes"""
    try:
        # First, analyze the lyrics using our text analysis
        text_analysis = analyze_text(TextInput(text=input_data.lyrics))
        
        # Generate story prompts based on the analysis
        story_suggestions = []
        
        if text_analysis["analysis"]["emotions_relevance"] > 70:
            story_suggestions.append("A character experiencing intense emotions similar to those in the lyrics")
        
        if text_analysis["analysis"]["themes_relevance"] > 70:
            story_suggestions.append("A narrative exploring the core themes found in the lyrics")
            
        if text_analysis["analysis"]["mood_tone_relevance"] > 70:
            story_suggestions.append("A story that captures the same atmosphere and tone as the lyrics")
        
        # Add any user-provided themes
        if input_data.story_themes:
            story_suggestions.extend(input_data.story_themes)
        
        return {
            "analysis": text_analysis["analysis"],
            "story_suggestions": story_suggestions,
            "recommended_prompt": f"Write a story inspired by these themes: {', '.join(story_suggestions[:3])}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lyrics analysis failed: {str(e)}")

@app.post("/generate-story/")
def generate_story(story_prompt: StoryPrompt):
    """Generate a fictional story using MPT-7B-StoryWriter"""
    try:
        # Format the prompt for story generation
        formatted_prompt = f"Write a fictional story based on the following prompt:\n\n{story_prompt.prompt}\n\nStory:"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": story_prompt.max_length,
                "temperature": story_prompt.temperature,
                "top_p": story_prompt.top_p,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        url = HF_API_URL_TEMPLATE.format(STORY_GENERATION_MODEL)
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the generated text
        if isinstance(result, list):
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = result.get("generated_text", "")
        
        # Clean up the generated text
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return {
            "story": generated_text,
            "prompt_used": story_prompt.prompt,
            "parameters": {
                "max_length": story_prompt.max_length,
                "temperature": story_prompt.temperature,
                "top_p": story_prompt.top_p
            }
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            raise HTTPException(
                status_code=503, 
                detail="Model is currently loading. Please try again in a few moments."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.post("/lyrics-to-story/")
def lyrics_to_story(lyrics: str, max_length: int = 500):
    """Complete pipeline: analyze lyrics and generate a story"""
    try:
        # Step 1: Analyze lyrics
        lyrics_analysis = analyze_lyrics(LyricsAnalysisInput(lyrics=lyrics))
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)