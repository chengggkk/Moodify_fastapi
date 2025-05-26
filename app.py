import os
import torch
from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from environment variables")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY is missing from environment variables")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v3",
    token=HF_API_KEY,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    token=HF_API_KEY,
    trust_remote_code=True,
).to(device)

# 1. Embedding function
def embed_texts(texts: Union[str, List[str]]) -> List[float]:
    """
    Generate embedding vectors for English and Chinese texts using jinaai/jina-embeddings-v3.

    Args:
        texts (Union[str, List[str]]): A single string or a list of strings (English or Chinese).

    Returns:
        List[float]: Embedding vector (or list of vectors if input is a list).
    """
    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize and pad
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Get model outputs
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Average pooling — use attention mask to account for padding
    attention_mask = encoded_input["attention_mask"]
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    # Convert to list of floats
    return embeddings[0].cpu().tolist() if len(texts) == 1 else [e.cpu().tolist() for e in embeddings]

# 2. Mood analyze function
def analyze_mood(text: str, embedding: list) -> str:
    """
    Analyze mood and emotional tone of text using GPT-4.
    
    Args:
        text (str): Input text to analyze
        embedding (list): Embedding vector of the text
    
    Returns:
        str: Mood analysis result
    """
    prompt = f"""
请分析以下文本的情绪和心情状态。结合文字内容和语义向量信息进行分析。

文本内容:
{text[:800]}

嵌入向量（前10维）:
{embedding[:10]}...

请用简短中文描述文本的主要情绪、心情状态和情感色彩（3-5句）。
"""
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()

# 3. Lyrics analyze function
def analyze_lyrics(lyrics: str, embedding: list) -> str:
    """
    Analyze lyrics emotion, theme and atmosphere using GPT-4.
    
    Args:
        lyrics (str): Lyrics content
        embedding (list): Embedding vector of the lyrics
    
    Returns:
        str: Lyrics analysis result
    """
    prompt = f"""
下面是歌词内容和它的语义向量嵌入。请结合文字和语义信息，分析其情绪、主题和氛围。

歌词（前段）:
{lyrics[:800]}

嵌入向量（前10维）:
{embedding[:10]}...

请用简短中文描述歌词的主要情感和主题（3-5句）。
"""
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()

# 4. Story generate function
def generate_story(analysis: str) -> str:
    """
    Generate a short story based on analysis content.
    
    Args:
        analysis (str): Analysis content to base the story on
    
    Returns:
        str: Generated story
    """
    prompt = f"请基于以下分析内容，创作一段反映相似情感和主题的短篇小说：\n\n{analysis}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    return completion.choices[0].message.content.strip()

# Request models
class MoodAnalysisRequest(BaseModel):
    text: str

class StoryGenerationRequest(BaseModel):
    lyrics: str

# API 1: Mood analyzing (embedding function + mood analyze function)
@app.post("/analyze-mood")
def mood_analysis_api(req: MoodAnalysisRequest):
    """
    API for mood analysis using embedding and mood analysis functions.
    """
    try:
        # Generate embedding
        embedding = embed_texts(req.text)
        
        # Analyze mood
        mood_analysis = analyze_mood(req.text, embedding)
        
        return {
            "text_snippet": req.text[:200] + "..." if len(req.text) > 200 else req.text,
            "mood_analysis": mood_analysis,
            "embedding_dimensions": len(embedding)
        }
    except Exception as e:
        return {"error": f"Error analyzing mood: {str(e)}"}

# API 2: Story generation (embedding function + lyrics analyze function + story generation)
@app.post("/generate-story")
def story_generation_api(req: StoryGenerationRequest):
    """
    API for story generation using embedding, lyrics analysis, and story generation functions.
    """
    try:
        # Generate embedding
        embedding = embed_texts(req.lyrics)
        
        # Analyze lyrics
        lyrics_analysis = analyze_lyrics(req.lyrics, embedding)
        
        # Generate story
        story = generate_story(lyrics_analysis)
        
        return {
            "lyrics_snippet": req.lyrics[:200] + "..." if len(req.lyrics) > 200 else req.lyrics,
            "lyrics_analysis": lyrics_analysis,
            "story": story,
            "embedding_dimensions": len(embedding)
        }
    except Exception as e:
        return {"error": f"Error generating story: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)