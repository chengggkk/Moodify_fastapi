import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import torch
import openai
import urllib.parse

load_dotenv()
app = FastAPI()

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Load DeepSeek embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-embedding-2")
model = AutoModel.from_pretrained("deepseek-ai/deepseek-embedding-2").to(device)

# --- Step 1: Fetch lyrics from SerpAPI ---
def fetch_lyrics(song: str, artist: str) -> str:
    search_query = f"{song} {artist} lyrics"
    encoded_query = urllib.parse.quote(search_query)
    serp_url = f"https://serpapi.com/search.json?engine=google&q={encoded_query}&api_key={SERPAPI_KEY}"

    response = requests.get(serp_url).json()
    for result in response.get("organic_results", []):
        url = result.get("link", "")
        if "lyrics" in url or "歌詞" in url:
            lyrics = scrape_lyrics(url)
            if lyrics:
                return lyrics
    return "Lyrics not found."

def scrape_lyrics(url: str) -> str:
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        lyrics = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        return lyrics
    except Exception:
        return ""

# --- Step 2: Embed lyrics using DeepSeek ---
def embed_lyrics(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        return embeddings[0].cpu().tolist()

# --- Step 3: Analyze with OpenAI ---
def analyze_lyrics(lyrics: str, embedding: list) -> str:
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
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()

# --- Step 4: Generate Story ---
def generate_story(analysis: str) -> str:
    prompt = f"请基于以下歌词分析，创作一段反映相似情感和主题的短篇小说：\n\n{analysis}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )
    return completion.choices[0].message.content.strip()

# --- Request Schema ---
class SongRequest(BaseModel):
    title: str
    artist: str

# --- Endpoint ---
@app.post("/generate-story")
def process_song(req: SongRequest):
    lyrics = fetch_lyrics(req.title, req.artist)
    if "Lyrics not found" in lyrics:
        return {"error": "Could not find lyrics."}

    embedding = embed_lyrics(lyrics)
    analysis = analyze_lyrics(lyrics, embedding)
    story = generate_story(analysis)

    return {
        "lyrics_snippet": lyrics[:500] + "...",
        "analysis": analysis,
        "story": story
    }