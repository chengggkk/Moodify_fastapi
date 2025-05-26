import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Initialize FastAPI and embedding model
app = FastAPI()
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Step 1: Search and Scrape Lyrics ---
def fetch_lyrics(song: str, artist: str) -> str:
    query = f"{song} {artist} lyrics"
    search_url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}&engine=google"

    response = requests.get(search_url).json()
    for result in response.get("organic_results", []):
        url = result.get("link", "")
        if "lyrics" in url:
            lyrics = scrape_lyrics(url)
            if lyrics:
                return lyrics
    return "Lyrics not found."

def scrape_lyrics(url: str) -> str:
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all("p")
        lyrics = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        return lyrics
    except Exception:
        return ""

# --- Step 2: Embed with Hugging Face ---
def embed_lyrics(lyrics: str) -> list:
    return embedding_model.encode(lyrics, convert_to_numpy=True).tolist()

# --- Step 3: Analyze with OpenAI ---
def analyze_lyrics(lyrics: str, embedding: list) -> str:
    prompt = f"""
Below are song lyrics and their semantic embedding vector.
Analyze the tone, mood, and main theme of the lyrics. Use both the text and embedding for a deep understanding.

Lyrics:
{lyrics[:1000]}

Embedding (shortened):
{embedding[:10]}...

Provide a concise analysis in 3-5 sentences.
"""
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()

# --- Step 4: Generate Story ---
def generate_story(analysis: str) -> str:
    prompt = f"Based on this song analysis:\n\n{analysis}\n\nWrite a creative short story that reflects the same themes and emotions."
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return completion.choices[0].message.content.strip()

# --- Request Schema ---
class SongRequest(BaseModel):
    title: str
    artist: str

# --- FastAPI Endpoint ---
@app.post("/generate-story")
def process_song(request: SongRequest):
    lyrics = fetch_lyrics(request.title, request.artist)
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