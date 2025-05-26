import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class SongRequest(BaseModel):
    title: str
    artist: str

# --- Step 1: Search & Scrape from lyrics.com ---
def fetch_lyrics(song: str, artist: str) -> str:
    search_url = f"https://www.lyrics.com/serp.php?st={requests.utils.quote(song)}&qtype=2"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find first result that matches artist
    links = soup.select("td.tal.qx > strong > a")
    for link in links:
        title = link.text.strip().lower()
        song_link = link["href"]
        parent_row = link.find_parent("tr")
        if artist.lower() in parent_row.text.lower():
            lyrics_url = f"https://www.lyrics.com{song_link}"
            return scrape_lyrics_from_url(lyrics_url)

    return "Lyrics not found."

def scrape_lyrics_from_url(url: str) -> str:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    lyrics_div = soup.find("pre", {"id": "lyric-body-text"})
    if lyrics_div:
        return lyrics_div.get_text(separator="\n").strip()
    return ""

# --- Step 2: Analyze with OpenAI Embedding + GPT ---
def analyze_lyrics(lyrics: str) -> str:
    # You can add embedding if needed
    prompt = f"Analyze the tone, themes, and emotional meaning of the following song lyrics:\n\n{lyrics[:1500]}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return completion.choices[0].message.content.strip()

# --- Step 3: Generate a Story ---
def generate_story(analysis: str) -> str:
    prompt = f"Based on the following analysis of a song's lyrics, write a short creative story with similar mood and themes:\n\n{analysis}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return completion.choices[0].message.content.strip()

# --- API Endpoint ---
@app.post("/generate-story")
def generate(request: SongRequest):
    lyrics = fetch_lyrics(request.title, request.artist)
    if not lyrics or "Lyrics not found" in lyrics:
        return {"error": "Lyrics not found."}

    analysis = analyze_lyrics(lyrics)
    story = generate_story(analysis)

    return {
        "lyrics": lyrics[:400] + "...",
        "analysis": analysis,
        "story": story
    }