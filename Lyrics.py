from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import os
from typing import Optional
from difflib import SequenceMatcher
import unicodedata
import re

lyrics_router = APIRouter(prefix="/lyrics", tags=["lyrics"])

class LyricsRequest(BaseModel):
    title: str
    artist: str

class LyricsResponse(BaseModel):
    title: str
    artist: str
    lyrics: str 
    source: str

@lyrics_router.post("/search", response_model=LyricsResponse)
async def get_lyrics(request: LyricsRequest):
    try:
        genius_result = await search_genius_api(request.title, request.artist)

        if genius_result:
            genius_url = genius_result.get("url", "https://genius.com")
            return LyricsResponse(
                title=request.title,
                artist=request.artist,
                lyrics=genius_url,
                source="genius"
            )

        search_urls = await search_web_for_lyrics(request.title, request.artist)

        if search_urls:
            processed_content = await process_lyrics_content(search_urls[0])
            return LyricsResponse(
                title=request.title,
                artist=request.artist,
                lyrics=processed_content,
                source="web_search"
            )

        raise HTTPException(status_code=404, detail="Lyrics not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s.lower())
    return s.strip()


def calculate_song_similarity(query_artist, query_title, result_artist, result_title, original_artist, original_title):
    artist_score = SequenceMatcher(None, query_artist, result_artist).ratio() * 100
    title_score = SequenceMatcher(None, query_title, result_title).ratio() * 100
    boost = 0
    if original_artist.lower() == result_artist.lower():
        boost += 10
    if original_title.lower() == result_title.lower():
        boost += 10
    return int((artist_score + title_score) / 2 + boost)


async def search_genius_api(title: str, artist: str) -> Optional[dict]:
    genius_token = os.getenv("GENIUS_API_KEY")
    if not genius_token:
        return None

    headers = {"Authorization": f"Bearer {genius_token}"}
    params = {"q": f"{artist} {title}"}

    try:
        response = requests.get("https://api.genius.com/search", headers=headers, params=params)
        response.raise_for_status()
        hits = response.json().get("response", {}).get("hits", [])
        if not hits:
            return None

        best_match, best_score = None, 0
        query_artist = normalize_string(artist)
        query_title = normalize_string(title)

        for hit in hits:
            result = hit.get("result", {})
            result_artist = normalize_string(result.get("primary_artist", {}).get("name", ""))
            result_title = normalize_string(result.get("title", ""))
            score = calculate_song_similarity(query_artist, query_title, result_artist, result_title, artist, title)
            if score > best_score:
                best_score, best_match = score, result

        return best_match if best_score >= 70 else None
    except requests.RequestException:
        return None


async def search_web_for_lyrics(title: str, artist: str) -> list:
    brave_api_key = os.getenv("BRAVE_API_KEY")
    if not brave_api_key:
        return []
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_api_key
    }
    
    params = {
        "q": f"{title} {artist} lyrics",
        "count": 3,
        "result_filter": "web"
    }

    try:
        response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
        response.raise_for_status()
        return [r.get("url") for r in response.json().get("web", {}).get("results", [])][:3]
    except requests.RequestException:
        return []


async def process_lyrics_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator='\n')
        text = re.sub(r'\n+', '\n', text).strip()
        
        return await format_with_mistral(text)

    except requests.RequestException:
        return "Content extraction failed"


async def format_with_mistral(content: str) -> str:
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        return "Mistral API key not available"

    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-large-latest",  # or your specific hosted model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that extracts only the clean lyrics from a web article."},
            {"role": "user", "content": f"Please extract only the lyrics from the following text and format(not markdown format, just simply clean and format):\n\n{content}"}
        ]
    }

    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Failed to format lyrics with Mistral: {str(e)}"


@lyrics_router.get("/info/{title}/{artist}")
async def get_song_info(title: str, artist: str):
    return {
        "title": title,
        "artist": artist,
        "message": "For lyrics, please visit official sources like the artist's website or licensed services.",
        "legal_alternatives": [
            "Official artist websites",
            "Licensed streaming services",
            "Purchase from digital music stores",
            "Official music videos with lyrics"
        ]
    }