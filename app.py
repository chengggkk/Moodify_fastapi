import os
import requests
import urllib.parse
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel
import openai
import re
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 tokenizer 和 model
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

def clean_text(text):
    """Clean and normalize text for better matching"""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def fetch_lyrics_genius(song: str, artist: str) -> str:
    """Fetch lyrics from Genius.com"""
    try:
        # Format the URL for Genius
        song_clean = clean_text(song).replace(' ', '-')
        artist_clean = clean_text(artist).replace(' ', '-')
        url = f"https://genius.com/{artist_clean}-{song_clean}-lyrics"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Genius uses different selectors for lyrics
            lyrics_containers = [
                '[data-lyrics-container="true"]',
                '.Lyrics__Container-sc-1ynbvzw-1',
                '.lyrics',
                '[class*="Lyrics"]'
            ]
            
            for selector in lyrics_containers:
                elements = soup.select(selector)
                if elements:
                    lyrics_text = ""
                    for element in elements:
                        lyrics_text += element.get_text(separator='\n', strip=True) + '\n'
                    
                    if lyrics_text.strip() and len(lyrics_text.split()) > 20:
                        return lyrics_text.strip()
        
        return None
    except Exception as e:
        print(f"Error fetching from Genius: {e}")
        return None

def fetch_lyrics_azlyrics(song: str, artist: str) -> str:
    """Fetch lyrics from AZLyrics"""
    try:
        # Format for AZLyrics URL structure
        artist_clean = re.sub(r'[^a-z0-9]', '', artist.lower())
        song_clean = re.sub(r'[^a-z0-9]', '', song.lower())
        url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{song_clean}.html"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # AZLyrics stores lyrics in divs without class
            lyrics_div = soup.find('div', class_=False, id=False)
            if lyrics_div and lyrics_div.get_text(strip=True):
                lyrics_text = lyrics_div.get_text(separator='\n', strip=True)
                if len(lyrics_text.split()) > 20:
                    return lyrics_text
        
        return None
    except Exception as e:
        print(f"Error fetching from AZLyrics: {e}")
        return None

def search_lyrics_with_serpapi(song: str, artist: str) -> str:
    """Use SerpAPI to find lyrics if available"""
    if not SERPAPI_KEY:
        return None
        
    try:
        query = f"{song} {artist} lyrics site:genius.com OR site:azlyrics.com"
        url = "https://serpapi.com/search"
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            for result in data.get("organic_results", []):
                link = result.get("link", "")
                if "genius.com" in link or "azlyrics.com" in link:
                    lyrics = scrape_lyrics_from_url(link)
                    if lyrics and len(lyrics.split()) > 20:
                        return lyrics
        
        return None
    except Exception as e:
        print(f"Error with SerpAPI: {e}")
        return None

def scrape_lyrics_from_url(url: str) -> str:
    """Generic lyrics scraper for various sites"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Try different selectors based on common lyrics sites
        selectors = [
            '[data-lyrics-container="true"]',  # Genius
            '.Lyrics__Container-sc-1ynbvzw-1',  # Genius new
            '.lyrics',  # Generic
            '.lyricsh',  # Some sites
            '#lyrics',  # ID-based
            '.song-lyrics',  # Generic
            'div[class=""]',  # AZLyrics style
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator='\n', strip=True)
                # Check if this looks like lyrics (reasonable length, multiple lines)
                if text and len(text.split()) > 30 and '\n' in text:
                    # Clean up the text
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    lyrics = '\n'.join(lines)
                    return lyrics
        
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def fetch_lyrics(song: str, artist: str) -> str:
    """Main function to fetch lyrics from multiple sources"""
    print(f"Searching for lyrics: {song} by {artist}")
    
    # Try different sources in order of reliability
    sources = [
        ("Genius", lambda: fetch_lyrics_genius(song, artist)),
        ("AZLyrics", lambda: fetch_lyrics_azlyrics(song, artist)),
        ("SerpAPI", lambda: search_lyrics_with_serpapi(song, artist)),
    ]
    
    for source_name, fetch_func in sources:
        try:
            print(f"Trying {source_name}...")
            lyrics = fetch_func()
            if lyrics and len(lyrics.split()) > 20:
                print(f"Found lyrics from {source_name}")
                return lyrics
            time.sleep(1)  # Be respectful with requests
        except Exception as e:
            print(f"Error with {source_name}: {e}")
            continue
    
    # If all else fails, try a broader search approach
    print("Trying alternative search methods...")
    alternative_lyrics = try_alternative_search(song, artist)
    if alternative_lyrics:
        return alternative_lyrics
    
    return "Lyrics not found."

def try_alternative_search(song: str, artist: str) -> str:
    """Try alternative methods to find lyrics"""
    try:
        # Try different variations of the song/artist name
        variations = [
            (song, artist),
            (song.replace("'", ""), artist),
            (song.replace("&", "and"), artist),
            (re.sub(r'\([^)]*\)', '', song).strip(), artist),  # Remove parentheses
        ]
        
        for s, a in variations:
            if s != song or a != artist:  # Only try if it's different
                lyrics = fetch_lyrics_genius(s, a)
                if lyrics and len(lyrics.split()) > 20:
                    return lyrics
        
        return None
    except Exception:
        return None

# 用 Jina model 做歌詞 embedding
def embed_lyrics(text: str, task: str = "text-matching", max_length: int = 2048, truncate_dim: int = 768) -> list:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = model.encode(
            [text] if isinstance(text, str) else text,
            task=task,
            max_length=max_length,
            truncate_dim=truncate_dim,
        )
    return outputs[0].cpu().tolist()

# 用 OpenAI GPT-4 分析歌詞情緒與主題
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
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()

# 基于分析内容生成短篇故事
def generate_story(analysis: str) -> str:
    prompt = f"请基于以下歌词分析，创作一段反映相似情感和主题的短篇小说：\n\n{analysis}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    return completion.choices[0].message.content.strip()

class SongRequest(BaseModel):
    title: str
    artist: str

@app.post("/generate-story")
def process_song(req: SongRequest):
    lyrics = fetch_lyrics(req.title, req.artist)
    if "Lyrics not found" in lyrics:
        return {"error": "Could not find lyrics for this song. Please check the song title and artist name."}

    embedding = embed_lyrics(lyrics)
    analysis = analyze_lyrics(lyrics, embedding)
    story = generate_story(analysis)

    return {
        "lyrics_snippet": lyrics[:500] + "..." if len(lyrics) > 500 else lyrics,
        "analysis": analysis,
        "story": story
    }