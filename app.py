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

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from environment variables")
if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY is missing from environment variables")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY is missing from environment variables")

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

def fetch_lyrics_with_serpapi(song: str, artist: str) -> str:
    """Fetch lyrics using SerpAPI Google Search"""
    try:
        print(f"Fetching lyrics via SerpAPI: {artist} - {song}")
        
        # Search for lyrics using SerpAPI
        search_query = f"{artist} {song} lyrics"
        serp_url = "https://serpapi.com/search.json"
        
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERPAPI_KEY
        }
        
        response = requests.get(serp_url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"SerpAPI request failed: {response.status_code}")
            return None
            
        data = response.json()
        print("SerpAPI search completed")
        
        lyrics_content = None
        
        # Check knowledge graph for direct lyrics
        if data.get("knowledge_graph") and data["knowledge_graph"].get("lyrics"):
            lyrics_content = data["knowledge_graph"]["lyrics"]
            print("Found lyrics in knowledge graph")
        
        # Check organic results for lyrics sites
        if not lyrics_content and data.get("organic_results"):
            lyrics_urls = [
                result for result in data["organic_results"]
                if result.get("link") and (
                    result["link"].find("genius.com") != -1 or
                    result["link"].find("azlyrics.com") != -1 or
                    result["link"].find("lyrics.com") != -1 or
                    result["link"].find("metrolyrics.com") != -1
                )
            ][:3]  # Try first 3 results
            
            for result in lyrics_urls:
                try:
                    print(f"Trying to fetch from: {result['link']}")
                    lyrics_content = scrape_lyrics_from_url(result["link"])
                    if lyrics_content and len(lyrics_content.split()) > 30:
                        print("Successfully fetched lyrics from URL")
                        break
                except Exception as e:
                    print(f"Failed to fetch from {result['link']}: {e}")
                    continue
        
        # Check answer box or featured snippet
        if not lyrics_content and data.get("answer_box"):
            if data["answer_box"].get("snippet"):
                lyrics_content = data["answer_box"]["snippet"]
                print("Found lyrics in answer box")
        
        return lyrics_content
        
    except Exception as e:
        print(f"SerpAPI lyrics fetch error: {e}")
        return None

def scrape_lyrics_from_url(url: str) -> str:
    """Enhanced lyrics scraper for various sites"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        lyrics = ""
        
        if "genius.com" in url:
            # Extract from Genius using multiple selectors
            selectors = [
                '[data-lyrics-container="true"]',
                '.Lyrics__Container-sc-1ynbvzw-1',
                '[class*="Lyrics__Container"]',
                '.lyrics'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    lyrics = '\n'.join(
                        element.get_text(separator='\n', strip=True) 
                        for element in elements
                    )
                    break
                    
        elif "azlyrics.com" in url:
            # Extract from AZLyrics pattern
            # Look for the comment pattern that indicates lyrics start
            lyrics_match = re.search(
                r'<!-- Usage of azlyrics\.com content.*?-->(.*?)<!-- MxM banner -->',
                html,
                re.DOTALL
            )
            if lyrics_match:
                lyrics_html = lyrics_match.group(1)
                lyrics_soup = BeautifulSoup(lyrics_html, 'html.parser')
                lyrics = lyrics_soup.get_text(separator='\n', strip=True)
            else:
                # Fallback: look for divs without class/id (AZLyrics style)
                divs = soup.find_all('div', {'class': False, 'id': False})
                for div in divs:
                    text = div.get_text(strip=True)
                    if len(text) > 200 and '\n' in text:
                        lyrics = text
                        break
                        
        else:
            # Generic extraction for other sites
            # Try common lyrics selectors
            selectors = [
                '.lyrics', '.lyric-body', '.song-lyrics', 
                '#lyrics', '.lyricsh', '[class*="lyric"]'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    lyrics = '\n'.join(
                        element.get_text(separator='\n', strip=True) 
                        for element in elements
                    )
                    break
            
            # If no specific selectors work, try to find large text blocks
            if not lyrics:
                all_divs = soup.find_all('div')
                candidates = []
                
                for div in all_divs:
                    text = div.get_text(separator='\n', strip=True)
                    # Filter for text that looks like lyrics
                    if (len(text) > 200 and 
                        text.count('\n') > 5 and 
                        len(text.split()) > 50):
                        candidates.append(text)
                
                if candidates:
                    # Return the longest candidate
                    lyrics = max(candidates, key=len)
        
        # Clean up the lyrics
        if lyrics:
            lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
            lyrics = '\n'.join(lines)
            
            # Remove common unwanted patterns
            unwanted_patterns = [
                r'advertisement',
                r'sponsor',
                r'click here',
                r'www\.',
                r'\.com',
                r'copyright',
                r'all rights reserved'
            ]
            
            for pattern in unwanted_patterns:
                lyrics = re.sub(pattern, '', lyrics, flags=re.IGNORECASE)
            
            # Final cleanup
            lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)  # Remove excessive newlines
            lyrics = lyrics.strip()
        
        return lyrics if lyrics and len(lyrics.split()) > 30 else None
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def format_lyrics_with_ai(raw_lyrics: str, artist: str, song: str) -> str:
    """Use OpenAI to clean and format the lyrics"""
    try:
        print("Formatting lyrics with AI...")
        
        format_prompt = f"""Please clean and format the following raw lyrics content:

Raw content:
{raw_lyrics[:2000]}

Song information:
- Artist: {artist}
- Song: {song}

Please perform the following processing:
1. Remove irrelevant website information, ads, copyright notices
2. Keep complete lyrics content
3. Organize paragraph structure with proper line breaks
4. Remove repetitive markers like [Verse], [Chorus] etc.
5. Ensure lyrics completeness and readability

Return only the cleaned lyrics content without any additional explanatory text."""

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": format_prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        formatted_lyrics = completion.choices[0].message.content.strip()
        print("Lyrics formatted successfully with AI")
        return formatted_lyrics
        
    except Exception as e:
        print(f"AI formatting error: {e}")
        # Return cleaned version without AI if AI fails
        return clean_lyrics_basic(raw_lyrics)

def clean_lyrics_basic(raw_lyrics: str) -> str:
    """Basic lyrics cleaning without AI"""
    try:
        # Remove common unwanted patterns
        unwanted_patterns = [
            r'advertisement.*?\n',
            r'sponsor.*?\n',
            r'click here.*?\n',
            r'www\..*?\n',
            r'\.com.*?\n',
            r'copyright.*?\n',
            r'all rights reserved.*?\n',
            r'\[.*?\]',  # Remove [Verse], [Chorus] etc.
            r'Embed$',
            r'Lyrics$'
        ]
        
        cleaned = raw_lyrics
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up whitespace
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
        
    except Exception:
        return raw_lyrics

def try_fallback_methods(song: str, artist: str) -> str:
    """Fallback methods if SerpAPI fails"""
    try:
        print(f"Trying fallback methods for: {artist} - {song}")
        
        # Try some direct site approaches as last resort
        fallback_sources = []
        
        # Try Genius direct
        song_clean = clean_text(song).replace(' ', '-')
        artist_clean = clean_text(artist).replace(' ', '-')
        genius_url = f"https://genius.com/{artist_clean}-{song_clean}-lyrics"
        fallback_sources.append(genius_url)
        
        # Try variations
        if ' ' in song:
            song_alt = song.replace(' ', '')
            genius_alt = f"https://genius.com/{artist_clean}-{song_alt}-lyrics"
            fallback_sources.append(genius_alt)
        
        for url in fallback_sources:
            try:
                print(f"Trying fallback: {url}")
                lyrics = scrape_lyrics_from_url(url)
                if lyrics and len(lyrics.split()) > 30:
                    print(f"Found lyrics from fallback: {url}")
                    return lyrics
                time.sleep(1)  # Be respectful
            except Exception as e:
                print(f"Fallback failed for {url}: {e}")
                continue
        
        return None
        
    except Exception:
        return None

def fetch_lyrics(song: str, artist: str) -> str:
    """Main function to fetch lyrics using SerpAPI"""
    if not SERPAPI_KEY:
        return "SERPAPI_KEY is missing. Please add it to your .env file."
    
    print(f"Searching for lyrics: {song} by {artist}")
    
    # Primary method: Use SerpAPI
    lyrics = fetch_lyrics_with_serpapi(song, artist)
    
    if lyrics and len(lyrics.split()) > 20:
        print("Successfully found lyrics via SerpAPI")
        return format_lyrics_with_ai(lyrics, artist, song)
    
    # Fallback: Try direct scraping as backup
    print("SerpAPI didn't return lyrics, trying fallback methods...")
    fallback_lyrics = try_fallback_methods(song, artist)
    
    if fallback_lyrics:
        return format_lyrics_with_ai(fallback_lyrics, artist, song)
    
    return "Lyrics not found."

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