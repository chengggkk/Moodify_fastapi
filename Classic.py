from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import httpx
import asyncio
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
import re

# Create router with prefix and tags
music_router = APIRouter(prefix="/music", tags=["music"])

# Request/Response Models
class MusicPrompt(BaseModel):
    prompt: str

class Song(BaseModel):
    title: str
    artist: str

class MusicResponse(BaseModel):
    songs: List[Song]
    query_used: str
    timestamp: str

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

class MusicRecommendationService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def analyze_prompt_with_mistral(self, user_prompt: str) -> str:
        """Step 1: Analyze user prompt and generate search query using Mistral AI"""
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a music search query generator. Given a user's music request, 
                    generate an optimized search query that would help find relevant songs. 
                    Focus on key terms like genre, artist, mood, year, or specific characteristics.
                    Return only the search query, nothing else."""
                },
                {
                    "role": "user",
                    "content": f"Generate a search query for: {user_prompt}"
                }
            ],
            "max_tokens": 100,
            "temperature": 0.3
        }
        
        try:
            response = await self.client.post(MISTRAL_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            search_query = result["choices"][0]["message"]["content"].strip()
            return search_query
        except Exception as e:
            # Fallback: create basic search query from prompt
            return f"{user_prompt} songs music"
    
    async def search_with_brave(self, query: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Step 2: Search using Brave Search API and return results + URLs"""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": 10,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate"
        }
        
        try:
            response = await self.client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            results = search_results.get("web", {}).get("results", [])
            
            # Extract URLs for HTML parsing
            urls = [result.get("url", "") for result in results if result.get("url")]
            
            return results, urls
        except Exception as e:
            print(f"Brave search error: {e}")
            return [], []
    
    async def extract_html_content(self, url: str) -> str:
        """Extract and clean HTML content using BeautifulSoup"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = await self.client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            return text[:2000] if len(text) > 2000 else text
            
        except Exception as e:
            print(f"HTML extraction error for {url}: {e}")
            return ""
    
    async def extract_music_info_from_html(self, urls: List[str]) -> str:
        """Extract music-related information from multiple URLs"""
        extracted_content = []
        
        for url in urls[:3]:  # Limit to first 3 URLs to avoid too many requests
            content = await self.extract_html_content(url)
            if content:
                # Look for music-related keywords
                music_keywords = ['song', 'artist', 'album', 'music', 'track', 'singer', 'band', 'OST', 'soundtrack']
                content_lower = content.lower()
                
                if any(keyword in content_lower for keyword in music_keywords):
                    extracted_content.append(content)
        
        return "\n\n".join(extracted_content)
    
    async def generate_songs_with_openai(self, user_prompt: str, search_results: List[Dict], html_content: str) -> List[Song]:
        """Step 3: Analyze search results and HTML content to generate song recommendations using OpenAI"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare search context
        search_context = ""
        for result in search_results[:5]:  # Use top 5 results
            search_context += f"Title: {result.get('title', '')}\nSnippet: {result.get('description', '')}\n\n"
        
        # Add HTML content if available
        html_section = ""
        if html_content:
            html_section = f"\nExtracted Website Content:\n{html_content[:1500]}\n"  # Limit to 1500 chars
        
        system_prompt = """You are a music recommendation expert. Based on the user's request, search results, 
        and extracted website content, recommend 8-12 songs that match their criteria. 
        Use the website content to find actual song titles and artists when possible.
        
        Return ONLY a valid JSON array with objects containing 'title' and 'artist' fields. 
        Do not include any additional text or formatting.
        
        Example format:
        [
            {"title": "Song Name", "artist": "Artist Name"},
            {"title": "Another Song", "artist": "Another Artist"}
        ]"""
        
        user_message = f"""User Request: {user_prompt}

Search Results Context:
{search_context}
{html_section}

Based on the user's request, search context, and website content, provide song recommendations in the specified JSON format."""
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1200,
            "temperature": 0.7
        }
        
        try:
            response = await self.client.post(OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            songs_data = json.loads(content)
            songs = [Song(title=song["title"], artist=song["artist"]) for song in songs_data]
            return songs
            
        except json.JSONDecodeError:
            # Fallback: create default songs based on prompt keywords
            return self.create_fallback_songs(user_prompt)
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return self.create_fallback_songs(user_prompt)
    
    def create_fallback_songs(self, prompt: str) -> List[Song]:
        """Create fallback songs when AI services fail"""
        fallback_songs = {
            "k-drama": [
                Song(title="You Are My Everything", artist="Gummy"),
                Song(title="Always", artist="Yoon Mirae"),
                Song(title="Stay With Me", artist="Chanyeol & Punch")
            ],
            "taylor swift": [
                Song(title="All Too Well", artist="Taylor Swift"),
                Song(title="Ronan", artist="Taylor Swift"),
                Song(title="Soon You'll Get Better", artist="Taylor Swift")
            ],
            "chinese": [
                Song(title="月亮代表我的心", artist="Teresa Teng"),
                Song(title="童话", artist="Guang Liang"),
                Song(title="听海", artist="A-Mei")
            ]
        }
        
        prompt_lower = prompt.lower()
        for key, songs in fallback_songs.items():
            if key in prompt_lower:
                return songs
        
        # Default fallback
        return [
            Song(title="Shape of You", artist="Ed Sheeran"),
            Song(title="Blinding Lights", artist="The Weeknd"),
            Song(title="Watermelon Sugar", artist="Harry Styles")
        ]
    
    async def close(self):
        await self.client.aclose()

# Initialize service
music_service = MusicRecommendationService()

@music_router.post("/recommend", response_model=MusicResponse)
async def recommend_music(request: MusicPrompt):
    """
    Main endpoint for music recommendations
    
    Process:
    1. Analyze prompt with Mistral AI to generate search query
    2. Search with Brave Search API synchronously 
    3. Extract HTML content from search result URLs using BeautifulSoup
    4. Analyze results and HTML content to generate recommendations with OpenAI
    
    Returns JSON formatted song list with title and artist
    """
    try:
        # Step 1: Analyze prompt and generate search query
        search_query = await music_service.analyze_prompt_with_mistral(request.prompt)
        print(search_query)
        
        # Step 2: Search with Brave API and get URLs
        search_results, urls = await music_service.search_with_brave(search_query)
        print(search_results, urls)
        
        # Step 2.5: Extract HTML content from search result URLs
        html_content = await music_service.extract_music_info_from_html(urls)
        print(html_content)
        
        # Step 3: Generate song recommendations with OpenAI using search results and HTML content
        songs = await music_service.generate_songs_with_openai(request.prompt, search_results, html_content)
        
        return MusicResponse(
            songs=songs,
            query_used=search_query,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@music_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@music_router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await music_service.close()

# Example usage endpoints for testing
@music_router.get("/examples")
async def get_examples():
    """Get example prompts for testing"""
    return {
        "examples": [
            "give me some K-drama romantic OST",
            "recommend taylor swift sad songs", 
            "2004 Chinese tracks",
            "upbeat workout songs",
            "jazz classics from the 1960s",
            "indie folk songs for rainy days"
        ]
    }

# Example of how to use this router in main app:
"""
from fastapi import FastAPI
from this_router_file import music_router

app = FastAPI(title="Music Recommendation API", version="1.0.0")
app.include_router(music_router)

# Required dependencies:
# pip install fastapi uvicorn httpx pydantic beautifulsoup4

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""