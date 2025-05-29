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
from concurrent.futures import ThreadPoolExecutor
import requests

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
    queries_used: List[str]
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
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def generate_search_queries_with_mistral(self, user_prompt: str) -> List[str]:
        """Step 1: Generate multiple refined search queries using Mistral AI"""
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
                    generate 3 different optimized search queries that would help find relevant songs.
                    
                    For each query, consider:
                    1. General English search terms
                    2. Native language terms (if applicable - Korean for K-drama, Chinese for C-pop, etc.)
                    3. Trending/temporal variations (current year, "best of", "top 10", etc.)
                    
                    Return ONLY a JSON array of 3 search query strings, nothing else.
                    
                    Examples:
                    Input: "give me K-drama romantic OST"
                    Output: ["best K-drama romantic OST", "최고의 로맨틱 OST", "top 10 K-drama OST this year"]
                    
                    Input: "taylor swift sad songs"
                    Output: ["taylor swift sad songs", "taylor swift heartbreak ballads", "taylor swift emotional songs best"]
                    """
                },
                {
                    "role": "user",
                    "content": f"Generate 3 search queries for: {user_prompt}"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.4
        }
        
        try:
            response = await self.client.post(MISTRAL_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            queries_text = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            queries = json.loads(queries_text)
            return queries if isinstance(queries, list) and len(queries) == 3 else [user_prompt]
            
        except Exception as e:
            print(f"Mistral query generation error: {e}")
            # Fallback: create basic variations manually
            return self.create_fallback_queries(user_prompt)
    
    def create_fallback_queries(self, prompt: str) -> List[str]:
        """Create fallback search queries when Mistral fails"""
        base_query = prompt.lower()
        
        # Generate variations
        queries = [
            prompt,
            f"best {prompt}",
            f"top {prompt} songs"
        ]
        
        # Add language-specific variations
        if "k-drama" in base_query or "korean" in base_query:
            queries[2] = f"{prompt} 한국"
        elif "chinese" in base_query or "c-pop" in base_query:
            queries[2] = f"{prompt} 中文"
        elif "japanese" in base_query or "j-pop" in base_query:
            queries[2] = f"{prompt} 日本"
        
        return queries
    
    def search_brave_sync(self, query: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Synchronous Brave search for better performance"""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_API_KEY
        }
        
        params = {
            "q": query,
            "count": 8,
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate"
        }
        
        try:
            response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            results = search_results.get("web", {}).get("results", [])
            
            # Extract URLs for HTML parsing
            urls = [result.get("url", "") for result in results if result.get("url")]
            
            return results, urls
        except Exception as e:
            print(f"Brave search error for query '{query}': {e}")
            return [], []
    
    async def search_multiple_queries_sync(self, queries: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
        """Step 2: Search multiple queries synchronously using ThreadPoolExecutor"""
        loop = asyncio.get_event_loop()
        
        # Run all searches concurrently in thread pool
        search_tasks = [
            loop.run_in_executor(self.executor, self.search_brave_sync, query)
            for query in queries
        ]
        
        # Wait for all searches to complete
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine all results
        all_results = []
        all_urls = []
        
        for result in search_results_list:
            if isinstance(result, tuple) and len(result) == 2:
                results, urls = result
                all_results.extend(results)
                all_urls.extend(urls)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen_urls = set()
        for url in all_urls:
            if url not in seen_urls:
                unique_urls.append(url)
                seen_urls.add(url)
        
        return all_results, unique_urls[:15]  # Limit to top 15 URLs
    
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
        # Process first 5 URLs concurrently for speed
        extraction_tasks = [
            self.extract_html_content(url) 
            for url in urls[:5]
        ]
        
        contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        extracted_content = []
        music_keywords = ['song', 'artist', 'album', 'music', 'track', 'singer', 'band', 'OST', 'soundtrack']
        
        for content in contents:
            if isinstance(content, str) and content:
                content_lower = content.lower()
                if any(keyword in content_lower for keyword in music_keywords):
                    extracted_content.append(content)
        
        return "\n\n".join(extracted_content)
    
    async def generate_songs_with_openai(self, user_prompt: str, search_results: List[Dict], html_content: str, queries_used: List[str]) -> List[Song]:
        """Step 3: Analyze search results and HTML content to generate song recommendations using OpenAI"""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare search context from multiple query results
        search_context = f"Search queries used: {', '.join(queries_used)}\n\n"
        for i, result in enumerate(search_results[:10]):  # Use top 10 results
            search_context += f"Result {i+1}:\nTitle: {result.get('title', '')}\nSnippet: {result.get('description', '')}\nURL: {result.get('url', '')}\n\n"
        
        # Add HTML content if available
        html_section = ""
        if html_content:
            html_section = f"\nExtracted Website Content:\n{html_content[:2000]}\n"  # Limit to 2000 chars
        
        system_prompt = """You are a music recommendation expert with deep knowledge of global music across genres and decades. Your goal is to recommend 10–15 songs that match the user’s request using detailed reasoning and validation.

You are given:
1. The user's request (mood, genre, story, vibe, theme, etc.).
2. Multiple search results derived from refined queries targeting various musical aspects (e.g., emotion, lyrics, historical context).
3. Extracted content from song lyrics, reviews, playlist curation sites, forums, and other sources.

Follow these steps:
- **Step 1 (Comprehension):** Understand the user's intent. What mood, style, or theme are they requesting?
- **Step 2 (Cross-reference):** Analyze the search results and extracted content. Identify recurring song mentions, lyrical themes, or genre tags that align with the user’s intent.
- **Step 3 (Self-consistency):** Choose songs that are consistently relevant across multiple sources.
- **Step 4 (Emotion tuning):** Ensure the tone, emotion, or storytelling of the song matches the user's desired theme (e.g., nostalgic, hopeful, dark, energetic).
- **Step 5 (Validate metadata):** Where possible, confirm the album name and year of publication.

⚠️ Output Constraints:
- Return ONLY a **valid JSON array** of the songs.
- Each object must have: 
  - `"title"` (string),
  - `"artist"` (string),
  - `"album"` (string or null),
  - `"publish year"` (number or null)
- NO extra text, explanation, or formatting outside the JSON.

✅ Example output:
[
  {"title": "Bohemian Rhapsody", "artist": "Queen", "album": "A Night at the Opera", "publish year": 1975},
  {"title": "Someone Like You", "artist": "Adele", "album": "21", "publish year": 2011},
  ...
]"""
        
        user_message = f"""User Request: {user_prompt}

{search_context}
{html_section}

Based on the user's request and the comprehensive search results from multiple optimized queries, 
generate a diverse playlist that aligns with the user's prompt in the specified JSON format."""
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        try:
            response = await self.client.post(OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            songs_data = json.loads(content)
            songs = [Song(title=song["title"], artist=song["artist"], album=song['album'], publish_year=song['publish year']) for song in songs_data]
            return songs
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
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
                Song(title="Stay With Me", artist="Chanyeol & Punch"),
                Song(title="Breath", artist="Lee Hi"),
                Song(title="Spring Day", artist="BTS")
            ],
            "taylor swift": [
                Song(title="All Too Well", artist="Taylor Swift"),
                Song(title="Ronan", artist="Taylor Swift"),
                Song(title="Soon You'll Get Better", artist="Taylor Swift"),
                Song(title="Death by a Thousand Cuts", artist="Taylor Swift"),
                Song(title="Sad Beautiful Tragic", artist="Taylor Swift")
            ],
            "chinese": [
                Song(title="月亮代表我的心", artist="Teresa Teng"),
                Song(title="童话", artist="Guang Liang"),
                Song(title="听海", artist="A-Mei"),
                Song(title="小幸运", artist="Hebe Tien"),
                Song(title="演员", artist="Xue Zhiqian")
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
            Song(title="Watermelon Sugar", artist="Harry Styles"),
            Song(title="Good 4 U", artist="Olivia Rodrigo"),
            Song(title="Levitating", artist="Dua Lipa")
        ]
    
    async def close(self):
        await self.client.aclose()
        self.executor.shutdown(wait=True)

# Initialize service
music_service = MusicRecommendationService()

@music_router.post("/recommend", response_model=MusicResponse)
async def recommend_music(request: MusicPrompt):
    """
    Enhanced music recommendation endpoint
    
    Process:
    1. Generate multiple refined search queries using Mistral AI
    2. Search all queries synchronously using ThreadPoolExecutor for speed
    3. Extract HTML content from search result URLs using BeautifulSoup
    4. Analyze combined results to generate recommendations with OpenAI
    
    Returns JSON formatted song list with title and artist
    """
    try:
        # Step 1: Generate multiple refined search queries with Mistral
        search_queries = await music_service.generate_search_queries_with_mistral(request.prompt)
        print(f"Generated queries: {search_queries}")
        
        # Step 2: Search all queries synchronously for better performance
        search_results, urls = await music_service.search_multiple_queries_sync(search_queries)
        print(f"Found {len(search_results)} results from {len(urls)} URLs")
        
        # Step 2.5: Extract HTML content from search result URLs
        html_content = await music_service.extract_music_info_from_html(urls)
        print(f"Extracted {len(html_content)} characters of HTML content")
        
        # Step 3: Generate song recommendations with OpenAI using all search results and HTML content
        songs = await music_service.generate_songs_with_openai(
            request.prompt, 
            search_results, 
            html_content, 
            search_queries
        )
        
        return MusicResponse(
            songs=songs,
            queries_used=search_queries,
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
            "indie folk songs for rainy days",
            "latest K-pop hits",
            "emotional Chinese ballads",
            "Japanese city pop"
        ]
    }

# Test endpoint to see query generation
@music_router.post("/test-queries")
async def test_query_generation(request: MusicPrompt):
    """Test endpoint to see what queries are generated for a prompt"""
    queries = await music_service.generate_search_queries_with_mistral(request.prompt)
    return {"original_prompt": request.prompt, "generated_queries": queries}

# Example of how to use this router in main app:
"""
from fastapi import FastAPI
from this_router_file import music_router

app = FastAPI(title="Enhanced Music Recommendation API", version="2.0.0")
app.include_router(music_router)

# Required dependencies:
# pip install fastapi uvicorn httpx pydantic beautifulsoup4 requests

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""