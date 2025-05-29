from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
    album: Optional[str] = None
    publish_year: Optional[int] = None

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
    
    def detect_language_and_count(self, user_prompt: str) -> tuple[str, int]:
        """Detect language preference and song count from user prompt"""
        prompt_lower = user_prompt.lower()
                
        # Look for specific numbers
        numbers = re.findall(r'\b(\d+)\b', user_prompt)
        song_count = int(numbers[0]) if numbers else 10  # default count
        
        # Detect language preference
        language_indicators = {
            'korean': ['k-drama', 'kdrama', '韓劇', '한드라마', 'korean', 'k-pop', 'kpop', '韓國', '한국'],
            'chinese': ['c-pop', 'cpop', 'chinese', '中文', '華語', '中國', '台灣', 'mandarin', 'cantonese'],
            'japanese': ['j-pop', 'jpop', 'japanese', '日本', '日語', 'anime', 'city pop'],
            'english': ['english', 'american', 'british', 'uk', 'us']
        }
        
        detected_language = 'english'  # default
        for lang, indicators in language_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                detected_language = lang
                break
        
        return detected_language, song_count
    
    async def generate_search_queries_with_mistral(self, user_prompt: str) -> List[str]:
        """Step 1: Generate multiple refined search queries using Mistral AI with language detection"""
        detected_language, song_count = self.detect_language_and_count(user_prompt)
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Enhanced system prompt with language-specific instructions
        system_content = f"""You are a music search query generator. Given a user's music request, 
        generate 3 different optimized search queries that would help find relevant songs.
        
        IMPORTANT: The user appears to prefer {detected_language} content and wants approximately {song_count} songs.
        
        Language-specific search strategies:
        - Korean: Use both English and Korean terms (한국어). Include "K-drama OST", "K-pop", "한국 음악"
        - Chinese: Use both English and Chinese terms (中文). Include "華語歌曲", "中文歌", "台灣音樂"  
        - Japanese: Use both English and Japanese terms (日本語). Include "J-pop", "日本音楽", "アニメソング"
        - English: Focus on English terms with genre and temporal specifications
        
        For each query, consider:
        1. Native language terms when applicable
        2. Genre-specific keywords  
        3. Temporal/trending variations ("2024", "latest", "popular")
        
        Return ONLY a JSON array of 3 search query strings, nothing else.
        
        Examples:
        Input: "recommend 10 taylor swift popular songs"
        Output: ["taylor swift most popular songs", "taylor swift greatest hits top songs", "taylor swift best songs all time"]
        
        Input: "韓劇浪漫OST推薦"
        Output: ["韓劇 로맨틱 OST 추천", "korean drama romantic soundtrack", "K-drama love songs OST"]
        
        Input: "中文流行歌曲"
        Output: ["中文流行歌曲 華語", "chinese pop songs mandarin", "華語流行音樂 2024"]
        """
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": f"Generate 3 search queries for: {user_prompt}"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.4
        }
        
        response = await self.client.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        queries_text = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        queries = json.loads(queries_text)
        if not isinstance(queries, list) or len(queries) != 3:
            raise ValueError("Invalid query response format")
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
            "count": 8,  # Balanced count
            "safesearch": "moderate",
            "freshness": "pw"  # Past week for more recent results
        }
        
        response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        search_results = response.json()
        results = search_results.get("web", {}).get("results", [])
        
        # Extract URLs for HTML parsing
        urls = [result.get("url", "") for result in results if result.get("url")]
        
        return results, urls
    
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
        
        return all_results, unique_urls[:6]  # Reduced to prevent timeout
    
    async def extract_html_content(self, url: str) -> str:
        """Extract and clean HTML content using BeautifulSoup"""
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
        
        # Limit text length to prevent token overflow
        return text[:1500] if len(text) > 1500 else text
    
    async def extract_music_info_from_html(self, urls: List[str]) -> str:
        """Extract music-related information from multiple URLs"""
        # Process fewer URLs to prevent timeout
        extraction_tasks = [
            self.extract_html_content(url) 
            for url in urls[:4]  # Reduced from 8 to 4
        ]
        
        contents = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        extracted_content = []
        music_keywords = ['song', 'artist', 'album', 'music', 'track', 'singer', 'band', 'OST', 'soundtrack', 
                         'lyrics', 'discography', 'release', 'spotify', 'apple music', 'playlist', 'chart',
                         '歌曲', '歌手', '專輯', '音樂', '歌詞', '원곡', '가수', '앨범', '음악']
        
        for content in contents:
            if isinstance(content, str) and content:
                content_lower = content.lower()
                if any(keyword in content_lower for keyword in music_keywords):
                    extracted_content.append(content)
        
        combined_content = "\n\n".join(extracted_content)
        # Limit total content to prevent token overflow
        return combined_content[:2500] if len(combined_content) > 2500 else combined_content
    
    async def generate_songs_with_openai(self, user_prompt: str, search_results: List[Dict], html_content: str, queries_used: List[str]) -> List[Song]:
        """Step 3: Generate song recommendations with language and count awareness"""
        detected_language, song_count = self.detect_language_and_count(user_prompt)
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare search context - limit to prevent token overflow
        search_context = f"Search queries used: {', '.join(queries_used)}\n\n"
        for i, result in enumerate(search_results[:10]):  # Reduced to 10 results
            search_context += f"Result {i+1}:\nTitle: {result.get('title', '')}\nSnippet: {result.get('description', '')}\n\n"
        
        # Add HTML content if available
        html_section = ""
        if html_content:
            html_section = f"\nExtracted Website Content:\n{html_content}\n"
        
        # Enhanced system prompt with language preservation
        system_prompt = f"""You are a music recommendation expert. Based on the user's request, search results, and website content, recommend exact songs.

        CRITICAL REQUIREMENTS:
        1. Return EXACTLY {song_count} songs (not more, not less)
        2. Primary language preference: {detected_language}
        3. Keep original song titles and artist names in their native language/script
        4. For Korean songs: Use 한글 (Hangul) for Korean titles/artists
        5. For Chinese songs: Use 中文 (Chinese characters) for Chinese titles/artists  
        6. For Japanese songs: Use 日本語 (Japanese) for Japanese titles/artists
        7. Only include English translations in parentheses if specifically requested
        
        Language-specific instructions:
        - Korean: Preserve 한글 titles like "봄날", "You Are My Everything" 
        - Chinese: Preserve 中文 titles like "月亮代表我的心", "聽海"
        - Japanese: Preserve titles like "桜", "Jupiter"
        - English: Use standard English titles
        
        Return ONLY a valid JSON array containing 'title' and 'artist' fields (required), 
        and optionally 'album' and 'publish_year' fields. No additional text or formatting.
        
        Example format for Korean:
        [
            {{"title": "봄날", "artist": "BTS", "album": "You Never Walk Alone", "publish_year": 2017}},
            {{"title": "You Are My Everything", "artist": "Gummy", "album": "Goblin OST", "publish_year": 2016}}
        ]"""
        
        user_message = f"""User Request: {user_prompt}

Language Preference: {detected_language}
Requested Count: {song_count} songs

{search_context}
{html_section}

Generate original language titles and artist names."""
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 2000,  # Increased for more songs
            "temperature": 0.6
        }
        
        response = await self.client.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        songs_data = json.loads(content)
        
        # Create Song objects
        songs = []
        for song in songs_data:
            song_obj = Song(
                title=song.get("title", "Unknown Title"),
                artist=song.get("artist", "Unknown Artist"),
                album=song.get("album"),
                publish_year=song.get("publish_year")
            )
            songs.append(song_obj)
        
        return songs
    
    async def close(self):
        await self.client.aclose()
        self.executor.shutdown(wait=True)

# Initialize service
music_service = MusicRecommendationService()

@music_router.post("/recommend", response_model=MusicResponse)
async def recommend_music(request: MusicPrompt):
    """
    Enhanced music recommendation endpoint with language detection and count limits
    
    Process:
    1. Detect language preference and song count from user prompt
    2. Generate multiple refined search queries using Mistral AI with language awareness
    3. Search all queries with optimized performance 
    4. Extract HTML content from search result URLs
    5. Generate exact number of recommendations with language preservation using OpenAI
    
    Returns JSON formatted song list with exact count and original language titles
    """
    # Step 0: Detect language and count preferences
    detected_language, song_count = music_service.detect_language_and_count(request.prompt)
    print(f"Detected language: {detected_language}, requested count: {song_count}")
    
    # Step 1: Generate language-aware search queries with Mistral
    search_queries = await music_service.generate_search_queries_with_mistral(request.prompt)
    print(f"Generated queries: {search_queries}")
    
    # Step 2: Search all queries with timeout protection
    search_results, urls = await music_service.search_multiple_queries_sync(search_queries)
    print(f"Found {len(search_results)} results from {len(urls)} URLs")
    
    # Step 2.5: Extract HTML content with limits to prevent timeouts
    html_content = await music_service.extract_music_info_from_html(urls)
    print(f"Extracted {len(html_content)} characters of HTML content")
    
    # Step 3: Generate exact count of recommendations with language preservation
    songs = await music_service.generate_songs_with_openai(
        request.prompt, 
        search_results, 
        html_content, 
        search_queries
    )
    
    print(f"Generated {len(songs)} songs")
    
    return MusicResponse(
        songs=songs,
        queries_used=search_queries,
        timestamp=datetime.now().isoformat()
    )

@music_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@music_router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await music_service.close()

# Enhanced example endpoints
@music_router.get("/examples")
async def get_examples():
    """Get example prompts for testing with language awareness"""
    return {
        "examples": [
            "recommend 10 taylor swift popular songs",
            "give me 5 K-drama romantic OST",
            "推薦15首中文流行歌曲", 
            "韓劇浪漫OST推薦8首",
            "20 upbeat workout songs",
            "5 jazz classics from the 1960s",
            "indie folk songs for rainy days",
            "latest 12 K-pop hits",
            "10 emotional Chinese ballads",
            "Japanese city pop 推薦7首"
        ]
    }

# Test endpoint to see language detection and query generation
@music_router.post("/test-queries")
async def test_query_generation(request: MusicPrompt):
    """Test endpoint to see language detection and query generation"""
    detected_language, song_count = music_service.detect_language_and_count(request.prompt)
    queries = await music_service.generate_search_queries_with_mistral(request.prompt)
    return {
        "original_prompt": request.prompt,
        "detected_language": detected_language,
        "requested_song_count": song_count,
        "generated_queries": queries
    }