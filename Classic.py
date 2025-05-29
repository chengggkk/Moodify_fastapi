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
        import re
        numbers = re.findall(r'\b(\d+)\b', user_prompt)
        
        
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
        
        return detected_language
    
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
            # Fallback: create language-aware variations manually
            return self.create_fallback_queries(user_prompt, detected_language)
    
    def create_fallback_queries(self, prompt: str, detected_language: str) -> List[str]:
        """Create fallback search queries with language awareness"""
        base_query = prompt.lower()
        
        # Language-specific query variations
        if detected_language == 'korean':
            queries = [
                prompt,
                f"{prompt} 한국",
                f"korean {prompt.replace('韓劇', 'k-drama').replace('韓國', 'korean')}"
            ]
        elif detected_language == 'chinese':
            queries = [
                prompt,
                f"{prompt} 中文",
                f"chinese {prompt.replace('中文', 'mandarin')}"
            ]
        elif detected_language == 'japanese':
            queries = [
                prompt,
                f"{prompt} 日本",
                f"japanese {prompt}"
            ]
        else:  # English or default
            queries = [
                prompt,
                f"best {prompt}",
                f"top {prompt} songs"
            ]
        
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
        
        try:
            response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=15)
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
        
        return all_results, unique_urls[:6]  # Reduced to prevent timeout
    
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
            
            # Limit text length to prevent token overflow
            return text[:1500] if len(text) > 1500 else text
            
        except Exception as e:
            print(f"HTML extraction error for {url}: {e}")
            return ""
    
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
        detected_language = self.detect_language_and_count(user_prompt)
        
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
        1. Return EXACTLY {user_prompt} songs if user request in the prompt (not more, not less)
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
Requested Count(if request): (user's prompt:{user_prompt}) songs

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
        
        try:
            response = await self.client.post(OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response
            songs_data = json.loads(content)
            
            # Create Song objects
            songs = []
            for song in songs_data:
                try:
                    song_obj = Song(
                        title=song.get("title", "Unknown Title"),
                        artist=song.get("artist", "Unknown Artist"),
                        album=song.get("album"),
                        publish_year=song.get("publish_year")
                    )
                    songs.append(song_obj)
                except Exception as e:
                    print(f"Error creating song object: {e}, song data: {song}")
                    songs.append(Song(
                        title=str(song.get("title", "Unknown Title")),
                        artist=str(song.get("artist", "Unknown Artist"))
                    ))
            
            return songs 
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {content}")
            return self.create_fallback_songs(user_prompt, detected_language)
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return self.create_fallback_songs(user_prompt, detected_language)
    
    def create_fallback_songs(self, prompt: str, detected_language: str) -> List[Song]:
        """Create language-appropriate fallback songs"""
        fallback_songs = {
            "korean": [
                Song(title="봄날", artist="BTS", album="You Never Walk Alone", publish_year=2017),
                Song(title="You Are My Everything", artist="Gummy", album="도깨비 OST", publish_year=2016),
                Song(title="Stay With Me", artist="찬열 & 펀치", album="도깨비 OST", publish_year=2016),
                Song(title="숨", artist="이하이", album="사이코지만 괜찮아 OST", publish_year=2020),
                Song(title="시간을 거슬러", artist="린", album="달의 연인 - 보보경심 려 OST", publish_year=2016),
                Song(title="나를 잊지마요", artist="크러쉬", album="도깨비 OST", publish_year=2016),
                Song(title="Love Yourself", artist="BTS", album="Love Yourself 承 'Her'", publish_year=2017),
                Song(title="밤편지", artist="아이유", album="Through the Night", publish_year=2017),
                Song(title="DNA", artist="BTS", album="Love Yourself 承 'Her'", publish_year=2017),
                Song(title="첫눈처럼 너에게 가겠다", artist="에일리", album="도깨비 OST", publish_year=2016)
            ],
            "chinese": [
                Song(title="月亮代表我的心", artist="鄧麗君", album="島國之情歌第五集", publish_year=1977),
                Song(title="童话", artist="光良", album="童話", publish_year=2005),
                Song(title="聽海", artist="張惠妹", album="Bad Boy", publish_year=1997),
                Song(title="小幸運", artist="田馥甄", album="我的少女時代 電影原聲帶", publish_year=2015),
                Song(title="演員", artist="薛之謙", album="紳士", publish_year=2015),
                Song(title="匆匆那年", artist="王菲", album="匆匆那年 電影原聲帶", publish_year=2014),
                Song(title="告白氣球", artist="周杰倫", album="周杰倫的床邊故事", publish_year=2016),
                Song(title="晴天", artist="周杰倫", album="葉惠美", publish_year=2003),
                Song(title="夜曲", artist="周杰倫", album="十一月的蕭邦", publish_year=2005),
                Song(title="紅豆", artist="王菲", album="唱遊", publish_year=1998)
            ],
            "japanese": [
                Song(title="桜", artist="コブクロ", album="CALLING", publish_year=2005),
                Song(title="Jupiter", artist="平原綾香", album="Jupiter", publish_year=2003),
                Song(title="恋", artist="星野源", album="恋", publish_year=2016),
                Song(title="津軽海峡冬景色", artist="石川さゆり", album="津軽海峡冬景色", publish_year=1977),
                Song(title="千の風になって", artist="秋川雅史", album="千の風になって", publish_year=2006),
                Song(title="贈る言葉", artist="海援隊", album="贈る言葉", publish_year=1979),
                Song(title="Summer", artist="久石譲", album="菊次郎の夏 サウンドトラック", publish_year=1999),
                Song(title="First Love", artist="宇多田ヒカル", album="First Love", publish_year=1999),
                Song(title="涙そうそう", artist="夏川りみ", album="涙そうそう", publish_year=2001),
                Song(title="HANABI", artist="Mr.Children", album="HOME", publish_year=2007)
            ]
        }
        
        if detected_language in fallback_songs:
            return fallback_songs[detected_language]
        
        # Default English fallback
        return [
            Song(title="Shape of You", artist="Ed Sheeran", album="÷", publish_year=2017),
            Song(title="Blinding Lights", artist="The Weeknd", album="After Hours", publish_year=2020),
            Song(title="Watermelon Sugar", artist="Harry Styles", album="Fine Line", publish_year=2020),
            Song(title="Good 4 U", artist="Olivia Rodrigo", album="SOUR", publish_year=2021),
            Song(title="Levitating", artist="Dua Lipa", album="Future Nostalgia", publish_year=2020),
            Song(title="Anti-Hero", artist="Taylor Swift", album="Midnights", publish_year=2022),
            Song(title="As It Was", artist="Harry Styles", album="Harry's House", publish_year=2022),
            Song(title="Heat Waves", artist="Glass Animals", album="Dreamland", publish_year=2020),
            Song(title="Stay", artist="The Kid LAROI & Justin Bieber", album="F*ck Love 3: Over You", publish_year=2021),
            Song(title="Bad Habit", artist="Steve Lacy", album="Gemini Rights", publish_year=2022)
        ]
    
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
    try:
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
        
    except Exception as e:
        print(f"Error in recommend_music: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

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