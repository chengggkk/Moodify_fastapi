from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict, Any
import asyncio
import json
import random
import os
from openai import AsyncOpenAI

audio_feature_router = APIRouter(prefix="/audio_feature", tags=["audio_feature"])

# Pydantic models for request/response
class AudioFeatureRequest(BaseModel):
    title: str
    artist: str

class AudioFeature(BaseModel):
    title: str
    artist: str
    key: Optional[str] = None
    bpm: Optional[int] = None
    time_signature: Optional[str] = None
    camelot: Optional[str] = None
    energy: Optional[int] = None
    danceability: Optional[int] = None
    happiness: Optional[int] = None
    loudness: Optional[str] = None
    acousticness: Optional[int] = None
    instrumentalness: Optional[int] = None
    liveness: Optional[int] = None
    speechiness: Optional[int] = None
    popularity: Optional[int] = None

class AudioFeatureService:
    def __init__(self, brave_api_key: Optional[str], openai_api_key: Optional[str]):
        self.brave_api_key = brave_api_key
        self.openai_api_key = openai_api_key
        self.brave_search_url = "https://api.search.brave.com/res/v1/web/search"
        self.use_api = brave_api_key is not None and brave_api_key != "YOUR_BRAVE_API_KEY_HERE"
        
        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            print("⚠️  OpenAI API key not provided - will use traditional parsing only")
    
    async def search_tunebat(self, title: str, artist: str) -> str:
        """Search for tunebat results using Brave Search API"""
        print(f"🔍 Starting search for: '{title}' by '{artist}'")
        
        if not self.brave_api_key or self.brave_api_key == "YOUR_BRAVE_API_KEY_HERE":
            print("❌ Brave Search API key not configured")
            raise HTTPException(
                status_code=500, 
                detail="Brave Search API key not configured. Please set a valid API key."
            )
        
        query = f"tunebat {title} {artist}"
        print(f"🔍 Search query: '{query}'")
        
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        
        params = {
            "q": query,
            "count": 1,
            "search_lang": "en",
            "country": "US",
            "safesearch": "off",
            "freshness": "",
            "text_decorations": False,
            "spellcheck": True
        }
        
        async with httpx.AsyncClient() as client:
            try:
                print("📡 Making search request...")
                response = await client.get(
                    self.brave_search_url,
                    headers=headers,
                    params=params,
                    timeout=15.0
                )
                
                print(f"📡 Search response status: {response.status_code}")
                
                # Handle specific error codes
                if response.status_code == 422:
                    print("❌ Search API error 422")
                    error_detail = "Invalid request parameters or API key issue"
                    try:
                        error_json = response.json()
                        if "message" in error_json:
                            error_detail = error_json["message"]
                    except:
                        pass
                    raise HTTPException(
                        status_code=422, 
                        detail=f"Brave Search API error: {error_detail}. Check your API key and parameters."
                    )
                elif response.status_code == 401:
                    print("❌ Search API error 401 - Unauthorized")
                    raise HTTPException(
                        status_code=401, 
                        detail="Unauthorized: Invalid or expired Brave Search API key"
                    )
                elif response.status_code == 429:
                    print("❌ Search API error 429 - Rate limit")
                    raise HTTPException(
                        status_code=429, 
                        detail="Rate limit exceeded. Please try again later."
                    )
                
                response.raise_for_status()
                search_results = response.json()
                
                if not search_results.get("web", {}).get("results"):
                    print("❌ No search results found")
                    raise HTTPException(status_code=404, detail="No search results found")
                
                first_result = search_results["web"]["results"][0]
                url = first_result.get("url", "")
                print(f"✅ Found TuneBat URL: {url}")
                return url
                
            except httpx.HTTPError as e:
                print(f"❌ HTTP error during search: {str(e)}")
                if "422" in str(e):
                    raise HTTPException(
                        status_code=422, 
                        detail="Brave Search API configuration error. Please check your API key and subscription status."
                    )
                raise HTTPException(status_code=500, detail=f"Search API error: {str(e)}")
    
    async def extract_with_openai(self, html_content: str, title: str, artist: str) -> Dict[str, Any]:
        """Use OpenAI to extract audio features from HTML content"""
        print("🤖 Starting OpenAI extraction...")
        
        if not self.openai_client:
            print("⚠️  OpenAI client not available, skipping AI extraction")
            return {}
        
        try:
            # Parse HTML first to extract clean, readable sections
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look specifically for the audio feature sections
            relevant_text_sections = []
            
            # Find Ant Design progress circles and their labels - use flexible approach
            progress_circles = soup.find_all('div', class_=lambda x: x and 'ant-progress-circle' in ' '.join(x) if isinstance(x, list) else 'ant-progress-circle' in str(x))
            
            for circle in progress_circles:
                # Get the value from the title attribute
                progress_text_span = circle.find('span', class_=lambda x: x and 'ant-progress-text' in ' '.join(x) if isinstance(x, list) else 'ant-progress-text' in str(x))
                if progress_text_span:
                    value = progress_text_span.get('title', '').strip()
                    
                    # Find the corresponding label - look for nearby spans with meaningful text
                    parent_containers = []
                    current = circle
                    # Go up the DOM tree to find potential containers
                    for _ in range(5):  # Check up to 5 levels up
                        if current.parent:
                            current = current.parent
                            parent_containers.append(current)
                        else:
                            break
                    
                    label = None
                    for container in parent_containers:
                        # Look for spans that contain feature names
                        spans = container.find_all('span')
                        for span in spans:
                            text = span.get_text().strip().lower()
                            if text in ['popularity', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']:
                                label = text
                                break
                        if label:
                            break
                    
                    if label and value:
                        relevant_text_sections.append(f"{label}: {value}")
            
            # Look for other musical data patterns in clean text
            page_text = soup.get_text()
            
            # Extract sections that mention musical terms
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                if any(term in line.lower() for term in ['bpm', 'key', 'major', 'minor', 'camelot', 'tempo']):
                    if len(line) < 100:  # Skip very long lines
                        relevant_text_sections.append(line)
            
            # Combine relevant sections
            combined_text = '\n'.join(relevant_text_sections)
            
            # If we still don't have good content, look for JSON in script tags
            if len(combined_text) < 50:
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string:
                        script_content = script.string
                        if any(term in script_content.lower() for term in ['bpm', 'key', 'energy', 'danceability']):
                            # Try to extract JSON objects
                            json_matches = re.findall(r'\{[^{}]*(?:"(?:bpm|key|energy|danceability|popularity)"[^{}]*)*\}', script_content)
                            for match in json_matches:
                                relevant_text_sections.append(match)
            
            # Final combined content
            combined_text = '\n'.join(relevant_text_sections)
            
            # Limit to reasonable size
            if len(combined_text) > 4000:
                combined_text = combined_text[:4000]
            
            print(f"📄 Extracted relevant content ({len(combined_text)} chars):")
            print(f"📄 Content preview: {combined_text[:500]}...")
            
            if not combined_text.strip():
                print("❌ No relevant content found for OpenAI processing")
                return {}
            
            prompt = f"""
            Extract audio features for "{title}" by "{artist}" from this content:

            {combined_text}

            Look for these patterns:
            1. "popularity: 86" -> popularity: 86
            2. "energy: 19" -> energy: 19  
            3. "danceability: 41" -> danceability: 41
            4. "happiness: 16" -> happiness: 16
            5. "acousticness: 64" -> acousticness: 64
            6. "instrumentalness: 0" -> instrumentalness: 0
            7. "liveness: 21" -> liveness: 21
            8. "speechiness: 4" -> speechiness: 4
            9. "loudness: -11 dB" -> loudness: "-11 dB"
            10. BPM/tempo numbers
            11. Musical keys (like "C Major", "F# Minor")
            12. Camelot codes (like "8A", "12B")

            Return only valid JSON:
            {{
                "bpm": null,
                "key": null,
                "time_signature": null,
                "camelot": null,
                "energy": null,
                "danceability": null,
                "happiness": null,
                "loudness": null,
                "acousticness": null,
                "instrumentalness": null,
                "liveness": null,
                "speechiness": null,
                "popularity": null
            }}
            """
            
            print("🤖 Sending focused request to OpenAI...")
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data extraction specialist. Extract only the numeric values and return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"🤖 OpenAI response: {ai_response}")
            
            # Parse JSON response
            try:
                if "```json" in ai_response:
                    ai_response = ai_response.split("```json")[1].split("```")[0]
                elif "```" in ai_response:
                    ai_response = ai_response.split("```")[1].split("```")[0]
                
                features = json.loads(ai_response)
                
                # Clean the extracted data
                cleaned_features = {}
                for key, value in features.items():
                    if value is not None and value != "null" and value != "":
                        cleaned_features[key] = value
                
                print(f"✅ OpenAI extracted and cleaned: {cleaned_features}")
                return cleaned_features
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse OpenAI JSON: {e}")
                return {}
                
        except Exception as e:
            print(f"❌ OpenAI extraction error: {str(e)}")
            return {}

    async def traditional_parsing(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Enhanced traditional parsing for Ant Design components"""
        print("🔧 Starting enhanced traditional parsing...")
        features = {}
        
        # Method 1: Parse Ant Design progress circles with flexible class matching
        print("🔧 Looking for Ant Design progress circles...")
        
        # Use flexible class matching since CSS classes can be dynamic
        progress_circles = soup.find_all('div', class_=lambda x: x and any('ant-progress-circle' in cls for cls in x) if isinstance(x, list) else 'ant-progress-circle' in str(x))
        
        # Also try alternative selectors in case the above doesn't work
        if not progress_circles:
            # Try finding by the SVG element which is more stable
            svg_elements = soup.find_all('svg', class_=lambda x: x and any('ant-progress-circle' in cls for cls in x) if isinstance(x, list) else 'ant-progress-circle' in str(x))
            progress_circles = [svg.find_parent('div') for svg in svg_elements if svg.find_parent('div')]
        
        print(f"🔧 Found {len(progress_circles)} progress circles")
        
        if progress_circles:
            for i, circle in enumerate(progress_circles):
                print(f"🔧 Processing progress circle {i+1}")
                
                # Get the value from the span with title attribute - use flexible class matching
                progress_text = circle.find('span', class_=lambda x: x and any('ant-progress-text' in cls for cls in x) if isinstance(x, list) else 'ant-progress-text' in str(x))
                if progress_text:
                    value_text = progress_text.get('title', '').strip()
                    print(f"🔧 Found value: '{value_text}'")
                    
                    # Find the corresponding label - look for any span that contains text after the progress circle
                    # Method 1: Try to find parent container and look for any span with text
                    parent_container = circle.find_parent('div')
                    label_element = None
                    
                    # Look in the immediate parent and siblings for a span with text content
                    if parent_container:
                        # Try different approaches to find the label
                        possible_labels = []
                        
                        # Approach 1: Look for spans that contain 'ant-typography' in their class
                        typography_spans = parent_container.find_all('span', class_=lambda x: x and 'ant-typography' in ' '.join(x) if isinstance(x, list) else 'ant-typography' in str(x))
                        possible_labels.extend(typography_spans)
                        
                        # Approach 2: Look for any span that comes after the progress circle and has meaningful text
                        all_spans = parent_container.find_all('span')
                        for span in all_spans:
                            text = span.get_text().strip().lower()
                            if text and text in ['popularity', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']:
                                possible_labels.append(span)
                        
                        # Approach 3: Look for spans with specific classes that might contain the label
                        for class_pattern in ['fd89q', 'label', 'text']:
                            class_spans = parent_container.find_all('span', class_=lambda x: x and class_pattern in ' '.join(x) if isinstance(x, list) else class_pattern in str(x))
                            possible_labels.extend(class_spans)
                        
                        # Take the first valid label we find
                        for possible_label in possible_labels:
                            text = possible_label.get_text().strip().lower()
                            if text and text in ['popularity', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']:
                                label_element = possible_label
                                break
                    
                    if label_element:
                            label = label_element.get_text().strip().lower()
                            print(f"🔧 Found label: '{label}'")
                            
                            # Map the label to our feature and extract the value
                            if label == 'popularity' and value_text.isdigit():
                                features['popularity'] = int(value_text)
                                print(f"✅ Extracted popularity: {features['popularity']}")
                            elif label == 'energy' and value_text.isdigit():
                                features['energy'] = int(value_text)
                                print(f"✅ Extracted energy: {features['energy']}")
                            elif label == 'danceability' and value_text.isdigit():
                                features['danceability'] = int(value_text)
                                print(f"✅ Extracted danceability: {features['danceability']}")
                            elif label == 'happiness' and value_text.isdigit():
                                features['happiness'] = int(value_text)
                                print(f"✅ Extracted happiness: {features['happiness']}")
                            elif label == 'acousticness' and value_text.isdigit():
                                features['acousticness'] = int(value_text)
                                print(f"✅ Extracted acousticness: {features['acousticness']}")
                            elif label == 'instrumentalness' and value_text.isdigit():
                                features['instrumentalness'] = int(value_text)
                                print(f"✅ Extracted instrumentalness: {features['instrumentalness']}")
                            elif label == 'liveness' and value_text.isdigit():
                                features['liveness'] = int(value_text)
                                print(f"✅ Extracted liveness: {features['liveness']}")
                            elif label == 'speechiness' and value_text.isdigit():
                                features['speechiness'] = int(value_text)
                                print(f"✅ Extracted speechiness: {features['speechiness']}")
                            elif label == 'loudness' and ('db' in value_text.lower() or 'dB' in value_text):
                                features['loudness'] = value_text
                                print(f"✅ Extracted loudness: {features['loudness']}")
                            else:
                                print(f"⚠️  Unrecognized label-value pair: '{label}' = '{value_text}'")

        
        # Additional Method: If no progress circles found or they didn't work, try alternative approaches
        if not features and not progress_circles:
            print("🔧 No progress circles found, trying alternative approaches...")
            
            # Look for any elements that might contain the audio feature data
            # Method A: Look for divs that contain both a number and a label
            all_divs = soup.find_all('div')
            for div in all_divs:
                text = div.get_text()
                # Look for patterns like "86 popularity" or "popularity 86"
                for feature_name in ['popularity', 'energy', 'danceability', 'happiness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'loudness']:
                    patterns = [
                        rf'(\d+)\s+{feature_name}',
                        rf'{feature_name}\s+(\d+)',
                        rf'{feature_name}[:\s]+(\d+)',
                        rf'(\d+)[:\s]+{feature_name}'
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match and not features.get(feature_name):
                            if feature_name == 'loudness':
                                # Special handling for loudness (might include dB)
                                loudness_match = re.search(rf'{feature_name}[:\s]+(-?\d+(?:\.\d+)?\s*dB)', text, re.IGNORECASE)
                                if loudness_match:
                                    features[feature_name] = loudness_match.group(1)
                                    print(f"✅ Extracted {feature_name} from div: {features[feature_name]}")
                            else:
                                features[feature_name] = int(match.group(1))
                                print(f"✅ Extracted {feature_name} from div: {features[feature_name]}")
                            break
        
        # Method 2: Look for other patterns in page text (BPM, key, etc.)
        print("🔧 Looking for additional musical data patterns...")
        page_text = soup.get_text()
        
        # BPM patterns
        if not features.get('bpm'):
            print("🔧 Searching for BPM...")
            bpm_patterns = [
                r'(\d+)\s*·\s*BPM',
                r'BPM\s*·\s*(\d+)', 
                r'(\d+)\s*BPM',
                r'BPM[:\s]*(\d+)',
                r'Tempo[:\s]*(\d+)',
                r'(\d+)\s*beats per minute'
            ]
            
            for pattern in bpm_patterns:
                bpm_match = re.search(pattern, page_text, re.IGNORECASE)
                if bpm_match:
                    features['bpm'] = int(bpm_match.group(1))
                    print(f"✅ Extracted BPM: {features['bpm']}")
                    break
        
        # Key patterns
        if not features.get('key'):
            print("🔧 Searching for musical key...")
            key_patterns = [
                r'([A-G][#♯♭b]?\s*(?:Major|Minor))\s*·\s*Key',
                r'Key\s*·\s*([A-G][#♯♭b]?\s*(?:Major|Minor))',
                r'Key[:\s]*([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)',
            ]
            
            for pattern in key_patterns:
                key_match = re.search(pattern, page_text, re.IGNORECASE)
                if key_match:
                    features['key'] = key_match.group(1).strip()
                    print(f"✅ Extracted key: {features['key']}")
                    break
        
        # Camelot patterns
        if not features.get('camelot'):
            print("🔧 Searching for Camelot key...")
            camelot_patterns = [
                r'(\d+[AB])\s*·\s*Camelot',
                r'Camelot\s*·\s*(\d+[AB])',
                r'Camelot[:\s]*(\d+[AB])',
            ]
            
            for pattern in camelot_patterns:
                camelot_match = re.search(pattern, page_text, re.IGNORECASE)
                if camelot_match:
                    features['camelot'] = camelot_match.group(1)
                    print(f"✅ Extracted Camelot: {features['camelot']}")
                    break
        
        # Time signature
        if not features.get('time_signature'):
            print("🔧 Searching for time signature...")
            time_patterns = [
                r'Time Signature[:\s]*(\d+/\d+)',
                r'(\d+/\d+)\s*time',
                r'Time[:\s]*(\d+/\d+)'
            ]
            
            for pattern in time_patterns:
                time_match = re.search(pattern, page_text, re.IGNORECASE)
                if time_match:
                    features['time_signature'] = time_match.group(1)
                    print(f"✅ Extracted time signature: {features['time_signature']}")
                    break
        
        # Method 3: Look for structured data in script tags
        if len(features) < 3:  # If we haven't found much, try script tags
            print("🔧 Searching script tags for JSON data...")
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string:
                    try:
                        # Look for JSON that might contain our data
                        if any(term in script.string.lower() for term in ['bpm', 'energy', 'danceability']):
                            print(f"🔧 Found potentially relevant script content")
                            # Try to extract JSON objects
                            json_pattern = r'\{[^{}]*(?:"(?:bpm|key|energy|danceability|popularity)"[^{}]*)+[^{}]*\}'
                            json_matches = re.findall(json_pattern, script.string)
                            for match in json_matches:
                                try:
                                    data = json.loads(match)
                                    if isinstance(data, dict):
                                        # Extract relevant fields
                                        for key in ['bpm', 'key', 'energy', 'danceability', 'popularity', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                                            if key in data and not features.get(key):
                                                features[key] = data[key]
                                                print(f"✅ Extracted {key} from JSON: {features[key]}")
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        continue
        
        print(f"🔧 Traditional parsing complete. Final features: {features}")
        return features
    
    async def scrape_tunebat_page(self, url: str, title: str = "", artist: str = "") -> Dict[str, Any]:
        """Scrape audio features from tunebat page with OpenAI assistance"""
        print(f"🌐 Starting to scrape URL: {url}")
        
        if not url:
            print("❌ No URL provided")
            raise HTTPException(status_code=404, detail="No URL found in search results")
        
        # Multiple user agents to rotate
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        selected_ua = random.choice(user_agents)
        print(f"🌐 Using User-Agent: {selected_ua[:50]}...")
        
        # Headers to mimic a real browser request
        headers = {
            'User-Agent': selected_ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/'
        }
        
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            timeout=20.0
        ) as client:
            try:
                # Add random delay to appear more human-like
                delay = random.uniform(1, 3)
                print(f"⏱️  Waiting {delay:.2f} seconds before request...")
                await asyncio.sleep(delay)
                
                print("🌐 Making HTTP request...")
                response = await client.get(url)
                
                print(f"🌐 Response status: {response.status_code}")
                print(f"🌐 Response headers: {dict(response.headers)}")
                
                if response.status_code == 403:
                    print("❌ Access denied (403)")
                    raise HTTPException(
                        status_code=403, 
                        detail="Access denied by tunebat. Try using a VPN or consider alternative data sources like Spotify Web API."
                    )
                
                response.raise_for_status()
                
                html_content = response.text
                print(f"📄 HTML content received: {len(html_content)} characters")
                
                print("🔍 Parsing HTML with BeautifulSoup...")
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Log the page title for verification
                page_title = soup.find('title')
                if page_title:
                    print(f"📄 Page title: {page_title.get_text().strip()}")
                
                # Method 1: Use OpenAI for smart extraction
                print("\n=== METHOD 1: OpenAI Extraction ===")
                openai_features = await self.extract_with_openai(html_content, title, artist)
                
                # Method 2: Traditional parsing as backup
                print("\n=== METHOD 2: Traditional Parsing ===")
                traditional_features = await self.traditional_parsing(soup)
                
                # Combine results, prioritizing OpenAI but filling gaps with traditional parsing
                print("\n=== COMBINING RESULTS ===")
                combined_features = {}
                
                # Start with traditional features as base
                combined_features.update(traditional_features)
                print(f"🔧 Traditional features: {traditional_features}")
                
                # Override/supplement with OpenAI features
                combined_features.update(openai_features)
                print(f"🤖 OpenAI features: {openai_features}")
                print(f"🎯 Final combined features: {combined_features}")
                
                return combined_features
                
            except httpx.HTTPError as e:
                print(f"❌ HTTP error during scraping: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to scrape page: {str(e)}")
    
    async def traditional_parsing(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Traditional BeautifulSoup parsing as backup"""
        print("🔧 Starting traditional parsing...")
        features = {}
        
        # Method 1: Look for Ant Design progress circles
        print("🔧 Looking for Ant Design progress circles...")
        progress_circles = soup.find_all('div', class_='ant-progress-circle')
        print(f"🔧 Found {len(progress_circles)} progress circles")
        
        for i, circle in enumerate(progress_circles):
            print(f"🔧 Processing progress circle {i+1}")
            
            # Find the value in the progress text span
            progress_text = circle.find('span', class_='ant-progress-text')
            if progress_text:
                value_text = progress_text.get('title', '').strip()
                print(f"🔧 Progress value: '{value_text}'")
                
                # Find the corresponding label
                label_element = circle.find_next('span', class_='ant-typography')
                if label_element:
                    label = label_element.get_text().strip().lower()
                    print(f"🔧 Progress label: '{label}'")
                    
                    # Parse the value based on the label
                    if label == 'popularity' and value_text.isdigit():
                        features['popularity'] = int(value_text)
                        print(f"✅ Extracted popularity: {features['popularity']}")
                    elif label == 'energy' and value_text.isdigit():
                        features['energy'] = int(value_text)
                        print(f"✅ Extracted energy: {features['energy']}")
                    elif label == 'danceability' and value_text.isdigit():
                        features['danceability'] = int(value_text)
                        print(f"✅ Extracted danceability: {features['danceability']}")
                    elif label == 'happiness' and value_text.isdigit():
                        features['happiness'] = int(value_text)
                        print(f"✅ Extracted happiness: {features['happiness']}")
                    elif label == 'acousticness' and value_text.isdigit():
                        features['acousticness'] = int(value_text)
                        print(f"✅ Extracted acousticness: {features['acousticness']}")
                    elif label == 'instrumentalness' and value_text.isdigit():
                        features['instrumentalness'] = int(value_text)
                        print(f"✅ Extracted instrumentalness: {features['instrumentalness']}")
                    elif label == 'liveness' and value_text.isdigit():
                        features['liveness'] = int(value_text)
                        print(f"✅ Extracted liveness: {features['liveness']}")
                    elif label == 'speechiness' and value_text.isdigit():
                        features['speechiness'] = int(value_text)
                        print(f"✅ Extracted speechiness: {features['speechiness']}")
                    elif label == 'loudness' and 'db' in value_text.lower():
                        features['loudness'] = value_text
                        print(f"✅ Extracted loudness: {features['loudness']}")
        
        # Method 2: Look for standard text patterns
        print("🔧 Looking for text patterns...")
        page_text = soup.get_text()
        
        # BPM patterns
        print("🔧 Searching for BPM...")
        bpm_patterns = [
            r'(\d+)\s*BPM',
            r'BPM[:\s]*(\d+)',
            r'Tempo[:\s]*(\d+)',
            r'(\d+)\s*beats per minute'
        ]
        
        for pattern in bpm_patterns:
            bpm_match = re.search(pattern, page_text, re.IGNORECASE)
            if bpm_match and not features.get('bpm'):
                features['bpm'] = int(bpm_match.group(1))
                print(f"✅ Extracted BPM: {features['bpm']}")
                break
        
        # Key patterns
        print("🔧 Searching for musical key...")
        key_patterns = [
            r'Key[:\s]*([A-G][#♯♭b]?(?:\s*(?:major|minor|maj|min))?)',
            r'([A-G][#♯♭b]?)\s*(?:major|minor|maj|min)',
            r'Key:\s*([A-G][#♯♭b]?)'
        ]
        
        for pattern in key_patterns:
            key_match = re.search(pattern, page_text, re.IGNORECASE)
            if key_match and not features.get('key'):
                features['key'] = key_match.group(1).strip()
                print(f"✅ Extracted key: {features['key']}")
                break
        
        # Time Signature patterns
        print("🔧 Searching for time signature...")
        time_sig_patterns = [
            r'Time Signature[:\s]*(\d+/\d+)',
            r'(\d+/\d+)\s*time',
            r'Time[:\s]*(\d+/\d+)'
        ]
        
        for pattern in time_sig_patterns:
            time_match = re.search(pattern, page_text, re.IGNORECASE)
            if time_match and not features.get('time_signature'):
                features['time_signature'] = time_match.group(1)
                print(f"✅ Extracted time signature: {features['time_signature']}")
                break
        
        # Camelot Key patterns
        print("🔧 Searching for Camelot key...")
        camelot_patterns = [
            r'Camelot[:\s]*(\d+[AB])',
            r'(\d+[AB])\s*Camelot',
            r'Camelot Key[:\s]*(\d+[AB])'
        ]
        
        for pattern in camelot_patterns:
            camelot_match = re.search(pattern, page_text, re.IGNORECASE)
            if camelot_match and not features.get('camelot'):
                features['camelot'] = camelot_match.group(1)
                print(f"✅ Extracted Camelot: {features['camelot']}")
                break
        
        print(f"🔧 Traditional parsing complete. Found: {features}")
        return features
    
    async def get_audio_features(self, title: str, artist: str) -> AudioFeature:
        """Main method to get audio features with fallback strategies"""
        print(f"\n🎵 ==> STARTING AUDIO FEATURE EXTRACTION <==")
        print(f"🎵 Song: '{title}' by '{artist}'")
        
        features = {}
        
        if self.use_api:
            print("📡 Using Brave API search...")
            try:
                # Primary strategy: Search tunebat via Brave API
                url = await self.search_tunebat(title, artist)
                features = await self.scrape_tunebat_page(url, title, artist)
            except HTTPException as e:
                print(f"❌ API search failed: {e.detail}")
                if e.status_code in [422, 401, 500]:
                    print("🔄 Falling back to direct approach...")
                    # Fallback to non-API methods
                    features = await self.try_direct_tunebat_approach(title, artist)
                else:
                    raise e
        else:
            print("🔄 Skipping API, using direct approach...")
            # Skip API and go directly to fallback methods
            features = await self.try_direct_tunebat_approach(title, artist)
        
        print(f"\n🎯 ==> FINAL RESULT <==")
        result = AudioFeature(
            title=title,
            artist=artist,
            **features
        )
        print(f"🎯 Returning: {result.dict()}")
        return result
    
    async def try_direct_tunebat_approach(self, title: str, artist: str) -> Dict[str, Any]:
        """Try to construct tunebat URL directly or use DuckDuckGo search"""
        print("🔧 Trying direct TuneBat approach...")
        
        # Clean the title and artist for URL construction
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        clean_artist = re.sub(r'[^\w\s-]', '', artist).strip()
        
        print(f"🔧 Cleaned title: '{clean_title}'")
        print(f"🔧 Cleaned artist: '{clean_artist}'")
        
        # Try common tunebat URL patterns
        possible_urls = [
            f"https://tunebat.com/Info/{clean_title.replace(' ', '-')}-{clean_artist.replace(' ', '-')}",
            f"https://tunebat.com/Info/{clean_artist.replace(' ', '-')}-{clean_title.replace(' ', '-')}",
        ]
        
        for i, url in enumerate(possible_urls):
            print(f"🔧 Trying direct URL {i+1}: {url}")
            try:
                features = await self.scrape_tunebat_page(url, title, artist)
                if features:  # If we got any features, return them
                    print(f"✅ Direct URL {i+1} worked!")
                    return features
            except Exception as e:
                print(f"❌ Direct URL {i+1} failed: {str(e)}")
                continue
        
        # If direct URLs don't work, try DuckDuckGo search (no API key required)
        print("🔄 Falling back to DuckDuckGo search...")
        return await self.search_with_duckduckgo(title, artist)
    
    async def search_with_duckduckgo(self, title: str, artist: str) -> Dict[str, Any]:
        """Use DuckDuckGo search as fallback (no API key required)"""
        print("🦆 Starting DuckDuckGo search...")
        query = f"site:tunebat.com {title} {artist}"
        print(f"🦆 DuckDuckGo query: '{query}'")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            try:
                print("🦆 Making DuckDuckGo API request...")
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": "1",
                        "skip_disambig": "1"
                    }
                )
                
                print(f"🦆 DuckDuckGo response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("Results", [])
                    print(f"🦆 Found {len(results)} DuckDuckGo results")
                    
                    # Look for tunebat URLs in results
                    for i, result in enumerate(results):
                        url = result.get("FirstURL", "")
                        print(f"🦆 Result {i+1} URL: {url}")
                        if "tunebat.com" in url:
                            print(f"✅ Found TuneBat URL in result {i+1}")
                            try:
                                return await self.scrape_tunebat_page(url, title, artist)
                            except Exception as e:
                                print(f"❌ Failed to scrape DuckDuckGo result {i+1}: {str(e)}")
                                continue
                
                print("❌ No usable TuneBat URLs found in DuckDuckGo results")
                return {}
                
            except Exception as e:
                print(f"❌ DuckDuckGo search failed: {str(e)}")
                return {}

# Initialize the service with API keys
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
audio_service = AudioFeatureService(BRAVE_API_KEY, OPENAI_API_KEY)

@audio_feature_router.get("/search", response_model=AudioFeature)
async def get_audio_features(
    title: str = Query(..., description="Song title"),
    artist: str = Query(..., description="Artist name")
):
    """
    Get audio features for a song by searching tunebat
    
    - **title**: The song title
    - **artist**: The artist name
    """
    try:
        audio_features = await audio_service.get_audio_features(title, artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@audio_feature_router.post("/search", response_model=AudioFeature)
async def get_audio_features_post(request: AudioFeatureRequest):
    """
    Get audio features for a song by searching tunebat (POST method)
    """
    try:
        audio_features = await audio_service.get_audio_features(request.title, request.artist)
        return audio_features
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in POST endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Health check endpoint
@audio_feature_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "audio_feature_router"}