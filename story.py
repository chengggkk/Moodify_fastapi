import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from embedding import preprocess_long_text

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

story_router = APIRouter(prefix="/story", tags=["story-generation"])

# Request model
class StoryRequest(BaseModel):
    lyrics: str
    customize: Optional[str] = None  # Can be a string or null

# Response model
class StoryResponse(BaseModel):
    story: str

# Prompt template
def generate_prompt(lyrics: str, customize: Optional[str] = None) -> str:
    instruction = ("""
CREATIVE ANALYSIS FRAMEWORK - Think step by step:

Step 1: EMOTIONAL ARCHAEOLOGY
- Dig beneath surface emotions to find hidden feelings
- Identify emotional contradictions and complexities
- Map the complete emotional journey from start to finish

Step 2: THEMATIC TREASURE HUNTING  
- Discover universal human experiences embedded in the lyrics
- Find metaphorical gold mines and symbolic meanings
- Identify archetypal patterns and mythic elements

Step 3: NARRATIVE DNA EXTRACTION
- Extract story seeds: potential characters, conflicts, settings
- Identify dramatic tensions and unresolved questions
- Find the beating heart of human drama within the lyrics

Step 4: CREATIVE CATALYST IDENTIFICATION
- Spot elements that spark imagination and wonder
- Find unique angles that haven't been explored before
- Identify emotional triggers that create story momentum

Please provide your creative analysis in English, focusing on unlocking maximum storytelling potential and creative inspiration.
                   """
    )
    if customize:
        instruction += f" Try to incorporate this idea or theme: '{customize}'."
    instruction += "\n\nLyrics:\n" + lyrics + "\n\nStory:"
    return instruction

# Main generation function
async def generate_story_with_openai(lyrics: str, customize: Optional[str] = None) -> str:
    prompt = generate_prompt(lyrics, customize)

    try:
        response = await run_in_threadpool(
            openai.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a poetic storyteller AI that turns lyrics into stories."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from OpenAI: {str(e)}")

@story_router.post("/", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    if not request.lyrics:
        raise HTTPException(status_code=400, detail="Lyrics must be provided.")

    preprocessed_lyrics = preprocess_long_text(request.lyrics)
    story = await generate_story_with_openai(preprocessed_lyrics, request.customize)
    return StoryResponse(story=story)