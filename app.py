from fastapi import FastAPI
from embedding import embedding_router
from mood import mood_router
from story import story_router
from donut import donut_router
from Classic import music_router
from Lyrics import lyrics_router

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Text Analysis API",
    description="API for text embedding, mood analysis, story generation, and playlist recognition",
    version="2.0.0"
)

# Include all routers
app.include_router(embedding_router)
app.include_router(mood_router)
app.include_router(story_router)
app.include_router(donut_router)
app.include_router(music_router)
app.include_router(lyrics_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)