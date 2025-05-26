import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModel
import openai

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

def fetch_lyrics(song: str, artist: str) -> str:
    query = f"{song} {artist} lyrics"
    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    }

    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    # 抓出所有搜尋結果的連結
    search_results = soup.select("a")
    links = []
    for a_tag in search_results:
        href = a_tag.get("href", "")
        if "/url?q=" in href:
            url = href.split("/url?q=")[1].split("&")[0]
            if "lyrics" in url or "lyric" in url or "歌词" in url:
                links.append(url)

    # 測試前幾個可能的歌詞頁面
    for link in links[:3]:
        print("Trying:", link)
        lyrics = scrape_lyrics(link)
        if lyrics and len(lyrics.split()) > 30:
            return lyrics

    return "Lyrics not found."

def scrape_lyrics(url: str) -> str:
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        # 篩選長度超過30的文字塊合併為歌詞
        lyrics = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 30)
        return lyrics.strip()
    except Exception:
        return ""

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
        return {"error": "Could not find lyrics."}

    embedding = embed_lyrics(lyrics)
    analysis = analyze_lyrics(lyrics, embedding)
    story = generate_story(analysis)

    return {
        "lyrics_snippet": lyrics[:500] + "...",
        "analysis": analysis,
        "story": story
    }