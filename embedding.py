import torch
import logging
from typing import Union, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
import os
from dotenv import load_dotenv

load_dotenv()

# Suppress warnings
logging.getLogger().setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()

embedding_router = APIRouter(prefix="/embedding", tags=["embedding"])

# Get API key
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY is missing from environment variables")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-embeddings-v3",
    token=HF_API_KEY,
    trust_remote_code=True,
)

embed_model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v3",
    token=HF_API_KEY,
    trust_remote_code=True,
).to(device)

def preprocess_long_text(text: str, max_length: int = 1000) -> str:
    """
    Preprocess long text by taking the most relevant parts.
    For lyrics, take the beginning and end to capture the full emotional arc.
    """
    if len(text) <= max_length:
        return text
    
    # For very long text, take first half and last quarter
    first_part_len = max_length // 2
    last_part_len = max_length // 4
    
    first_part = text[:first_part_len]
    last_part = text[-last_part_len:]
    
    return first_part + "\n...\n" + last_part

def embed_texts(texts: Union[str, List[str]]) -> List[float]:
    """
    Generate embedding vectors for English and Chinese texts using jinaai/jina-embeddings-v3.

    Args:
        texts (Union[str, List[str]]): A single string or a list of strings (English or Chinese).

    Returns:
        List[float]: Embedding vector (or list of vectors if input is a list).
    """
    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize and pad
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Get model outputs
    with torch.no_grad():
        model_output = embed_model(**encoded_input)

    # Average pooling — use attention mask to account for padding
    attention_mask = encoded_input["attention_mask"]
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    # Convert to list of floats
    return embeddings[0].cpu().tolist() if len(texts) == 1 else [e.cpu().tolist() for e in embeddings]

# Request models
class TextEmbeddingRequest(BaseModel):
    text: str

# Embedding endpoints
@embedding_router.post("/generate")
def generate_text_embedding(req: TextEmbeddingRequest):
    """
    Generate text embedding using jinaai/jina-embeddings-v3
    """
    try:
        print(f"Generating embedding for text length: {len(req.text)}")
        
        # Preprocess if text is too long
        processed_text = preprocess_long_text(req.text, 2000)
        
        # Generate embedding
        embedding = embed_texts(processed_text)
        
        return {
            "original_text_length": len(req.text),
            "processed_text_length": len(processed_text),
            "text_snippet": req.text[:200] + "..." if len(req.text) > 200 else req.text,
            "embedding": embedding,
            "embedding_dimensions": len(embedding),
            "model": "jinaai/jina-embeddings-v3"
        }
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@embedding_router.get("/test")
def test_embedding():
    """
    Test endpoint for embedding functionality
    """
    try:
        test_text = "This is a test"
        embedding = embed_texts(test_text)
        return {
            "status": "success", 
            "embedding_length": len(embedding), 
            "text": test_text,
            "model": "jinaai/jina-embeddings-v3",
            "device": str(device)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}