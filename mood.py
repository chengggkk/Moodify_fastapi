import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from embedding import embed_texts, preprocess_long_text

load_dotenv()

mood_router = APIRouter(prefix="/mood", tags=["mood-analysis"])

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from environment variables")

openai.api_key = OPENAI_API_KEY

def create_cot_mood_prompt(text: str, embedding: list) -> str:
    """
    Create Chain of Thought prompt for mood analysis with multiple expert perspectives.
    """
    truncated_text = text[:1000] if len(text) > 1000 else text
    truncated_embedding = embedding[:10] if len(embedding) > 10 else embedding
    
    return f"""
PRINCIPLE INSTRUCTION: You are an expert emotional intelligence analyst. Your task is to provide comprehensive mood analysis through systematic reasoning.

CONTEXT: Analyze the emotional state and mood of the given text using both linguistic content and semantic vector information.

TEXT TO ANALYZE:
{truncated_text}

SEMANTIC VECTOR (first 10 dimensions):
{truncated_embedding}...

ANALYSIS FRAMEWORK - Follow these steps systematically:

Step 1: LINGUISTIC ANALYSIS
- Identify key emotional words and phrases
- Determine tone indicators (positive, negative, neutral)
- Note any emotional intensity markers

Step 2: SEMANTIC VECTOR INTERPRETATION  
- Analyze the numerical patterns in the embedding
- Consider how vector values relate to emotional dimensions
- Cross-reference with linguistic findings

Step 3: CONTEXTUAL EVALUATION
- Consider cultural and situational context
- Evaluate implicit vs explicit emotional content
- Assess emotional complexity and nuance

Step 4: MOOD SYNTHESIS
- Combine linguistic and semantic insights
- Identify primary and secondary emotions
- Determine overall emotional trajectory

EXPERT PERSPECTIVES - Consider these viewpoints:
🧠 Cognitive Psychologist: Focus on thought patterns and mental states
💝 Emotion Specialist: Emphasize feeling intensity and emotional categories  
🎭 Behavioral Analyst: Examine expression patterns and social signals
📊 Data Scientist: Interpret numerical patterns in semantic vectors

VALIDATION CHECKPOINT: After analysis, verify your conclusions by:
- Ensuring consistency between linguistic and semantic findings
- Confirming emotional categories are accurately identified
- Checking that intensity levels match the evidence

OUTPUT REQUIREMENTS:
- Provide comprehensive analysis (minimum 4-5 detailed sentences)
- Include confidence levels for each emotional dimension
- Mention specific evidence from both text and vector analysis
- Use precise emotional vocabulary
- Conclude with actionable insights about the emotional state

Please provide your detailed mood analysis in Chinese, following the systematic approach outlined above.
"""

def get_self_consistency_analysis(text: str, embedding: list, num_attempts: int = 3) -> str:
    """
    Implement self-consistency by running multiple analyses and finding consensus.
    """
    results = []
    
    for i in range(num_attempts):
        # Slightly vary the prompt for each attempt
        variation_prompts = [
            create_cot_mood_prompt(text, embedding),
            create_cot_mood_prompt(text, embedding).replace("ANALYSIS FRAMEWORK", "SYSTEMATIC EVALUATION METHOD"),
            create_cot_mood_prompt(text, embedding).replace("MOOD SYNTHESIS", "EMOTIONAL INTEGRATION")
        ]
        
        selected_prompt = variation_prompts[i % len(variation_prompts)]
        
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": selected_prompt}],
            temperature=0.3 + (i * 0.1),  # Slight temperature variation
            max_tokens=800
        )
        results.append(completion.choices[0].message.content.strip())
    
    # Create consensus prompt
    consensus_prompt = f"""
PRINCIPLE INSTRUCTION: You are an expert analyst synthesizing multiple emotional assessments. Create a definitive mood analysis by identifying common themes and resolving discrepancies.

MULTIPLE ANALYSIS RESULTS:

Analysis 1:
{results[0]}

Analysis 2:
{results[1]}

Analysis 3:
{results[2]}

CONSENSUS METHODOLOGY:
Step 1: Identify recurring themes across all analyses
Step 2: Note areas of agreement and disagreement  
Step 3: Evaluate quality and evidence strength of each analysis
Step 4: Synthesize most reliable and consistent findings
Step 5: Resolve conflicts using strongest evidence

QUALITY VALIDATION:
- Prioritize analyses with specific evidence citations
- Weight conclusions supported by multiple attempts
- Maintain emotional nuance and complexity
- Ensure comprehensive coverage of emotional dimensions

Provide the final consensus analysis in Chinese.
"""
    
    final_completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": consensus_prompt}],
        temperature=0.2,
        max_tokens=600
    )
    
    return final_completion.choices[0].message.content.strip()

def analyze_mood_enhanced(text: str, embedding: list) -> Dict[str, Any]:
    """
    Enhanced mood analysis using Chain of Thought and Self-Consistency.
    """
    # Get self-consistent analysis
    mood_analysis = get_self_consistency_analysis(text, embedding)
    
    # Additional verification step
    verification_prompt = f"""
PRINCIPLE INSTRUCTION: You are a quality assurance specialist for emotional analysis. Verify the accuracy and completeness of this mood analysis.

ORIGINAL TEXT SAMPLE: {text[:200]}...
MOOD ANALYSIS TO VERIFY: {mood_analysis}

VERIFICATION CHECKLIST:
✓ Emotional accuracy - Do conclusions match textual evidence?
✓ Analytical depth - Is analysis sufficiently comprehensive?  
✓ Evidence support - Are claims backed by specific examples?
✓ Emotional granularity - Are subtle emotions captured?
✓ Contextual awareness - Is broader context considered?

IMPROVEMENT AREAS: Identify any missing elements or inaccuracies
CONFIDENCE RATING: Rate analysis quality (1-10) with justification
ENHANCEMENT SUGGESTIONS: Provide specific improvement recommendations

Provide verification feedback in Chinese.
"""
    
    verification = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0.1,
        max_tokens=400
    )
    
    return {
        "mood_analysis": mood_analysis,
        "verification": verification.choices[0].message.content.strip()
    }

# Request models
class MoodAnalysisRequest(BaseModel):
    text: str

# Mood analysis endpoints
@mood_router.post("/analyze")
def analyze_mood_basic(req: MoodAnalysisRequest):
    """
    Basic mood analysis endpoint (maintained for backward compatibility).
    """
    try:
        print(f"Received text length: {len(req.text)}")
        
        # Generate embedding
        print("Generating embedding...")
        embedding = embed_texts(req.text)
        print(f"Embedding generated with {len(embedding)} dimensions")
        
        # Basic mood analysis using enhanced prompt
        truncated_text = req.text[:1000] if len(req.text) > 1000 else req.text
        truncated_embedding = embedding[:10] if len(embedding) > 10 else embedding
        
        prompt = f"""
according to the text and embedding to analyze the mood
text:
{truncated_text}

embedding:
{truncated_embedding}...

use short sentence to describe to mood specifically。
"""
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        mood_analysis = completion.choices[0].message.content.strip()
        print(f"Mood analysis completed: {mood_analysis[:100]}...")
        
        return {
            "text_snippet": req.text[:200] + "..." if len(req.text) > 200 else req.text,
            "mood_analysis": mood_analysis,
            "embedding_dimensions": len(embedding),
            "analysis_type": "basic"
        }
    except Exception as e:
        print(f"Error in mood analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing mood: {str(e)}")

@mood_router.post("/analyze-enhanced")
async def analyze_mood_enhanced_endpoint(req: MoodAnalysisRequest):
    """
    Enhanced mood analysis using Chain of Thought, Self-Consistency, and expert validation.
    """
    try:
        print(f"Processing enhanced mood analysis for text length: {len(req.text)}")
        
        # Generate embedding
        print("Generating semantic embedding...")
        embedding = embed_texts(req.text)
        print(f"Embedding generated: {len(embedding)} dimensions")
        
        # Enhanced mood analysis with multiple techniques
        print("Performing enhanced mood analysis...")
        enhanced_analysis = await run_in_threadpool(analyze_mood_enhanced, req.text, embedding)
        print("Enhanced mood analysis completed")
        
        return {
            "text_snippet": req.text[:200] + "..." if len(req.text) > 200 else req.text,
            "embedding_dimensions": len(embedding),
            "enhanced_mood_analysis": enhanced_analysis["mood_analysis"],
            "quality_verification": enhanced_analysis["verification"],
            "analysis_method": "Chain of Thought + Self-Consistency + Expert Validation",
            "analysis_type": "enhanced"
        }
    except Exception as e:
        print(f"Error in enhanced mood analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced mood analysis failed: {str(e)}")

@mood_router.get("/health")
def mood_health_check():
    """
    Health check endpoint for mood analysis service
    """
    return {
        "status": "healthy", 
        "service": "mood-analysis",
        "openai_configured": bool(OPENAI_API_KEY)
    }