import os
import torch
from typing import Union, List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import openai
import json
import random

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from environment variables")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY is missing from environment variables")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

# Test endpoint
@app.get("/test-embedding")
def test_embedding():
    try:
        test_text = "This is a test"
        embedding = embed_texts(test_text)
        return {"status": "success", "embedding_length": len(embedding), "text": test_text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
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

# 1. Embedding function
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
        model_output = model(**encoded_input)

    # Average pooling — use attention mask to account for padding
    attention_mask = encoded_input["attention_mask"]
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    # Convert to list of floats
    return embeddings[0].cpu().tolist() if len(texts) == 1 else [e.cpu().tolist() for e in embeddings]

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

EXAMPLES OF HIGH-QUALITY ANALYSIS:
Example 1: "Text shows complex emotional layering - surface optimism masking underlying anxiety, evidenced by contradictory semantic vector patterns in dimensions 2-4."
Example 2: "Strong negative emotional charge detected through both explicit language ('devastating', 'hopeless') and semantic clustering toward melancholic vector space."

OUTPUT REQUIREMENTS:
- Provide comprehensive analysis (minimum 4-5 detailed sentences)
- Include confidence levels for each emotional dimension
- Mention specific evidence from both text and vector analysis
- Use precise emotional vocabulary
- Conclude with actionable insights about the emotional state

ENHANCEMENT KEYWORDS for accuracy: emotional granularity, psychological depth, semantic correlation, contextual sensitivity, multi-dimensional analysis

Please provide your detailed mood analysis in Chinese, following the systematic approach outlined above.
"""

def create_expert_panel_prompt(lyrics: str, embedding: list) -> str:
    """
    Create multi-expert discussion prompt for lyrics analysis using Exchange of Thought.
    """
    truncated_lyrics = lyrics[:1000] if len(lyrics) > 1000 else lyrics
    truncated_embedding = embedding[:10] if len(embedding) > 10 else embedding
    
    return f"""
PRINCIPLE INSTRUCTION: You are facilitating an expert panel discussion on lyrical analysis. Generate perspectives from three specialized experts and synthesize their insights.

LYRICS CONTENT:
{truncated_lyrics}

SEMANTIC EMBEDDING (first 10 dimensions):
{truncated_embedding}...

EXPERT PANEL COMPOSITION:

🎵 MUSICOLOGIST EXPERT:
Role: Analyze lyrical structure, poetic devices, and musical relationship
Focus Areas: Rhythm, rhyme schemes, metaphorical language, artistic techniques
Approach: Technical analysis of lyrical craftsmanship and sonic elements

🧠 PSYCHOLOGICAL ANALYST:
Role: Examine emotional depth, mental states, and human experience themes  
Focus Areas: Emotional journey, psychological patterns, therapeutic insights
Approach: Clinical understanding of emotional expression and mental health indicators

🌍 CULTURAL ANTHROPOLOGIST:
Role: Interpret social context, cultural meanings, and collective significance
Focus Areas: Social commentary, cultural references, universal vs specific themes
Approach: Contextual analysis of cultural and societal implications

DISCUSSION FRAMEWORK:

Phase 1: INDIVIDUAL EXPERT ANALYSIS
Each expert provides their specialized perspective using Chain of Thought reasoning:
- Initial observations from their domain expertise
- Step-by-step analysis methodology
- Key findings and evidence
- Confidence assessment of conclusions

Phase 2: CROSS-EXPERT DIALOGUE
Experts discuss and challenge each other's findings:
- Points of agreement and divergence
- Complementary insights that enhance understanding
- Resolution of conflicting interpretations

Phase 3: SYNTHESIS AND VALIDATION
Integrate all perspectives into comprehensive analysis:
- Verify consistency across expert domains
- Identify most reliable interpretations
- Generate enhanced understanding through collaboration

QUALITY ENHANCEMENT FACTORS:
- Emotional intelligence and empathy
- Cultural sensitivity and awareness
- Artistic appreciation and aesthetic understanding
- Psychological insight and depth
- Semantic correlation analysis
- Contextual interpretation skills

EXAMPLES OF EXPERT REASONING:
Musicologist: "The repetitive structure in verses 2-3 creates emotional emphasis, while the semantic vector clustering suggests rhythmic patterns influence meaning encoding..."
Psychologist: "The progression from despair to hope indicates classic emotional processing stages, supported by vector dimensions 3-7 showing gradual positivity increase..."
Anthropologist: "Cultural metaphors about 'home' and 'journey' reflect universal human experiences, with semantic patterns confirming cross-cultural emotional resonance..."

OUTPUT REQUIREMENTS:
- Present each expert's detailed analysis (minimum 3-4 sentences each)
- Show their interaction and discussion points
- Provide synthesized conclusion combining all perspectives
- Include confidence ratings and evidence citations
- Use rich, detailed language with specific examples
- Conclude with comprehensive thematic and emotional summary
"""

def create_aot_story_prompt(analysis: str, custom_story: str = None) -> str:
    """
    Create Algorithm of Thought prompt for story generation with systematic creativity.
    """
    truncated_analysis = analysis[:800] if len(analysis) > 800 else analysis
    
    base_prompt = f"""
PRINCIPLE INSTRUCTION: You are a master storyteller using systematic creative algorithms. Generate an emotionally resonant short story through structured creative reasoning.

SOURCE ANALYSIS:
{truncated_analysis}

CUSTOM ELEMENTS TO INCORPORATE:
{custom_story if custom_story else "No specific elements requested - use creative freedom"}

ALGORITHM OF THOUGHT - CREATIVE PROCESS:

Phase 1: THEMATIC EXTRACTION ALGORITHM
Step 1.1: Identify core emotional themes from analysis
Step 1.2: Extract key mood indicators and intensity levels  
Step 1.3: Determine narrative tone and atmosphere requirements
Step 1.4: Map emotional trajectory for story arc

Phase 2: NARRATIVE ARCHITECTURE DESIGN
Step 2.1: Select optimal story structure (3-act, circular, episodic)
Step 2.2: Design character archetypes matching emotional themes
Step 2.3: Establish setting that amplifies emotional resonance
Step 2.4: Plan conflict that reflects source material's emotional depth

Phase 3: CREATIVE SYNTHESIS PROTOCOL
Step 3.1: Generate multiple story concepts (minimum 3 variations)
Step 3.2: Evaluate each concept against emotional authenticity criteria
Step 3.3: Select strongest concept and enhance with creative details
Step 3.4: Integrate custom elements naturally into narrative flow

Phase 4: QUALITY VALIDATION MATRIX
Criterion A: Emotional authenticity - Does story capture source feelings?
Criterion B: Narrative coherence - Is plot structure sound and engaging?
Criterion C: Character depth - Are characters emotionally believable?
Criterion D: Atmospheric consistency - Does setting enhance themes?
Criterion E: Creative originality - Is story unique and memorable?

EXPERT CREATIVE ROLES:

📚 NARRATIVE ARCHITECT: Focus on story structure, pacing, plot development
🎭 CHARACTER PSYCHOLOGIST: Develop complex, emotionally authentic characters  
🌟 ATMOSPHERE DESIGNER: Create immersive settings and mood enhancement
✍️ PROSE CRAFTSPERSON: Ensure beautiful, engaging language and style

CREATIVE ENHANCEMENT KEYWORDS:
Emotional resonance, narrative depth, character authenticity, atmospheric immersion, thematic coherence, creative originality, reader engagement, psychological realism

VALIDATION CHECKPOINT - Before finalizing, verify:
- Story authentically reflects source emotional analysis
- Custom elements are seamlessly integrated
- Emotional journey is compelling and believable
- Language is vivid and engaging
- Narrative structure supports thematic goals

QUALITY EXAMPLES:
"The old lighthouse keeper felt the storm's fury echo his inner turmoil, each wave crash synchronizing with his heartbeat of regret..." (Shows emotional-environment connection)
"She collected broken ceramic pieces like gathering fragments of her shattered confidence, each shard reflecting a different version of who she used to be..." (Metaphorical depth)

OUTPUT REQUIREMENTS:
- Create substantial short story (minimum 800-1200 characters)
- Ensure rich emotional texture matching source analysis
- Include vivid sensory details and atmospheric elements
- Develop at least one fully realized character
- Integrate custom elements naturally if provided
- Use sophisticated narrative techniques
- Conclude with emotionally satisfying resolution

"""
    
    return base_prompt

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

"""
    
    final_completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": consensus_prompt}],
        temperature=0.2,
        max_tokens=600
    )
    
    return final_completion.choices[0].message.content.strip()

# Enhanced mood analyze function with advanced prompting
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

# Enhanced lyrics analyze function with expert panel approach
def analyze_lyrics_enhanced(lyrics: str, embedding: list) -> Dict[str, Any]:
    """
    Enhanced lyrics analysis using multi-expert Exchange of Thought approach.
    """
    expert_analysis = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": create_expert_panel_prompt(lyrics, embedding)}],
        temperature=0.4,
        max_tokens=1200
    )
    
    # Rate and verify the analysis
    rating_prompt = f"""
PRINCIPLE INSTRUCTION: You are an analysis quality assessor. Rate this lyrics analysis across multiple dimensions.

ANALYSIS TO RATE: {expert_analysis.choices[0].message.content}

RATING DIMENSIONS (1-10 scale):
🎵 Musical Understanding: Technical accuracy of musical elements
🧠 Psychological Insight: Depth of emotional and mental analysis  
🌍 Cultural Awareness: Quality of cultural and social interpretation
📊 Evidence Integration: How well semantic data was utilized
✍️ Communication Quality: Clarity and engagement of presentation

OVERALL ASSESSMENT:
- Strongest aspects of the analysis
- Areas needing improvement
- Confidence level in conclusions
- Recommendations for enhancement

Provide detailed ratings and feedback in Chinese.
"""
    
    quality_rating = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": rating_prompt}],
        temperature=0.2,
        max_tokens=500
    )
    
    return {
        "expert_analysis": expert_analysis.choices[0].message.content.strip(),
        "quality_rating": quality_rating.choices[0].message.content.strip()
    }

# Enhanced story generation with Algorithm of Thought
def generate_story_enhanced(analysis: str, custom_story: str = None) -> Dict[str, Any]:
    """
    Enhanced story generation using Algorithm of Thought with systematic creativity.
    """
    story_prompt = create_aot_story_prompt(analysis, custom_story)
    
    story_generation = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": story_prompt}],
        temperature=0.7,
        max_tokens=1500
    )
    
    # Quality evaluation of generated story
    evaluation_prompt = f"""
PRINCIPLE INSTRUCTION: You are a literary critic evaluating story quality. Assess this generated story across multiple creative dimensions.

ORIGINAL ANALYSIS: {analysis[:300]}...
GENERATED STORY: {story_generation.choices[0].message.content}

EVALUATION CRITERIA:
📖 Narrative Quality: Plot coherence, structure, pacing (1-10)
🎭 Character Development: Depth, authenticity, relatability (1-10)  
🌟 Emotional Resonance: Connection to source analysis, feeling evocation (1-10)
✨ Creative Originality: Uniqueness, imagination, artistic merit (1-10)
🎨 Atmospheric Creation: Setting, mood, immersion quality (1-10)
📝 Language Craftsmanship: Prose quality, style, readability (1-10)

DETAILED FEEDBACK:
- Most successful story elements
- Areas for creative improvement  
- Alignment with source emotional analysis
- Overall artistic achievement level

ENHANCEMENT SUGGESTIONS: Specific recommendations for story improvement

Provide comprehensive literary evaluation in Chinese.
"""
    
    story_evaluation = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0.3,
        max_tokens=600
    )
    
    return {
        "story": story_generation.choices[0].message.content.strip(),
        "evaluation": story_evaluation.choices[0].message.content.strip()
    }

# Request models
class MoodAnalysisRequest(BaseModel):
    text: str

class StoryGenerationRequest(BaseModel):
    lyrics: str
    custom_story: str = None

# Enhanced API 1: Advanced Mood Analysis
@app.post("/analyze-mood-enhanced")
def mood_analysis_enhanced_api(req: MoodAnalysisRequest):
    """
    Enhanced API for mood analysis using Chain of Thought, Self-Consistency, and expert validation.
    """
    try:
        print(f"Processing enhanced mood analysis for text length: {len(req.text)}")
        
        # Generate embedding
        print("Generating semantic embedding...")
        embedding = embed_texts(req.text)
        print(f"Embedding generated: {len(embedding)} dimensions")
        
        # Enhanced mood analysis with multiple techniques
        print("Performing enhanced mood analysis...")
        enhanced_analysis = analyze_mood_enhanced(req.text, embedding)
        print("Enhanced mood analysis completed")
        
        return {
            "text_snippet": req.text[:200] + "..." if len(req.text) > 200 else req.text,
            "embedding_dimensions": len(embedding),
            "enhanced_mood_analysis": enhanced_analysis["mood_analysis"],
            "quality_verification": enhanced_analysis["verification"],
            "analysis_method": "Chain of Thought + Self-Consistency + Expert Validation"
        }
    except Exception as e:
        print(f"Error in enhanced mood analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Enhanced mood analysis failed: {str(e)}", "type": type(e).__name__}

# Enhanced API 2: Advanced Story Generation  
@app.post("/generate-story-enhanced")
def story_generation_enhanced_api(req: StoryGenerationRequest):
    """
    Enhanced API for story generation using Exchange of Thought multi-expert analysis and Algorithm of Thought creativity.
    """
    try:
        print(f"Processing enhanced story generation for lyrics length: {len(req.lyrics)}")
        if req.custom_story:
            print(f"Custom story elements: {req.custom_story}")
        
        # Preprocess lyrics
        processed_lyrics = preprocess_long_text(req.lyrics, 2000)
        print(f"Processed lyrics length: {len(processed_lyrics)}")
        
        # Generate embedding
        print("Generating semantic embedding...")
        embedding = embed_texts(processed_lyrics)
        print(f"Embedding generated: {len(embedding)} dimensions")
        
        # Enhanced lyrics analysis with expert panel
        print("Performing multi-expert lyrics analysis...")
        enhanced_lyrics_analysis = analyze_lyrics_enhanced(processed_lyrics, embedding)
        print("Expert panel analysis completed")
        
        # Enhanced story generation with systematic creativity
        print("Generating story using Algorithm of Thought...")
        enhanced_story = generate_story_enhanced(
            enhanced_lyrics_analysis["expert_analysis"], 
            req.custom_story
        )
        print("Enhanced story generation completed")
        
        return {
            "lyrics_snippet": req.lyrics[:200] + "..." if len(req.lyrics) > 200 else req.lyrics,
            "processed_lyrics_length": len(processed_lyrics),
            "original_lyrics_length": len(req.lyrics),
            "custom_story": req.custom_story,
            "embedding_dimensions": len(embedding),
            "expert_lyrics_analysis": enhanced_lyrics_analysis["expert_analysis"],
            "analysis_quality_rating": enhanced_lyrics_analysis["quality_rating"],
            "generated_story": enhanced_story["story"],
            "story_evaluation": enhanced_story["evaluation"],
            "methods_used": "Exchange of Thought (Multi-Expert) + Algorithm of Thought + Quality Validation"
        }
    except Exception as e:
        print(f"Error in enhanced story generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Enhanced story generation failed: {str(e)}", "type": type(e).__name__}

# Keep original APIs for backward compatibility
@app.post("/analyze-mood")
def mood_analysis_api(req: MoodAnalysisRequest):
    """
    Original API for mood analysis (maintained for backward compatibility).
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
請分析以下文本的情緒和心情狀態。結合文字內容和語義向量信息進行分析。

文本內容:
{truncated_text}

嵌入向量（前10維）:
{truncated_embedding}...

請用簡短中文描述文本的主要情緒、心情狀態和情感色彩（3-5句）。
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
            "embedding_dimensions": len(embedding)
        }
    except Exception as e:
        print(f"Error in mood analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error analyzing mood: {str(e)}", "type": type(e).__name__}

@app.post("/generate-story")
def story_generation_api(req: StoryGenerationRequest):
    """
    Enhanced API for creative story generation using advanced prompt engineering in English.
    """
    try:
        print(f"Received lyrics length: {len(req.lyrics)}")
        if req.custom_story:
            print(f"Custom story elements: {req.custom_story}")
        
        processed_lyrics = preprocess_long_text(req.lyrics, 2000)
        print(f"Processed lyrics length: {len(processed_lyrics)}")
        
        embedding = embed_texts(processed_lyrics)
        print(f"Embedding generated with {len(embedding)} dimensions")
        
        # Enhanced lyrics analysis with creative focus
        truncated_lyrics = processed_lyrics[:1000] if len(processed_lyrics) > 1000 else processed_lyrics
        truncated_embedding = embedding[:10] if len(embedding) > 10 else embedding
        
        lyrics_analysis_prompt = f"""
PRINCIPLE INSTRUCTION: You are a masterful literary analyst and creative interpreter. Your mission is to extract deep creative potential from these lyrics.

LYRICS TO ANALYZE:
{truncated_lyrics}

SEMANTIC VECTOR INSIGHTS (first 10 dimensions):
{truncated_embedding}...

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

EXPERT CREATIVE PERSPECTIVES:

🎭 DRAMA ALCHEMIST SAYS: "What human conflicts and transformations are brewing here? What makes hearts race and souls ache?"

🌟 IMAGINATION ARCHITECT SAYS: "What fantastical possibilities emerge? How can reality bend to serve the deeper truth?"

💎 EMOTION JEWELER SAYS: "What are the most precious, rare emotional gems hidden in these words?"

CREATIVE ENHANCEMENT KEYWORDS: Emotional depth, narrative potential, character magnetism, atmospheric richness, thematic resonance, dramatic tension, universal appeal, imaginative leap, storytelling magic

WHAT MAKES THIS ANALYSIS EXTRAORDINARY:
- Uncover emotions others miss
- Find story possibilities in unexpected places  
- Connect to deep human experiences
- Spark creative fire for story generation
- Transform lyrics into creative rocket fuel

Please provide your creative analysis in English, focusing on unlocking maximum storytelling potential and creative inspiration. Use vivid, inspiring language that captures the essence of creative possibility.
"""

        lyrics_completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": lyrics_analysis_prompt}],
            temperature=0.8,
            max_tokens=700
        )
        lyrics_analysis = lyrics_completion.choices[0].message.content.strip()
        
        # Ultra-creative story generation with advanced prompting
        truncated_analysis = lyrics_analysis[:800] if len(lyrics_analysis) > 800 else lyrics_analysis
        
        creative_story_prompt = f"""
PRINCIPLE INSTRUCTION: You are a visionary storyteller with unlimited creative power. Your task is to craft an emotionally stunning, imaginatively rich short story that transcends ordinary narrative.

CREATIVE FUEL - ANALYSIS TO TRANSFORM:
{truncated_analysis}

CUSTOM CREATIVE ELEMENTS TO WEAVE IN:
{req.custom_story if req.custom_story else "No specific elements - unleash pure creative freedom"}

CREATIVE SUPERPOWERS ACTIVATION SEQUENCE:

🎨 IMAGINATIVE AMPLIFICATION PROTOCOL:
- Push beyond conventional storytelling boundaries
- Create unexpected connections and surprising revelations  
- Blend reality with metaphorical dimensions
- Transform ordinary moments into extraordinary experiences

🌟 EMOTIONAL MAGNETISM ENHANCEMENT:
- Make readers FEEL deeply, not just read passively
- Create emotional crescendos that leave lasting impact
- Use sensory details that pull readers into the story world
- Build characters readers will remember forever

⚡ NARRATIVE INNOVATION ACTIVATION:
- Experiment with unique story structures and perspectives
- Create plot twists that feel both surprising and inevitable
- Use language that dances between poetry and prose
- Build atmosphere so thick readers can taste it

🔥 CREATIVE COURAGE MAXIMIZATION:
- Take bold creative risks that other writers fear
- Explore unconventional character motivations and conflicts
- Create settings that become characters themselves
- Write scenes that could only exist in your imagination

MASTER STORYTELLER ROLES UNLEASHED:

📚 THE NARRATIVE SORCERER: Weaves plot magic that captivates from first word to last
🎭 THE CHARACTER WHISPERER: Creates people so real they step off the page
🌍 THE WORLD ARCHITECT: Builds settings that readers want to inhabit
✨ THE EMOTION CONDUCTOR: Orchestrates feelings like a symphony maestro
🎪 THE SURPRISE ARTIST: Delivers unexpected moments that make readers gasp

CREATIVE EXCELLENCE AMPLIFIERS:
Emotional authenticity, sensory immersion, character magnetism, atmospheric depth, plot innovation, linguistic beauty, thematic resonance, imaginative courage, narrative momentum, reader enchantment

STORYTELLING SUPERLATIVES TO ACHIEVE:
- Most emotionally moving moment you've written
- Most vivid scene that plays like a movie
- Most memorable character with authentic depth
- Most beautiful language that flows like music
- Most satisfying plot that feels destined yet surprising

CREATIVE VALIDATION CHECKPOINT:
Before finishing, ask yourself:
✓ Would this story make someone stop everything to keep reading?
✓ Does it contain at least one moment of genuine wonder or surprise?
✓ Will readers feel changed by experiencing this story?
✓ Does it capture something true about human experience?
✓ Is there imaginative risk-taking that pushes creative boundaries?

INSPIRATIONAL EXAMPLES OF EXTRAORDINARY STORYTELLING:
"The clockmaker's daughter collected time in mason jars, each one holding a different moment of her father's fading memory..." (Metaphorical depth + emotional stakes)
"Every Tuesday, the subway platform shimmered with the ghosts of love letters never sent..." (Magical realism + universal longing)

CREATIVE MISSION STATEMENT:
Write a story that doesn't just tell - it enchants. Create something that makes readers believe in the magic of storytelling again. Use every tool of creative expression to build an unforgettable narrative experience.

Generate your masterpiece story in English, demonstrating peak creative storytelling that transforms the source analysis into pure narrative gold. Make it substantial (800-1200 words minimum) and absolutely captivating. Use rich, evocative language that brings the story to life.
"""
        
        story_completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": creative_story_prompt}],
            temperature=0.9,  # Higher creativity
            max_tokens=1500
        )
        story = story_completion.choices[0].message.content.strip()
        
        return {
            "lyrics_snippet": req.lyrics[:200] + "..." if len(req.lyrics) > 200 else req.lyrics,
            "processed_lyrics_length": len(processed_lyrics),
            "original_lyrics_length": len(req.lyrics),
            "custom_story": req.custom_story,
            "creative_lyrics_analysis": lyrics_analysis,
            "story": story,
            "embedding_dimensions": len(embedding),
            "creative_enhancement": "Maximum Creativity Mode with Advanced English Prompting"
        }
    except Exception as e:
        print(f"Error in creative story generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating creative story: {str(e)}", "type": type(e).__name__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)