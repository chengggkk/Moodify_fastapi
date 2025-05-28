import os
import torch
import logging
import numpy as np
import cv2
import warnings
import io
from typing import Union, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from transformers import DonutProcessor, VisionEncoderDecoderModel, logging as transformers_logging
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set logging level to ERROR to suppress warnings and info messages
logging.getLogger().setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the router
donut_router = APIRouter(prefix="/ocr", tags=["OCR"])

# Donut model configuration
DONUT_MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for the Donut model
donut_processor = None
donut_model = None

def initialize_donut_model():
    """Initialize the Donut model for OCR tasks"""
    global donut_processor, donut_model
    
    try:
        print("Loading Donut model for OCR...")
        donut_processor = DonutProcessor.from_pretrained(
            DONUT_MODEL_NAME,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        donut_model = VisionEncoderDecoderModel.from_pretrained(
            DONUT_MODEL_NAME,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        donut_model.to(device)
        print(f"Donut model loaded successfully on {device}!")
        return True
    except Exception as e:
        print(f"Error loading Donut model: {e}")
        donut_processor = None
        donut_model = None
        return False

def preprocess_image(pil_img):
    """
    Apply optimized preprocessing techniques for better text recognition
    """
    try:
        # Convert PIL to OpenCV format
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Resize for better processing
        target_width = 1024
        h, w = img.shape[:2]
        scale = target_width / w
        new_height = int(h * scale)
        img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        # Convert back to PIL format
        enhanced_img = Image.fromarray(binary)
        
        # Sharpen the image
        enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)
        
        return enhanced_img
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return pil_img

def create_processing_variants(pil_img):
    """
    Create image variants for better OCR processing
    """
    try:
        variants = []
        
        # Original image
        variants.append(pil_img)
        
        # Standard preprocessing
        try:
            variants.append(preprocess_image(pil_img))
        except Exception as e:
            print(f"Error creating preprocessed variant: {str(e)}")
        
        # High contrast grayscale
        try:
            gray_img = pil_img.convert('L')
            contrast_enhancer = ImageEnhance.Contrast(gray_img)
            high_contrast_img = contrast_enhancer.enhance(2.5)
            variants.append(high_contrast_img)
        except Exception as e:
            print(f"Error creating high contrast variant: {str(e)}")
        
        # Enhanced brightness variant
        try:
            brightness_enhancer = ImageEnhance.Brightness(pil_img)
            bright_img = brightness_enhancer.enhance(1.3)
            variants.append(bright_img)
        except Exception as e:
            print(f"Error creating brightness variant: {str(e)}")
        
        return variants
    except Exception as e:
        print(f"Error in creating image variants: {str(e)}")
        return [pil_img]

# Initialize the model when the module is imported
model_loaded = initialize_donut_model()

@donut_router.get("/health")
def ocr_health_check():
    """Health check for OCR service"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "message": "OCR service is running" if model_loaded else "OCR model failed to load"
    }

@donut_router.post("/recognize")
async def recognize_image(image: UploadFile = File(...)):
    """
    OCR API for extracting raw text content from images using Donut model
    """
    try:
        # Check if Donut model is loaded
        if donut_processor is None or donut_model is None:
            raise HTTPException(status_code=500, detail="Donut model not loaded correctly")
        
        # Read the uploaded image
        img_bytes = await image.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Create processed variants of the image
        image_variants = create_processing_variants(original_img)
        
        results = []
        
        # Try with each image variant using Donut model
        for i, img_variant in enumerate(image_variants):
            try:
                # Define a task-specific prompt for document understanding
                task_prompt = "<s_cord-v2>"
                
                # Process the image
                pixel_values = donut_processor(img_variant, return_tensors="pt").pixel_values.to(device)
                decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                
                # Generate text from image
                generated_ids = donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    early_stopping=True,
                    num_beams=4,
                    do_sample=False,
                    num_return_sequences=1,
                    length_penalty=1.0,
                    use_cache=True
                )
                
                # Decode the generated text
                donut_output = donut_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                print(f"Donut OCR output for variant {i}: {donut_output}")
                
                results.append({
                    "variant": i,
                    "preprocessing": "original" if i == 0 else f"variant_{i}",
                    "raw_text": donut_output,
                    "text_length": len(donut_output)
                })
                
            except Exception as e:
                print(f"Error processing variant {i}: {e}")
                results.append({
                    "variant": i,
                    "preprocessing": "original" if i == 0 else f"variant_{i}",
                    "raw_text": "",
                    "text_length": 0,
                    "error": str(e)
                })
        
        # Find the best result (longest text output)
        best_result = max(results, key=lambda x: x.get("text_length", 0))
        
        return {
            "best_result": best_result,
            "all_results": results,
            "total_variants_processed": len(results),
            "image_info": {
                "filename": image.filename,
                "content_type": image.content_type,
                "size": len(img_bytes)
            }
        }
        
    except Exception as e:
        print(f"Error in image recognition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@donut_router.post("/recognize-simple")
async def recognize_image_simple(image: UploadFile = File(...)):
    """
    Simple OCR API that returns only the raw text content from the best variant
    """
    try:
        # Check if Donut model is loaded
        if donut_processor is None or donut_model is None:
            raise HTTPException(status_code=500, detail="Donut model not loaded correctly")
        
        # Read the uploaded image
        img_bytes = await image.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Create processed variants of the image
        image_variants = create_processing_variants(original_img)
        
        best_text = ""
        best_length = 0
        
        # Try with each image variant using Donut model
        for i, img_variant in enumerate(image_variants):
            try:
                # Define a task-specific prompt for document understanding
                task_prompt = "<s_cord-v2>"
                
                # Process the image
                pixel_values = donut_processor(img_variant, return_tensors="pt").pixel_values.to(device)
                decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                
                # Generate text from image
                generated_ids = donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    early_stopping=True,
                    num_beams=4,
                    do_sample=False,
                    num_return_sequences=1,
                    length_penalty=1.0,
                    use_cache=True
                )
                
                # Decode the generated text
                donut_output = donut_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                print(f"Donut OCR output for variant {i}: {donut_output}")
                
                # Keep track of the best result
                if len(donut_output) > best_length:
                    best_text = donut_output
                    best_length = len(donut_output)
                
            except Exception as e:
                print(f"Error processing variant {i}: {e}")
                continue
        
        return {
            "raw_text": best_text,
            "text_length": best_length,
            "success": best_length > 0
        }
        
    except Exception as e:
        print(f"Error in simple image recognition: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")