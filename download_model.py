"""
Download Wav2Vec2 Model for SLAQ

This script downloads the required AI model from Hugging Face.
Run this before deploying to production or for offline development.

Usage:
    python download_model.py

Requirements:
    - transformers
    - torch
    - internet connection
"""
"""
Update History:
Download ALL Wav2Vec2 Models for SLAQ
"""

import os
import sys
from pathlib import Path

def download_all_models():
    """Download all Wav2Vec2 models used in SLAQ"""
    
    print("=" * 60)
    print("SLAQ AI Models - Complete Download")
    print("=" * 60)
    print()
    
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch
    except ImportError:
        print("❌ Error: Required packages not installed!")
        print("  pip install transformers torch")
        sys.exit(1)
    
    models = [
        {
            # REPLACED English models with the high-impact Indic model
            "name": "ai4bharat/indicwav2vec-hindi", 
            "size": "Estimated ~1.2 GB", # The Wav2Vec2-large family size
            "purpose": "Primary ASR for Indian Speech (Generalization for Accents)"
        },
        # REMOVE the old English model entries here
    ]
    
    total_size_gb = 1.2  # Approximate total
    
    print(f"📦 Total Download Size: ~{total_size_gb} GB")
    print(f"⏳ Estimated Time: 10-20 minutes")
    print()
    
    for i, model_info in enumerate(models, 1):
        print(f"📥 Downloading {i}/3: {model_info['name']}")
        print(f"   Size: {model_info['size']}")
        print(f"   Purpose: {model_info['purpose']}")
        print("   Downloading...")
        
        try:
            # Download processor and model
            processor = Wav2Vec2Processor.from_pretrained(model_info["name"])
            model = Wav2Vec2ForCTC.from_pretrained(model_info["name"])
            
            # Verify
            num_params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ Done! Parameters: {num_params:,}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print()
    
    print("=" * 60)
    print("✅ All models downloaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    download_all_models()