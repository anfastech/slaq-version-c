import torch
import torchaudio
import numpy as np
from transformers import BertTokenizer, BertModel, Wav2Vec2FeatureExtractor

class HybridFeatureExtractor:
    """
    Core Part Extracted: utils.py & train.py
    Combines Acoustic features (Wav2Vec2) with Linguistic features (BERT).
    """
    def __init__(self):
        # Text Encoder
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Audio Encoder
        self.audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    def get_hybrid_features(self, audio_path, transcript):
        # 1. Get Text Embeddings (Context)
        inputs = self.bert_tokenizer(transcript, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_outputs = self.bert_model(**inputs)
        # Use CLS token for sentence-level context (Shape: 768)
        text_embedding = text_outputs.last_hidden_state[:, 0, :].numpy()

        # 2. Get Audio Matrix (Signal)
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        # Extract features (Shape: 1, TimeSteps)
        audio_features = self.audio_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="np").input_values
        audio_matrix = audio_features.squeeze()

        # 3. Concatenate (The "Secret Sauce")
        # We tile the text embedding to match the audio length so every audio frame 
        # "knows" the context of the sentence.
        
        time_steps = audio_matrix.shape[0]
        
        # If audio is too long/short, we might need to adjust, but following the old repo logic:
        # Reshape audio to (Time, 1)
        audio_col = audio_matrix.reshape(-1, 1)
        
        # Tile text to (Time, 768)
        text_tiled = np.tile(text_embedding, (time_steps, 1))
        
        # Final Vector: (Time, 769) -> [Audio_Amplitude, Bert_Dim_1, ..., Bert_Dim_768]
        combined_features = np.concatenate((audio_col, text_tiled), axis=1)
        
        return combined_features