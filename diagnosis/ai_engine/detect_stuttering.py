# diagnosis/ai_engine/detect_stuttering.py
import librosa
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.nn import CTCLoss
import logging
from typing import Dict, List, Tuple
import time

logger = logging.getLogger(__name__)


class StutterDetector:
    """
    Stutter detection using Wav2Vec2 models
    Adapted from: https://github.com/wittyicon29/Stutter_Detection
    """
    
    def __init__(self):
        """Initialize models - load once and reuse"""
        logger.info("🔄 Initializing StutterDetector models...")
        
        try:
            # Load base model for transcription
            self.base_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.base_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Load large model for detailed analysis
            self.large_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            self.large_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
            
            # Load XLSR model for target transcript generation
            self.xlsr_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            self.xlsr_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            
            logger.info("✅ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    
    def analyze_audio(self, audio_file_path: str, proper_transcript: str = "") -> Dict:
        """
        Complete analysis pipeline
        
        Args:
            audio_file_path: Path to audio file
            proper_transcript: Optional expected transcript (if available)
        
        Returns:
            Dictionary with complete analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting analysis for: {audio_file_path}")
            
            # Step 1: Generate target transcript if not provided
            if not proper_transcript:
                proper_transcript = self.generate_target_transcript(audio_file_path)
                logger.info(f"📝 Generated target transcript: {proper_transcript}")
            
            proper_transcript = proper_transcript.upper()
            
            # Step 2: Transcribe and detect stuttering
            transcription_result = self.transcribe_and_detect(audio_file_path, proper_transcript)
            
            # Step 3: Calculate CTC loss and find stutter timestamps
            ctc_loss, stutter_timestamps = self.calculate_stutter_timestamps(
                audio_file_path, 
                proper_transcript
            )
            
            # Step 4: Aggregate results
            analysis_duration = time.time() - start_time
            
            result = {
                'actual_transcript': transcription_result['transcription'],
                'target_transcript': proper_transcript,
                'mismatched_chars': transcription_result['stuttered_chars'],
                'mismatch_percentage': transcription_result['mismatch_percentage'],
                'ctc_loss_score': ctc_loss,
                'stutter_timestamps': stutter_timestamps,
                'total_stutter_duration': self._calculate_total_duration(stutter_timestamps),
                'stutter_frequency': self._calculate_frequency(stutter_timestamps, audio_file_path),
                'severity': self._determine_severity(transcription_result['mismatch_percentage']),
                'confidence_score': self._calculate_confidence(transcription_result, ctc_loss),
                'analysis_duration_seconds': round(analysis_duration, 2),
                'model_version': 'wav2vec2-base-960h',
            }
            
            logger.info(f"✅ Analysis complete in {analysis_duration:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise
    
    
    def generate_target_transcript(self, audio_file: str) -> str:
        """Generate expected transcript using XLSR model"""
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            input_values = self.xlsr_processor(waveform[0], return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = self.xlsr_model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_sentences = self.xlsr_processor.batch_decode(predicted_ids)
            
            return predicted_sentences[0]
            
        except Exception as e:
            logger.error(f"Target transcript generation failed: {e}")
            raise
    
    
    def transcribe_and_detect(self, audio_file: str, proper_transcript: str) -> Dict:
        """Transcribe audio and detect stuttering patterns"""
        try:
            # Load audio
            input_audio, _ = librosa.load(audio_file, sr=16000)
            
            # Tokenize
            input_features = self.base_processor(input_audio, return_tensors="pt").input_values
            
            # Get predictions
            with torch.no_grad():
                logits = self.base_model(input_features).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.base_processor.batch_decode(predicted_ids)[0]
            
            # Find stuttered sequences
            stuttered_chars = self.find_sequences_not_in_common(transcription, proper_transcript)
            
            # Calculate mismatch percentage
            total_mismatched = sum(len(segment) for segment in stuttered_chars)
            mismatch_percentage = (total_mismatched / len(proper_transcript)) * 100 if len(proper_transcript) > 0 else 0
            mismatch_percentage = min(round(mismatch_percentage), 100)
            
            return {
                'transcription': transcription,
                'stuttered_chars': stuttered_chars,
                'mismatch_percentage': mismatch_percentage
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    
    def calculate_stutter_timestamps(self, audio_file: str, proper_transcript: str) -> Tuple[float, List[Tuple[float, float]]]:
        """Calculate CTC loss and find exact stutter timestamps"""
        try:
            # Load waveform
            waveform, sample_rate = torchaudio.load(audio_file)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process with base model for CTC loss
            input_values = self.base_processor(waveform[0], return_tensors="pt").input_values
            
            with torch.no_grad():
                logits = self.base_model(input_values).logits
            
            # Calculate CTC loss
            tokens = self.base_processor.tokenizer(proper_transcript, return_tensors="pt", padding=True, truncation=True)
            target_ids = tokens.input_ids
            
            log_probs = torch.log_softmax(logits, dim=-1)
            input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
            target_lengths = torch.tensor([target_ids.shape[1]], dtype=torch.long)
            
            ctc_loss = CTCLoss(blank=self.base_model.config.pad_token_id)
            loss = ctc_loss(log_probs.transpose(0, 1), targets=target_ids, 
                          input_lengths=input_lengths, target_lengths=target_lengths)
            
            # Find stutter timestamps using large model
            input_audio, sample_rate = librosa.load(audio_file, sr=16000)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                input_audio = resampler(torch.from_numpy(input_audio)).numpy()
            
            input_features = self.large_processor(input_audio, return_tensors='pt').input_values
            
            with torch.no_grad():
                logits = self.large_model(input_features).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            blank_token_id = self.large_model.config.pad_token_id
            
            # Extract timestamp ranges
            stuttering_seconds = []
            prev_token = blank_token_id
            frame_shift = 0.02  # 20ms per frame
            audio_duration = len(input_audio) / sample_rate
            
            for frame_idx, token_id in enumerate(predicted_ids[0]):
                if token_id != blank_token_id and token_id != prev_token:
                    start_frame = frame_idx
                    end_frame = frame_idx + token_id.item() - 1
                    start_second = min(start_frame * frame_shift, audio_duration)
                    end_second = min(end_frame * frame_shift, audio_duration)
                    
                    # Detect prolongations (duration > 0.4s)
                    if end_second - start_second > 0.4:
                        stuttering_seconds.append((round(start_second, 2), round(end_second, 2)))
                
                prev_token = token_id
            
            return round(loss.item(), 2), stuttering_seconds
            
        except Exception as e:
            logger.error(f"Timestamp calculation failed: {e}")
            return 0.0, []
    
    
    def find_max_common_characters(self, transcription1: str, transcript2: str) -> str:
        """Longest Common Subsequence algorithm"""
        m, n = len(transcription1), len(transcript2)
        lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if transcription1[i - 1] == transcript2[j - 1]:
                    lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])
        
        # Backtrack to find LCS
        lcs_characters = []
        i, j = m, n
        while i > 0 and j > 0:
            if transcription1[i - 1] == transcript2[j - 1]:
                lcs_characters.append(transcription1[i - 1])
                i -= 1
                j -= 1
            elif lcs_matrix[i - 1][j] > lcs_matrix[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        lcs_characters.reverse()
        return ''.join(lcs_characters)
    
    
    def find_sequences_not_in_common(self, transcription1: str, proper_transcript: str) -> List[str]:
        """Find stuttered character sequences"""
        common_characters = self.find_max_common_characters(transcription1, proper_transcript)
        sequences = []
        sequence = ""
        i, j = 0, 0
        
        while i < len(transcription1) and j < len(common_characters):
            if transcription1[i] == common_characters[j]:
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
                i += 1
                j += 1
            else:
                sequence += transcription1[i]
                i += 1
        
        if sequence:
            sequences.append(sequence)
        
        return sequences
    
    
    def _calculate_total_duration(self, timestamps: List[Tuple[float, float]]) -> float:
        """Calculate total stuttering duration"""
        return sum(end - start for start, end in timestamps)
    
    
    def _calculate_frequency(self, timestamps: List[Tuple[float, float]], audio_file: str) -> float:
        """Calculate stutters per minute"""
        try:
            audio_duration = librosa.get_duration(path=audio_file)
            if audio_duration > 0:
                return (len(timestamps) / audio_duration) * 60
            return 0.0
        except:
            return 0.0
    
    
    def _determine_severity(self, mismatch_percentage: float) -> str:
        """Determine severity level"""
        if mismatch_percentage < 10:
            return 'none'
        elif mismatch_percentage < 25:
            return 'mild'
        elif mismatch_percentage < 50:
            return 'moderate'
        else:
            return 'severe'
    
    
    def _calculate_confidence(self, transcription_result: Dict, ctc_loss: float) -> float:
        """Calculate confidence score for the analysis"""
        # Lower mismatch and lower CTC loss = higher confidence
        mismatch_factor = 1 - (transcription_result['mismatch_percentage'] / 100)
        loss_factor = max(0, 1 - (ctc_loss / 10))  # Normalize loss
        confidence = (mismatch_factor + loss_factor) / 2
        return round(min(max(confidence, 0.0), 1.0), 2)


# diagnosis/ai_engine/model_loader.py
"""Singleton pattern for model loading"""
_detector_instance = None

def get_stutter_detector():
    """Get or create singleton StutterDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = StutterDetector()
    return _detector_instance