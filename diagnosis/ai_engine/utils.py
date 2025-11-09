# diagnosis/ai_engine/utils.py
import os
import logging
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


def convert_audio_to_wav(input_path: str) -> str:
    """
    Convert audio file to WAV format if needed.
    
    Args:
        input_path: Path to input audio file
        
    Returns:
        Path to WAV file (either converted or original if already WAV)
    """
    # Check if already WAV
    if input_path.lower().endswith('.wav'):
        return input_path
    
    try:
        # Generate output path
        base_path = os.path.splitext(input_path)[0]
        output_path = f"{base_path}_converted.wav"
        
        # Check if converted file already exists
        if os.path.exists(output_path):
            logger.info(f"📁 Using existing converted file: {output_path}")
            return output_path
        
        logger.info(f"🔄 Converting {input_path} to WAV format...")
        
        # Load audio with librosa (handles many formats via soundfile/audioread)
        # For webm, we need to use soundfile which wraps libsndfile
        try:
            # Try loading directly with librosa
            audio_data, sample_rate = librosa.load(input_path, sr=None)
            
            # Save as WAV
            sf.write(output_path, audio_data, sample_rate)
            
            logger.info(f"✅ Converted to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Librosa conversion failed: {e}")
            
            # Fallback: Try using soundfile directly
            try:
                audio_data, sample_rate = sf.read(input_path)
                sf.write(output_path, audio_data, sample_rate)
                logger.info(f"✅ Converted using soundfile: {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"❌ Soundfile conversion also failed: {e2}")
                raise RuntimeError(
                    f"Could not convert audio file. Supported formats: WAV, MP3, FLAC, OGG. "
                    f"WebM requires FFmpeg. Error: {e}"
                )
    
    except Exception as e:
        logger.error(f"❌ Audio conversion failed: {e}")
        raise


def cleanup_converted_file(file_path: str):
    """
    Remove temporary converted audio file.
    
    Args:
        file_path: Path to file to delete
    """
    if file_path and file_path.endswith('_converted.wav') and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"🗑️ Cleaned up converted file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete converted file {file_path}: {e}")
