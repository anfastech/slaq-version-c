# diagnosis/ai_engine/model_loader.py
"""Singleton pattern for model loading

This loader is resilient to the detector class name. Prefer AdvancedStutterDetector
if present, otherwise fall back to StutterDetector for backwards compatibility.
"""
import importlib

_DetectorClass = None
try:
    mod = importlib.import_module('.detect_stuttering', package=__package__)
    if hasattr(mod, 'AdvancedStutterDetector'):
        _DetectorClass = getattr(mod, 'AdvancedStutterDetector')
    elif hasattr(mod, 'StutterDetector'):
        _DetectorClass = getattr(mod, 'StutterDetector')
except Exception:
    _DetectorClass = None

_detector_instance = None

def get_stutter_detector():
    """Get or create singleton detector instance"""
    global _detector_instance
    if _DetectorClass is None:
        raise ImportError("No StutterDetector implementation available in detect_stuttering.py")
    if _detector_instance is None:
        _detector_instance = _DetectorClass()
    return _detector_instance
