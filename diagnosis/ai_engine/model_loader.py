# diagnosis/ai_engine/model_loader.py
"""
Singleton pattern for API-based detector loading

This loader provides a singleton instance of the StutterDetector class,
which is an API client that calls the external AI Engine service.
No local models are loaded - all processing happens via API calls.

Architecture: Django App (Client) → AI Engine API (Service)
"""
import logging
import importlib

logger = logging.getLogger(__name__)

_DetectorClass = None
_detector_instance = None


def _load_detector_class():
    """Load the detector class from detect_stuttering module."""
    global _DetectorClass
    
    if _DetectorClass is None:
        try:
            mod = importlib.import_module('.detect_stuttering', package=__package__)
            # Prefer AdvancedStutterDetector, fall back to StutterDetector
            if hasattr(mod, 'AdvancedStutterDetector'):
                _DetectorClass = getattr(mod, 'AdvancedStutterDetector')
                logger.info("✅ Loaded AdvancedStutterDetector (API client)")
            elif hasattr(mod, 'StutterDetector'):
                _DetectorClass = getattr(mod, 'StutterDetector')
                logger.info("✅ Loaded StutterDetector (API client)")
            else:
                raise AttributeError("No detector class found in detect_stuttering module")
        except Exception as e:
            logger.error(f"❌ Failed to load detector class: {e}")
            raise ImportError("No StutterDetector implementation available in detect_stuttering.py") from e
    
    return _DetectorClass


def get_stutter_detector():
    """
    Get or create singleton detector instance (API client).
    
    This returns an API client that calls the external AI Engine service.
    No local ML models are loaded - all processing is done via HTTP requests.
    
    Returns:
        StutterDetector or AdvancedStutterDetector: API client instance
        
    Raises:
        ImportError: If detector class cannot be loaded
    """
    global _detector_instance
    
    # Load detector class if not already loaded
    _load_detector_class()
    
    if _DetectorClass is None:
        raise ImportError("No StutterDetector implementation available in detect_stuttering.py")
    
    # Create singleton instance if not exists
    if _detector_instance is None:
        logger.info("🔄 Creating detector instance (API client mode)...")
        _detector_instance = _DetectorClass()
        logger.info("✅ Detector instance created (no local models loaded)")
    
    return _detector_instance


def reset_detector():
    """
    Reset the singleton instance (useful for testing or reconfiguration).
    
    Note: This will force recreation of the API client on next get_stutter_detector() call.
    """
    global _detector_instance
    _detector_instance = None
    logger.info("🔄 Detector instance reset")
