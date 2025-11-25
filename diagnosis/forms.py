# diagnosis/forms.py
from django import forms
from django.core.validators import FileExtensionValidator
from django.conf import settings
import os

class AudioUploadForm(forms.Form):
    """Form for uploading audio files"""
    
    audio_file = forms.FileField(
        label='Audio File',
        # CHANGED: MAX_AUDIO_FILE_SIZE -> MAX_UPLOAD_SIZE
        help_text=f'Max file size: {settings.MAX_UPLOAD_SIZE / (1024*1024)}MB. Allowed formats: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}',
        required=True,
        widget=forms.FileInput(attrs={
            'accept': 'audio/*,.wav,.mp3,.m4a,.ogg,.webm',
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brand-green focus:border-transparent'
        }),
        validators=[
            FileExtensionValidator(
                allowed_extensions=[ext.lstrip('.') for ext in settings.ALLOWED_AUDIO_FORMATS],
                message='Invalid audio format'
            )
        ]
    )
    
    def clean_audio_file(self):
        """Validate audio file size and format"""
        audio_file = self.cleaned_data.get('audio_file')
        
        if audio_file:
            # Check file size
            # CHANGED: MAX_AUDIO_FILE_SIZE -> MAX_UPLOAD_SIZE
            if audio_file.size > settings.MAX_UPLOAD_SIZE:
                max_mb = settings.MAX_UPLOAD_SIZE / (1024*1024)
                raise forms.ValidationError(
                    f'File too large. Maximum size is {max_mb}MB'
                )
            
            # Check file extension
            file_ext = os.path.splitext(audio_file.name)[1].lower()
            if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
                raise forms.ValidationError(
                    f'Invalid file format. Allowed: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}'
                )
        
        return audio_file