# diagnosis/forms.py
from django import forms
from django.core.validators import FileExtensionValidator
from django.conf import settings


class AudioUploadForm(forms.Form):
    """Form for uploading audio files"""
    
    audio_file = forms.FileField(
        label='Audio File',
        help_text=f'Max file size: {settings.MAX_AUDIO_FILE_SIZE / (1024*1024)}MB. Allowed formats: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}',
        required=True,
        widget=forms.FileInput(attrs={
            'accept': 'audio/*,.wav,.mp3,.m4a,.ogg,.webm',
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-brand-green focus:border-transparent'
        }),
        validators=[
            FileExtensionValidator(
                allowed_extensions=['wav', 'mp3', 'm4a', 'ogg', 'webm'],
                message='Invalid audio format'
            )
        ]
    )
    
    def clean_audio_file(self):
        """Validate audio file size and format"""
        audio_file = self.cleaned_data.get('audio_file')
        
        if audio_file:
            # Check file size
            if audio_file.size > settings.MAX_AUDIO_FILE_SIZE:
                raise forms.ValidationError(
                    f'File too large. Maximum size is {settings.MAX_AUDIO_FILE_SIZE / (1024*1024)}MB'
                )
            
            # Check file extension
            import os
            file_ext = os.path.splitext(audio_file.name)[1].lower()
            if file_ext not in settings.ALLOWED_AUDIO_FORMATS:
                raise forms.ValidationError(
                    f'Invalid file format. Allowed: {", ".join(settings.ALLOWED_AUDIO_FORMATS)}'
                )
        
        return audio_file
