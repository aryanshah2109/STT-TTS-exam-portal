import os
import uuid
from gtts import gTTS
import tempfile
from pathlib import Path

 


def generate_tts_audio(text: str, language: str = "en", slow: bool = False) -> str:
    """
    Generate speech audio from text using gTTS.
    Saves file locally inside generated_audio/ folder.
    Returns the local file path.
    
    For HF Spaces, uses /tmp for temporary storage.
    """

    # Generate audio
    tts = gTTS(text=text, lang=language, slow=slow)

    # Create unique filename
    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(BASE_DIR, filename)

    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to disk
    tts.save(file_path)

    return file_path
