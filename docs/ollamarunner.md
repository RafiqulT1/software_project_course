GitHub Copilot: # Ollama Voice Chat Documentation

## Overview
This script creates a voice-based chat interface with an LLM (Language Learning Model) using Ollama. It supports:
- Voice input using microphone
- Speech-to-text using Whisper
- Text-to-speech using Piper TTS
- Chat interaction with Ollama models
- **Emotion Detection:** Understands the user's emotion to provide empathetic responses.
- **Conversation Logging:** Saves the entire conversation to `log.txt`.
- **Memory Recall:** Can answer questions about past conversations by summarizing the log.

## Setup Instructions

### 1. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 2. Install Required Packages
```powershell
pip install ollama
pip install sounddevice
pip install numpy
pip install openai-whisper
pip install soundfile
pip install librosa
pip install piper-tts
pip install text2emotion
pip install emoji==1.7.0
```

### 3. Download NLTK Data
The emotion detection feature requires data from the NLTK library. Run this command once to download it:
```powershell
python -c "import nltk; nltk.download('punkt')"
```
or install it with the python script:
```powershell
python download_nltk_data.py
```

### 4. Download Required Models

#### Ollama Model
```powershell
# Install Ollama from https://ollama.ai/download
# Pull a model (e.g., llama2)
ollama pull llama2
```

#### Piper Voice Model
Create a directory for Piper models and download a voice:
```powershell
# Create directory
mkdir piper_models

# Download voice model (example: Finnish voice)
curl -L -o piper_models/fi_FI-harri-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx
curl -L -o piper_models/fi_FI-harri-medium.json https://huggingface.co/rhasspy/piper-voices/resolve/main/fi/fi_FI/harri/medium/fi_FI-harri-medium.json
```

Available voices can be found at: https://huggingface.co/rhasspy/piper-voices

### 5. Additional Requirements
- FFmpeg is required for Whisper. Install using:
```powershell
winget install Gyan.FFmpeg
```

## Directory Structure
```
software_project_course/
├── venv/
├── piper_models/
│   ├── fi_FI-harri-medium.onnx
│   └── fi_FI-harri-medium.json
├── temp_audio/          # Created automatically
├── log.txt              # Created automatically for conversation history
└── ollamarunner.py
```

## Usage

1. Make sure Ollama is running (check system tray)
2. Activate virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

3. Run the script:
```powershell
python ollamarunner.py
```

4. Interact with the chat:
   - Speak when prompted "Recording... Speak now"
   - Wait for the assistant's voice response
   - Ask questions about your past conversation, like "What did we talk about last time?"
   - Press Ctrl+C to exit

## Troubleshooting

1. **Ollama Connection Error**
   - Ensure Ollama is running
   - Check if models are downloaded using `ollama list`

2. **Emotion Detection Errors**
   - If you see an error like `module 'emoji' has no attribute 'UNICODE_EMOJI'`, ensure you have the correct version by running `pip install emoji==1.7.0 --upgrade`.
   - If you see `Resource punkt not found`, run the NLTK download command from the setup section.

3. **Audio Recording Issues**
   - Check microphone settings in Windows
   - Verify microphone permissions

4. **TTS Issues**
   - Verify Piper model files exist in `piper_models` directory
   - Check file paths in `initialize_tts()`

5. **Memory Issues**
   - Consider using a smaller Whisper model
   - Adjust recording duration in `record_audio()`

## Configuration

Key parameters can be adjusted in the code:
- Recording duration: `record_audio(duration=5)`
- Audio sample rate: `sample_rate=16000`
- TTS settings in `SynthesisConfig`
- Whisper model size in `whisper.load_model("base")`

