GitHub Copilot: # Ollama Voice Chat Documentation

## Overview
This script creates a voice-based chat interface with an LLM (Language Learning Model) using Ollama. It supports:
- Voice input using microphone
- Speech-to-text using Whisper
- Text-to-speech using Piper TTS
- Chat interaction with Ollama models

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
```

### 3. Download Required Models

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

### 4. Additional Requirements
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
   - Press Ctrl+C to exit

## Troubleshooting

1. **Ollama Connection Error**
   - Ensure Ollama is running
   - Check if models are downloaded using `ollama list`

2. **Audio Recording Issues**
   - Check microphone settings in Windows
   - Verify microphone permissions

3. **TTS Issues**
   - Verify Piper model files exist in `piper_models` directory
   - Check file paths in `initialize_tts()`

4. **Memory Issues**
   - Consider using a smaller Whisper model
   - Adjust recording duration in `record_audio()`

## Configuration

Key parameters can be adjusted in the code:
- Recording duration: `record_audio(duration=5)`
- Audio sample rate: `sample_rate=16000`
- TTS settings in `SynthesisConfig`
- Whisper model size in `whisper.load_model("base")`

