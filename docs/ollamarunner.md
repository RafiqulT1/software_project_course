# Ollama Voice Chat Documentation

## Overview
This script creates a voice-based chat interface with an LLM (Language Learning Model) using Ollama. It supports:
- **Dynamic Voice Input:** Listens until the user stops speaking.
- Speech-to-text using Whisper.
- Text-to-speech using Piper TTS.
- Chat interaction with Ollama models.
- **Emotion Detection:** Understands the user's emotion to provide empathetic responses.
- **Audio Event Detection:** Recognizes non-speech sounds like coughing and throat clearing.
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
pip install torch transformers
```

### 3. Download NLTK Data
The emotion detection feature requires data from the NLTK library. Run this command once to download it:
```powershell
python -c "import nltk; nltk.download('punkt')"
```
or install it using the python script:
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
   - Speak when prompted "Listening..."
   - The script will automatically detect when you stop talking.
   - If you cough or clear your throat, the assistant will react.
   - Wait for the assistant's voice response.
   - Ask questions about your past conversation, like "What did we talk about last time?"
   - Press Ctrl+C to exit.

## Troubleshooting

1. **Ollama Connection Error**
   - Ensure Ollama is running.
   - Check if models are downloaded using `ollama list`.

2. **Emotion Detection Errors**
   - If you see an error like `module 'emoji' has no attribute 'UNICODE_EMOJI'`, ensure you have the correct version by running `pip install emoji==1.7.0 --upgrade`.
   - If you see `Resource punkt not found`, run the NLTK download command from the setup section.

3. **Audio Recording Issues**
   - **Speech Not Detected:** If the script doesn't seem to hear you, you need to calibrate the `silence_threshold`. Run the script and watch the `Mic Level (RMS)` value. Set `silence_threshold` to a value that is higher than your background noise but lower than your speaking volume.
   - Check microphone settings in Windows.
   - Verify microphone permissions.

4. **Audio Event Detection Issues**
   - **Model Download Fails:** Ensure you have a stable internet connection. The first time you run the script, it will download a model of several hundred megabytes.
   - **Event Not Detected:** If the script doesn't react to a cough, watch the `[DEBUG] Audio Event Classified As:` output. You may need to lower the confidence threshold (e.g., from `0.7` to `0.6`) or add the detected label (e.g., 'Sneeze') to the `health_events` list in the `run()` method.

5. **TTS Issues**
   - Verify Piper model files exist in `piper_models` directory.
   - Check file paths in `_initialize_tts()`.

## Configuration
Key parameters can be adjusted directly in the `ollamarunner.py` script.

### Assistant Personality
- **`SYSTEM_PROMPT`**: Modify the `SYSTEM_PROMPT` string at the top of the `VoiceChatAssistant` class to change the LLM's behavior, personality, and instructions.

### Audio Event Detection
- **`run()` method**:
  - `health_events`: Modify this list to include other sound labels you want the assistant to react to (e.g., `['Cough', 'Throat clearing', 'Sneeze']`).
  - **Confidence Threshold**: Adjust the `event['score'] > 0.7` value to make the detection more or less sensitive.

### Voice Activity Detection (VAD)
- **`_record_audio()` method signature**:
  - `silence_threshold`: The RMS volume level that speech must cross to be detected. **This is the most important setting to calibrate for your microphone.**
  - `silence_duration_s`: How many seconds of silence are required before the recording stops.
  - `max_record_s`: A safety timeout to prevent infinitely long recordings.

### Other Settings
- **Whisper Model Size**: Change `"base"` in `self.whisper_model = whisper.load_model("base")` to other sizes like `"tiny"`, `"small"`, `"medium"`, or `"large"` depending on your hardware.
- **TTS Settings**: Modify the `SynthesisConfig` in the `_initialize_tts()` method to change the voice's speed, volume, etc.

