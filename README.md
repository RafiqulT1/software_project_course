# Software_project_course
# Finnish Spoken Robot Dialogue Documentation (Ollama based voice assistant)
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
pip install ollama sounddevice numpy openai-whisper soundfile librosa piper-tts text2emotion "emoji==1.7.0" torch transformers PyYAML
```

### 3. Download NLTK Data
The emotion detection feature requires data from the NLTK library. Run this command once to download it:
```powershell
python -c "import nltk; nltk.download('punkt')"
```
Alternatively, you can run the provided helper script:
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
├── temp_audio/              # Created automatically
├── log.txt                  # Created automatically
├── config.yaml              # Main configuration file
├── ollamarunner.py          # Main application logic
├── audio_handler.py         # Handles recording and TTS
├── ai_services.py           # Handles all AI model interactions
└── download_nltk_data.py    # Helper script
```

## Usage

1. Make sure Ollama is running (check system tray).
2. Calibrate your microphone settings in `config.yaml`.
3. Activate virtual environment: `.\venv\Scripts\Activate.ps1`
4. Run the script: `python ollamarunner.py`
5. Interact with the chat:
   - Speak when prompted "Listening..."
   - The script will automatically detect when you stop talking.
   - If you cough or clear your throat, the assistant will react.
   - Press Ctrl+C to exit.

## Configuration
All settings are controlled in the **`config.yaml`** file.

### Audio Settings (`audio:`)
- `silence_threshold`: The RMS volume level that speech must cross to be detected. **This is the most important setting to calibrate for your microphone.**
- `silence_duration_s`: How many seconds of silence are required before the recording stops.
- `min_record_s`: The minimum recording length in seconds. Crucial for ensuring short sounds like coughs are fully captured for analysis.
- `max_record_s`: A safety timeout to prevent infinitely long recordings.

### AI Model Settings (`models:`)
- `ollama_model`: The default Ollama model to use (e.g., "llama2").
- `whisper_model_size`: Can be "tiny", "base", "small", "medium", or "large".
- `audio_classifier_model_id`: The Hugging Face model for sound detection.

### Logic and Behavior (`logic:`)
- `system_prompt`: A multi-line string that defines the assistant's personality and instructions.
- `health_events`: A list of sound labels (from the audio classifier) that the assistant should react to.
- `negative_emotions`: A list of emotions that, when detected, can trigger a contextual response related to a recent health event.
- `health_event_confidence`: The minimum confidence score (0.0 to 1.0) required for the assistant to react to a detected health event.

## Troubleshooting

1. **Ollama Connection Error**
   - Ensure Ollama is running.
   - Check if models are downloaded using `ollama list`.

2. **Speech Not Detected / Coughs Ignored**
   - This is a calibration issue. Run the script and watch the `Mic Level (RMS)` value.
   - **To detect quiet speech:** Lower the `silence_threshold` in `config.yaml`.
   - **To detect short coughs:** Ensure `min_record_s` is set to at least `1.5` in `config.yaml`.

3. **Health Event Detected but Not Acted Upon**
   - The model's confidence score was too low. Watch the `[DEBUG] Audio Event...` output and lower the `health_event_confidence` value in `config.yaml` to be below the observed score.

4. **Model Download Fails**
   - Ensure you have a stable internet connection. The first time you run the script, it will download several hundred megabytes for the Whisper and audio classification models.




### First steps
- Telegram(?) chat group with me to direct communication
- Timeline and gates (deadlines)
- Roles in the group
- Found a github repository for:
  - Code
  - Documentation
- Weekly status on Friday of (to telegram):
- What has been done
- Whats up next week
- Risk analysis
- Log your work
- Hours, work done, who did, etc.
- Get familiar with tech
- State of the Art survey
- Requirement specification and review with us
- We will provide  data when plan is good



**Work logs**
| Date  | Members | Hours | Task |
| ------------- | ------------- | ------------- | ------------- |
| 1.10.2025 | Rafiqul | 1h | Created Discord channel|
| 03.10.2025 | M, R, V | 0.5h | Had a discussion about this project |
| 5.10.2025 | Rafiqul | 1h | Created git repo and shared to all members |
| 7.10.2025 | Rafiqul | 2h | Gathering knowledge about the project subject
| 10.10.2025 | M, R, V | 1h | discussion about project and presentation |
| 15.10.2025 | Valtteri | 1h | Writing presentation slides |
| 15.10.2025 | Mikael | 4h | Writing presentation slides and getting familiar with local LLMs ollama/llama.cpp |
| 17.10.2025 | Mikael | 2h | Working on llama.cpp |
| 17.10.2025 | M, R, V | 1h |Had a meeting about presentation and getting familier with hugging face |
| 20.10.2025 | R, V | 2.5h | testing hugging face tools and models |
| 21.10.2025 | R, V | 3h | testing hugging face modules and testing model AL model chaining thorugh python |
| 22.10.2025 | R, V | 3h | tested whisper AI model with audio files (success), and made the mindmap / AI neural map |
| 25.10.2025 | R    | 1.5h | Prepping of the midterm presentation | 
| 26.10.2025 | R, V | 3h | Finished writing the midterm presentation |
| 27.10.2025 | R, V | 2h | Midterm presentation                      |
| 03.11.2025 | R | 3h | Worked on whisper & bart AI wrote python code (working with realtime voice input) |
| 04.11.2025 | M, R, V | 0.5h | Sync call |
| 04.11.2025 | M | 2h | Working on python script (ollamarunner.py) |
| 05.11.2025 | M, R, V | 1h | Sync call |
| 05.11.2025 | M | 2h | Working on python script (ollamarunner.py) |
| 22.11.2025 | M | 4h | Working on python script (ollamarunner.py and related) |
| 09.12.2025 | R,V | 0.5h | Try to find out exact reason of Jetson output not showing to the display |
| 10.12.2025 | V | 2h | went to buy USB-C to Displayport cable for Jetson | 
| 10.12.2025 | V,R | 3h | working on Jetson: trying run the code |
| 11.12.2025 | R | 2h | working on Jetson |
| 11.12.2025 | M, R, V | 1h | online meeting about Jetson and project implementation, testing and project report |
| 11.12.2025 | V | 5h | fixing issues with various module version and testing the code execution and AI model |
| 13.12.2025 | M, R, V | 1h | online meeting about project report and writing |
| 15.12.2025 | R | 3h | Writing project report |
| 16.12.2025 | R | 1h | Writing project report |
| 16.12.2025 | M, R, V | 1h | Online meeting about project report and test planning |
| 18.12.2025 | R | 4h | Writing project report |
| 18.12.2025 | M, R, V | 2h | Online meeting about project report, preparing presentation, record presentation |
| 23.12.2025 | R | 1.5h | Final editing of project report and submission |


