import ollama
import sys
import sounddevice as sd
import numpy as np
import wave
import whisper
from datetime import datetime
import os
import glob
import soundfile as sf
import librosa
import win32com.client

# Create temp directory for audio files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

def check_ollama_status():
    """
    Check if Ollama is running and has models installed
    """
    try:
        response = ollama.list()
        models = response.get('models', [])
        if not models:
            print("No models found. Please install a model using 'ollama pull modelname'")
            print("Example: ollama pull llama2")
            sys.exit(1)
        return models
    except ConnectionError:
        print("Error: Cannot connect to Ollama. Please make sure it's running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def record_audio(duration=5, sample_rate=16000):
    """
    Record audio from microphone
    """
    print("Recording... Speak now")
    try:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        print("Recording finished")
        
        # Verify recording data
        if np.any(recording):
            print(f"Recording shape: {recording.shape}, dtype: {recording.dtype}")
            return recording
        else:
            print("Warning: Recording appears to be empty")
            return None
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        return None

def cleanup_old_recordings():
    """
    Clean up old WAV files from temp directory
    """
    for file in glob.glob(os.path.join(TEMP_DIR, "temp_recording_*.wav")):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Warning: Could not remove old recording {file}: {e}")

def save_audio(recording, sample_rate=16000):
    """
    Save recording to WAV file with error checking
    """
    if recording is None:
        print("Error: No recording data to save")
        return None
        
    try:
        filename = os.path.join(TEMP_DIR, f"temp_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        print(f"Saving audio to: {filename}")
        
        # Convert to float32 and normalize
        audio_float = recording.astype(np.float32) / np.iinfo(np.int16).max
        
        # Save using soundfile
        sf.write(filename, audio_float, sample_rate)
            
        # Verify file was created
        if os.path.exists(filename):
            print(f"Audio file saved successfully ({os.path.getsize(filename)} bytes)")
            return os.path.abspath(filename)  # Return absolute path
        else:
            print("Error: File was not created")
            return None
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return None

def speech_to_text(audio_file, whisper_model):
    """
    Convert speech to text using Whisper
    """
    try:
        # Load and resample audio
        audio, _ = librosa.load(audio_file, sr=16000)
        
        # Transcribe using loaded model
        result = whisper_model.transcribe(audio)
        return result["text"].strip()
    except Exception as e:
        print(f"Error in speech recognition: {str(e)}")
        return None

def initialize_tts():
    """
    Initialize Windows SAPI text-to-speech engine
    """
    try:
        print("[DEBUG] Initializing Windows SAPI...")
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        # Optional: Adjust voice settings
        speaker.Volume = 100  # 0 to 100
        speaker.Rate = 0     # -10 to 10
        print("[DEBUG] SAPI initialized successfully")
        return speaker
    except Exception as e:
        print(f"[DEBUG] Error initializing SAPI: {str(e)}")
        return None

def speak_text(speaker, text):
    """
    Convert text to speech using Windows SAPI (synchronous)
    """
    print(f"[DEBUG] Starting speak_text with SAPI: {speaker is not None}")
    
    if speaker is None:
        print("[DEBUG] SAPI is None, reinitializing...")
        speaker = initialize_tts()
    
    try:
        print("[DEBUG] Attempting to speak...")
        print(f"[DEBUG] Text length: {len(text)} characters")
        speaker.Speak(text)  # Removed the async flag (1)
        print("[DEBUG] Speech completed")
        return speaker
    except Exception as e:
        print(f"[DEBUG] Error in text-to-speech: {str(e)}")
        print("[DEBUG] Attempting to reinitialize SAPI...")
        return initialize_tts()

def voice_chat_conversation(model="llama2"):
    """
    Start an interactive voice chat session with the model
    """
    print(f"Starting voice chat with {model}")
    print("Press Ctrl+C to exit")
    messages = []
    
    print("[DEBUG] Setting up initial TTS engine")
    tts_engine = initialize_tts()
    print(f"[DEBUG] Initial TTS engine state: {tts_engine is not None}")
    
    # Cleanup old recordings and load Whisper model
    cleanup_old_recordings()
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded!")
    
    while True:
        try:
            print("\nListening...")
            recording = record_audio()
            if recording is None:
                continue
                
            audio_file = save_audio(recording)
            if audio_file is None:
                continue
            
            try:
                print(f"Transcribing audio file: {audio_file}")
                text_input = speech_to_text(audio_file, whisper_model)
                
                if text_input:
                    print(f"\nYou said: {text_input}")
                    
                    response = ollama.chat(
                        model=model,
                        messages=[*messages, {'role': 'user', 'content': text_input}]
                    )
                    assistant_response = response['message']['content']
                    messages.append({'role': 'user', 'content': text_input})
                    messages.append({'role': 'assistant', 'content': assistant_response})
                    
                    print(f"\nAssistant: {assistant_response}")
                    print(f"[DEBUG] Current TTS engine state before speaking: {tts_engine is not None}")
                    tts_engine = speak_text(tts_engine, assistant_response)
                    print(f"[DEBUG] TTS engine state after speaking: {tts_engine is not None}")
                else:
                    print("No text was transcribed from the audio")
                    
            except Exception as e:
                print(f"Error in transcription: {str(e)}")
                print(f"Audio file path: {os.path.abspath(audio_file)}")
                print(f"File exists: {os.path.exists(audio_file)}")
                print(f"File size: {os.path.getsize(audio_file) if os.path.exists(audio_file) else 'N/A'}")
            
        except KeyboardInterrupt:
            print("\nExiting voice chat...")
            cleanup_old_recordings()
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Check Ollama status and get available models
    available_models = check_ollama_status()
    default_model = available_models[0].model
    print(f"Available models: {[model.model for model in available_models]}")
    
    # Start voice chat
    voice_chat_conversation(model=default_model)