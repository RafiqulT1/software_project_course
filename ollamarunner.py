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
from piper import PiperVoice, SynthesisConfig
import logging
import text2emotion as te
# --- NEW IMPORTS for Sound Classification ---
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification

class VoiceChatAssistant:
    # --- LLM INSTRUCTIONS ---
    # Modify this string to change the assistant's personality and behavior.
    SYSTEM_PROMPT = (
        "You are a helpful, friendly, and empathetic voice assistant. "
        "Keep your answers concise and to the point. Don't use any special formatting such as asterix etc."
        "Analyze the user's emotion and respond in an appropriate and supportive manner."
        "You will mostly engage with elder people."
        "Use only one sentence for your answers" #to make answers short for debugging
    )

    def __init__(self, ollama_model="llama2"):
        """
        Initializes the Voice Chat Assistant.
        """
        self.ollama_model = ollama_model
        self.messages = []
        self.temp_dir = self._setup_directories()
        self.logger = self._setup_logging()
        
        # --- NEW: State tracking for health checks ---
        self.health_check_pending = False
        self.last_health_event = None
        
        print("Loading models, please wait... (This may take a moment)")
        self.whisper_model = whisper.load_model("base")
        self.tts_engine = self._initialize_tts()
        
        # --- NEW: Load Audio Classification Model ---
        print("Loading audio classification model...")
        try:
            # FIX: Corrected the Hugging Face model identifier to the canonical name.
            model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self.audio_classifier_processor = AutoProcessor.from_pretrained(model_id)
            self.audio_classifier_model = AutoModelForAudioClassification.from_pretrained(model_id)
            print("Audio classification model loaded.")
        except Exception as e:
            print(f"Could not load audio classification model: {e}")
            self.audio_classifier_model = None

        print("Models loaded successfully.")

    def _setup_directories(self):
        """Creates the temporary directory for audio files."""
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _setup_logging(self):
        """Sets up the conversation logger."""
        logger = logging.getLogger('conversation')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('log.txt', mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _initialize_tts(self):
        """Initializes the Piper TTS engine."""
        try:
            print("[DEBUG] Initializing Piper TTS...")
            voice = PiperVoice.load("piper_models/fi_FI-harri-medium.onnx")
            syn_config = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, volume=1.0, normalize_audio=True)
            print("[DEBUG] Piper TTS initialized successfully")
            return voice, syn_config
        except Exception as e:
            print(f"[DEBUG] Error initializing Piper TTS: {e}")
            return None, None

    def _record_audio(self, sample_rate=16000, chunk_size=1024, silence_threshold=40, silence_duration_s=2, max_record_s=15):
        """
        Records audio from the microphone until a period of silence is detected.
        """
        print("Listening... (waiting for speech)")
        
        recorded_frames = []
        is_speaking = False
        silent_chunks = 0
        num_silent_chunks_to_stop = int((silence_duration_s * sample_rate) / chunk_size)
        max_chunks = int((max_record_s * sample_rate) / chunk_size)

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16, blocksize=chunk_size) as stream:
                for i in range(max_chunks):
                    audio_chunk, overflowed = stream.read(chunk_size)
                    if overflowed:
                        print("Warning: Audio buffer overflowed")

                    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                    print(f"\rMic Level (RMS): {rms:.2f}", end="")

                    if rms > silence_threshold:
                        if not is_speaking:
                            print("\r" + " " * 30 + "\r", end="") 
                            print("Speech detected, recording...")
                            is_speaking = True
                        silent_chunks = 0
                        recorded_frames.append(audio_chunk)
                    elif is_speaking:
                        silent_chunks += 1
                        recorded_frames.append(audio_chunk)
                        if silent_chunks > num_silent_chunks_to_stop:
                            print("\r" + " " * 30 + "\r", end="")
                            print("Silence detected, finishing recording.")
                            break
                
                print("\r" + " " * 30 + "\r", end="")

                if not recorded_frames:
                    print("No speech detected within the time limit.")
                    return None

                print("Recording finished")
                return np.concatenate(recorded_frames, axis=0)

        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    # --- NEW METHOD for Sound Classification ---
    def _classify_audio_event(self, audio_data, sample_rate=16000):
        """Classifies the primary sound event in an audio clip."""
        if self.audio_classifier_model is None or audio_data is None:
            return None
        try:
            # FIX: Squeeze the 2D audio array (n_samples, 1) into a 1D array (n_samples,)
            # This is required by the audio classification model.
            if audio_data.ndim > 1:
                audio_data = np.squeeze(audio_data)

            # Normalize audio from int16 to float32 for the model
            audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            
            inputs = self.audio_classifier_processor(audio_float, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits = self.audio_classifier_model(**inputs).logits
            
            predicted_class_ids = torch.argmax(logits, dim=-1).item()
            predicted_label = self.audio_classifier_model.config.id2label[predicted_class_ids]

            scores = torch.nn.functional.softmax(logits, dim=-1)
            predicted_score = scores[0][predicted_class_ids].item()

            return {"label": predicted_label, "score": predicted_score}
        except Exception as e:
            print(f"Could not classify audio event: {e}")
            return None

    def _save_audio(self, recording, sample_rate=16000):
        """Saves the recorded audio to a temporary WAV file."""
        if recording is None:
            print("Error: No recording data to save")
            return None
        try:
            filename = os.path.join(self.temp_dir, f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            audio_float = recording.astype(np.float32) / np.iinfo(np.int16).max
            sf.write(filename, audio_float, sample_rate)
            if os.path.exists(filename):
                return os.path.abspath(filename)
        except Exception as e:
            print(f"Error saving audio: {e}")
        return None

    def _speech_to_text(self, audio_file):
        """Transcribes audio file to text using Whisper."""
        try:
            audio, _ = librosa.load(audio_file, sr=16000)
            result = self.whisper_model.transcribe(audio)
            return result["text"].strip()
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def _speak(self, text):
        """Synthesizes and speaks the given text."""
        voice, syn_config = self.tts_engine
        if not voice:
            print("[DEBUG] TTS engine not available.")
            return
        try:
            temp_wav = os.path.join(self.temp_dir, f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            with wave.open(temp_wav, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            data, sample_rate = sf.read(temp_wav)
            sd.play(data, sample_rate, blocking=True)
            os.remove(temp_wav)
        except Exception as e:
            print(f"[DEBUG] Error in text-to-speech: {e}")

    def _detect_emotion(self, text):
        """Detects the dominant emotion from text."""
        if not text or not text.strip():
            return "Neutral"
        try:
            emotions = te.get_emotion(text)
            if emotions and any(score > 0 for score in emotions.values()):
                return max(emotions, key=emotions.get)
        except Exception as e:
            print(f"Could not detect emotion: {e}")
        return "Neutral"

    def _is_memory_question(self, text):
        """Checks if the text is a question about past conversations."""
        memory_keywords = ["remember", "what did we talk about", "last time", "yesterday", "previously"]
        return any(keyword in text.lower() for keyword in memory_keywords)

    def _get_conversation_history(self, num_lines=50):
        """Retrieves recent conversation history from the log."""
        log_file_path = 'log.txt'
        if not os.path.exists(log_file_path):
            return "No conversation history found."
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return "".join(lines[-num_lines:])
        except Exception as e:
            print(f"Could not read conversation log: {e}")
            return "Error reading history."

    def _cleanup_recordings(self):
        """Cleans up old temporary audio files."""
        for file in glob.glob(os.path.join(self.temp_dir, "*.wav")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove old recording {file}: {e}")

    def run(self):
        """Starts the main conversation loop."""
        print(f"Starting voice chat with model '{self.ollama_model}'. Press Ctrl+C to exit.")
        self._cleanup_recordings()

        while True:
            try:
                print("\nListening...")
                recording = self._record_audio()
                if recording is None:
                    continue

                # --- Audio Event Classification Logic ---
                event = self._classify_audio_event(recording)

                # --- DEBUGGING: Print the classification result every time ---
                if event:
                    print(f"\n[DEBUG] Audio Event Classified As: {event['label']} (Score: {event['score']:.2f})")
                else:
                    print("\n[DEBUG] Audio event could not be classified.")

                # --- NEW: Define a list of health-related events to react to ---
                health_events = ['Cough', 'Throat clearing']
                
                # Check if the detected event is in our list and meets the confidence threshold.
                if event and event['label'] in health_events and event['score'] > 0.7:
                    print(f"Health Event Detected: {event['label']} (Score: {event['score']:.2f})")
                    self.logger.info(f"User Event: {event['label']} Detected (Score: {event['score']:.2f})")
                    
                    # --- NEW: Set state before asking the question ---
                    self.health_check_pending = True
                    self.last_health_event = event['label']
                    
                    assistant_response = f"I noticed that you had a {self.last_health_event}. Are you feeling alright?"
                    
                    # Log the assistant's question to the console and file
                    print(f"\nAssistant: {assistant_response}")
                    self.logger.info(f"Assistant: {assistant_response}")
                    
                    # Add the assistant's question to the message history for LLM context
                    self.messages.append({'role': 'assistant', 'content': assistant_response})
                    
                    # Speak the response
                    self._speak(assistant_response)
                    
                    continue # Skip the rest of the loop and listen for the user's answer

                audio_file = self._save_audio(recording)
                if not audio_file:
                    continue

                user_text = self._speech_to_text(audio_file)
                if not user_text:
                    print("No text was transcribed from the audio.")
                    continue
                
                print(f"\nYou said: {user_text}")
                
                prompt_to_llm = ""
                
                # --- NEW: Check for a pending health response first ---
                if self.health_check_pending:
                    print("Received response to health check.")
                    emotion = self._detect_emotion(user_text)
                    prompt_to_llm = (
                        f"I previously detected a '{self.last_health_event}' and asked the user if they were feeling alright. "
                        f"They now seem to be feeling {emotion} and have responded. "
                        f"Based on this health context, reply to their message: '{user_text}'"
                        f"Mention the health event '{self.last_health_event}' in your response."
                    )
                    self.logger.info(f"User (Health Response): {user_text} [Emotion: {emotion}]")
                    # Reset the state after using it
                    self.health_check_pending = False
                    self.last_health_event = None
                
                elif self._is_memory_question(user_text):
                    print("Memory question detected. Checking logs...")
                    history = self._get_conversation_history()
                    prompt_to_llm = f"Based on our recent conversation history below, please answer my question.\n\nHistory:\n{history}\n\nMy question is: {user_text}"
                    self.logger.info(f"User (Memory Question): {user_text}")
                else:
                    emotion = self._detect_emotion(user_text)
                    print(f"Detected Emotion: {emotion}")
                    self.logger.info(f"User: {user_text} [Emotion: {emotion}]")
                    prompt_to_llm = f"The user seems to be feeling {emotion}. Respond to the following: {user_text}"

                full_messages = [
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    *self.messages,
                    {'role': 'user', 'content': prompt_to_llm}
                ]

                response = ollama.chat(
                    model=self.ollama_model,
                    messages=full_messages
                )
                assistant_response = response['message']['content']
                
                self.messages.append({'role': 'user', 'content': user_text})
                self.messages.append({'role': 'assistant', 'content': assistant_response})
                
                print(f"\nAssistant: {assistant_response}")
                self.logger.info(f"Assistant: {assistant_response}")
                
                self._speak(assistant_response)

            except KeyboardInterrupt:
                print("\nExiting voice chat...")
                self._cleanup_recordings()
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

def check_ollama_status():
    """Checks if the Ollama service is running and models are available."""
    try:
        response = ollama.list()
        if not response.get('models'):
            print("No Ollama models found. Please run 'ollama pull <model_name>'")
            sys.exit(1)
        return response['models']
    except Exception as e:
        print(f"Ollama service not running or accessible: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Checking Ollama status...")
    available_models = check_ollama_status()

    print("\n--- Ollama Model Data ---")
    print(available_models)
    print("-------------------------\n")

    if not available_models:
        print("Error: No Ollama models found. Please run 'ollama pull <model_name>'")
        sys.exit(1)

    try:
        default_model = available_models[0].model
        model_names = [model.model for model in available_models]
        print(f"Available models: {model_names}")
    except AttributeError:
        print("Error: Could not find the '.model' attribute in the model data.")
        print("Please check the debug output above to see the actual structure of the model data from your Ollama instance.")
        sys.exit(1)
    except IndexError:
        print("Error: The list of available models is empty.")
        sys.exit(1)
    
    assistant = VoiceChatAssistant(ollama_model=default_model)
    assistant.run()