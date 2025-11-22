import os
import sys
import glob
import logging
import yaml
import soundfile as sf
from datetime import datetime
import ollama
import numpy as np

# Import our new modules
from audio_handler import AudioHandler
from ai_services import AIServices

def load_config(path='config.yaml'):
    """Loads the YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

class VoiceChatAssistant:
    def __init__(self, config):
        self.config = config
        self.messages = []
        self.last_health_event = None
        
        self.temp_dir = self._setup_directories()
        self.logger = self._setup_logging()
        
        # Instantiate handlers
        self.audio = AudioHandler(config)
        self.ai = AIServices(config)
        
        self.ollama_model = config['models']['ollama_model']

    def _setup_directories(self):
        temp_dir = self.config['paths']['temp_dir']
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _setup_logging(self):
        logger = logging.getLogger('conversation')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.config['paths']['log_file'], mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _save_audio(self, recording):
        if recording is None: return None
        try:
            filename = os.path.join(self.temp_dir, f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            # Normalize the int16 array to a float32 array before saving.
            # This ensures the WAV file is in a standard format.
            audio_float = recording.astype(np.float32) / np.iinfo(np.int16).max
            sf.write(filename, audio_float, self.config['audio']['sample_rate'])
            return filename
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def _is_memory_question(self, text):
        return any(keyword in text.lower() for keyword in self.config['logic']['memory_keywords'])

    def _get_conversation_history(self, num_lines=50):
        log_file = self.config['paths']['log_file']
        if not os.path.exists(log_file): return "No conversation history found."
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return "".join(f.readlines()[-num_lines:])
        except Exception as e:
            print(f"Could not read conversation log: {e}")
            return "Error reading history."

    def _cleanup_recordings(self):
        for file in glob.glob(os.path.join(self.temp_dir, "*.wav")):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove old recording {file}: {e}")

    def run(self):
        print(f"Starting voice chat with model '{self.ollama_model}'. Press Ctrl+C to exit.")
        self._cleanup_recordings()

        while True:
            try:
                recording = self.audio.record_audio()
                if recording is None: continue

                # Add detailed debugging for the recording object
                duration_s = len(recording) / self.config['audio']['sample_rate']
                print(f"\n[DEBUG] Recording captured. Duration: {duration_s:.2f}s, Shape: {recording.shape}, Dtype: {recording.dtype}")

                event = self.ai.classify_audio_event(recording)
                
                # Add detailed debugging for the event object
                if event: 
                    print(f"[DEBUG] Audio Event Classified As: {event['label']} (Score: {event['score']:.2f})")
                else:
                    print("[DEBUG] Audio event classification returned None.")

                cfg_logic = self.config['logic']
                if event and event['label'] in cfg_logic['health_events'] and event['score'] > cfg_logic['health_event_confidence']:
                    self.last_health_event = event['label']
                    response = f"I noticed that {self.last_health_event}. Are you feeling alright?"
                    print(f"\nAssistant: {response}")
                    self.logger.info(f"User Event: {self.last_health_event} Detected (Score: {event['score']:.2f})")
                    self.logger.info(f"Assistant: {response}")
                    self.messages.append({'role': 'assistant', 'content': response})
                    self.audio.speak(response, self.temp_dir)
                    continue

                audio_file = self._save_audio(recording)
                if not audio_file: continue

                user_text = self.ai.transcribe_audio(audio_file)
                if not user_text:
                    print("No text was transcribed from the audio.")
                    continue
                
                print(f"\nYou said: {user_text}")
                
                emotion = self.ai.detect_emotion(user_text)
                print(f"Detected Emotion: {emotion}")
                
                prompt_to_llm = ""
                if self._is_memory_question(user_text):
                    history = self._get_conversation_history()
                    prompt_to_llm = f"Based on our recent conversation history below, please answer my question.\n\nHistory:\n{history}\n\nMy question is: {user_text}"
                    self.logger.info(f"User (Memory Question): {user_text}")
                elif self.last_health_event and emotion in cfg_logic['negative_emotions']:
                    prompt_to_llm = (
                        f"The user seems to be feeling {emotion}. A short while ago, I detected a '{self.last_health_event}'. "
                        f"Please consider this health context in your empathetic response to their message: '{user_text}'"
                    )
                    self.logger.info(f"User (Health Context): {user_text} [Emotion: {emotion}, Event: {self.last_health_event}]")
                    self.last_health_event = None
                else:
                    self.logger.info(f"User: {user_text} [Emotion: {emotion}]")
                    prompt_to_llm = f"The user seems to be feeling {emotion}. Respond to the following: {user_text}"

                full_messages = [{'role': 'system', 'content': cfg_logic['system_prompt']}, *self.messages, {'role': 'user', 'content': prompt_to_llm}]
                assistant_response = self.ai.get_llm_response(self.ollama_model, full_messages)
                
                self.messages.append({'role': 'user', 'content': user_text})
                self.messages.append({'role': 'assistant', 'content': assistant_response})
                
                print(f"\nAssistant: {assistant_response}")
                self.logger.info(f"Assistant: {assistant_response}")
                self.audio.speak(assistant_response, self.temp_dir)

            except KeyboardInterrupt:
                print("\nExiting voice chat...")
                self._cleanup_recordings()
                break
            except Exception as e:
                print(f"An unexpected error occurred in the main loop: {e}")

def check_ollama_status():
    try:
        ollama.list()
    except Exception as e:
        print(f"Ollama service not running or accessible: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_ollama_status()
    config = load_config()
    assistant = VoiceChatAssistant(config)
    assistant.run()