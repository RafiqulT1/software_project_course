import whisper
import text2emotion as te
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification
import numpy as np
import librosa
import ollama
import sys

class AIServices:
    def __init__(self, config):
        self.config = config
        print("Loading AI models, please wait...")
        self.whisper_model = whisper.load_model(config['models']['whisper_model_size'])
        self.audio_classifier_processor, self.audio_classifier_model = self._load_audio_classifier()
        print("AI models loaded successfully.")

    def _load_audio_classifier(self):
        try:
            model_id = self.config['models']['audio_classifier_model_id']
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForAudioClassification.from_pretrained(model_id)
            print("Audio classification model loaded.")
            return processor, model
        except Exception as e:
            print(f"\n--- CRITICAL ERROR ---")
            print(f"Could not load the audio classification model: {e}")
            print("The health event detection feature will not work.")
            print("Please check your internet connection and the model name in config.yaml.")
            sys.exit(1)

    def transcribe_audio(self, audio_file):
        try:
            audio, _ = librosa.load(audio_file, sr=16000)
            result = self.whisper_model.transcribe(audio)
            return result["text"].strip()
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def classify_audio_event(self, audio_data, sample_rate=16000):
        if self.audio_classifier_model is None or audio_data is None:
            return None
        try:
            # The audio data is now 1D, so no squeeze is needed here.
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

    def detect_emotion(self, text):
        if not text or not text.strip():
            return "Neutral"
        try:
            emotions = te.get_emotion(text)
            if emotions and any(score > 0 for score in emotions.values()):
                return max(emotions, key=emotions.get)
        except Exception as e:
            print(f"Could not detect emotion: {e}")
        return "Neutral"

    def get_llm_response(self, model, messages):
        try:
            response = ollama.chat(model=model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return "I'm sorry, I'm having trouble connecting to my brain right now."