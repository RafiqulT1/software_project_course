import sounddevice as sd
import numpy as np
import wave
import soundfile as sf
from piper import PiperVoice, SynthesisConfig
import os

class AudioHandler:
    def __init__(self, config):
        self.config = config
        self.tts_engine = self._initialize_tts()

    def _initialize_tts(self):
        try:
            print("[DEBUG] Initializing Piper TTS...")
            voice = PiperVoice.load(self.config['paths']['piper_model'])
            syn_config = SynthesisConfig(length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, volume=1.0, normalize_audio=True)
            print("[DEBUG] Piper TTS initialized successfully")
            return voice, syn_config
        except Exception as e:
            print(f"[DEBUG] Error initializing Piper TTS: {e}")
            return None, None

    def record_audio(self):
        print("Listening... (waiting for speech)")
        cfg = self.config['audio']
        min_record_s = cfg.get('min_record_s', 1.0) # Default to 1.0s if not in config

        recorded_frames = []
        is_speaking = False
        silent_chunks = 0
        
        num_silent_chunks_to_stop = int((cfg['silence_duration_s'] * cfg['sample_rate']) / cfg['chunk_size'])
        min_chunks_to_record = int((min_record_s * cfg['sample_rate']) / cfg['chunk_size'])
        max_chunks = int((cfg['max_record_s'] * cfg['sample_rate']) / cfg['chunk_size'])

        try:
            with sd.InputStream(samplerate=cfg['sample_rate'], channels=1, dtype=np.int16, blocksize=cfg['chunk_size']) as stream:
                for _ in range(max_chunks):
                    audio_chunk, overflowed = stream.read(cfg['chunk_size'])
                    if overflowed: print("Warning: Audio buffer overflowed")
                    
                    rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
                    print(f"\rMic Level (RMS): {rms:.2f}", end="")

                    if rms > cfg['silence_threshold']:
                        if not is_speaking:
                            print("\r" + " " * 30 + "\r", end="") 
                            print("Speech detected, recording...")
                            is_speaking = True
                        silent_chunks = 0
                        recorded_frames.append(audio_chunk)
                    elif is_speaking:
                        silent_chunks += 1
                        recorded_frames.append(audio_chunk)
                        if silent_chunks > num_silent_chunks_to_stop and len(recorded_frames) > min_chunks_to_record:
                            print("\r" + " " * 30 + "\r", end="")
                            print("Silence detected, finishing recording.")
                            break
                
                print("\r" + " " * 30 + "\r", end="")
                if not recorded_frames:
                    print("No speech detected within the time limit.")
                    return None
                
                print("Recording finished")
                # Concatenate all the recorded frames into a single audio clip.
                recording = np.concatenate(recorded_frames, axis=0)
                # Squeeze the 2D array (n_samples, 1) into a 1D array (n_samples,).
                # This is the standard format required by the AI models.
                return np.squeeze(recording)
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def speak(self, text, temp_dir):
        voice, syn_config = self.tts_engine
        if not voice:
            print("[DEBUG] TTS engine not available.")
            return
        try:
            temp_wav = os.path.join(temp_dir, f"speech_{os.urandom(4).hex()}.wav")
            with wave.open(temp_wav, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            data, sample_rate = sf.read(temp_wav)
            sd.play(data, sample_rate, blocking=True)
            os.remove(temp_wav)
        except Exception as e:
            print(f"[DEBUG] Error in text-to-speech: {e}")