import os
import json
import torch
import numpy as np
from datetime import datetime, timedelta
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# -----------------------------
# CONFIGURATION
# -----------------------------
RECORD_SECONDS = 15            # length of each mic chunk
SAMPLE_RATE = 16000            # Whisper expects 16 kHz audio
OUTPUT_DIR = "output"
SUMMARY_DIR = "summarized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading Whisper model...")
whisper_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    whisper_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(whisper_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
    return_timestamps=True,
)

print("Loading summarization model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model=bart_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
from datetime import timedelta
import json
import os

def format_timestamp(seconds: float, always_include_hours=False):
    """
    Safe formatter. If `seconds` is None, returns '00:00:00,000' (or '00:00,000' depending on hours).
    """
    if seconds is None:
        # Return a zero timestamp placeholder
        if always_include_hours:
            return "00:00:00,000"
        else:
            return "00:00,000"
    # round to milliseconds
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // (1000*60)) % 60
    h = total_ms // (1000*60*60)
    if always_include_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return f"{m:02d}:{s:02d},{ms:03d}"


def save_transcripts(result, base_name, output_dir="output"):
    """
    Save txt/json/srt/vtt for a single transcription `result`.
    Handles segments in either:
      - seg["timestamp"] == [start, end]
      - seg["start"], seg["end"]
    Fallback rules:
      - if end is None -> end = start + fallback_duration
      - if start is None -> start = 0
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_path  = os.path.join(output_dir, f"{base_name}_{timestamp_now}.txt")
    json_path = os.path.join(output_dir, f"{base_name}_{timestamp_now}.json")
    srt_path  = os.path.join(output_dir, f"{base_name}_{timestamp_now}.srt")
    vtt_path  = os.path.join(output_dir, f"{base_name}_{timestamp_now}.vtt")

    # save text + json
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result.get("text", ""))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # get segments robustly
    segments = result.get("chunks") or result.get("segments") or []
    # fallback duration when end is missing (in seconds)
    fallback_duration = 0.75

    # write SRT
    with open(srt_path, "w", encoding="utf-8") as srt:
        for i, seg in enumerate(segments, start=1):
            # extract start/end robustly
            start = None
            end = None
            if "timestamp" in seg and isinstance(seg["timestamp"], (list, tuple)) and len(seg["timestamp"]) >= 2:
                start, end = seg["timestamp"][0], seg["timestamp"][1]
            else:
                # try start/end keys
                start = seg.get("start")
                end = seg.get("end")

            if start is None and end is None:
                # nothing to do; skip or set zeros
                start = 0.0
                end = fallback_duration
            elif start is None and end is not None:
                # set start a bit before end
                start = max(0.0, end - fallback_duration)
            elif end is None and start is not None:
                end = start + fallback_duration

            # ensure numeric
            try:
                start_f = float(start)
            except Exception:
                start_f = 0.0
            try:
                end_f = float(end)
            except Exception:
                end_f = start_f + fallback_duration

            start_ts = format_timestamp(start_f, always_include_hours=True)
            end_ts = format_timestamp(end_f, always_include_hours=True)
            text = seg.get("text", "").strip()
            srt.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")

    # write VTT
    with open(vtt_path, "w", encoding="utf-8") as vtt:
        vtt.write("WEBVTT\n\n")
        for seg in segments:
            start = None
            end = None
            if "timestamp" in seg and isinstance(seg["timestamp"], (list, tuple)) and len(seg["timestamp"]) >= 2:
                start, end = seg["timestamp"][0], seg["timestamp"][1]
            else:
                start = seg.get("start")
                end = seg.get("end")

            if start is None and end is None:
                start = 0.0
                end = fallback_duration
            elif start is None and end is not None:
                start = max(0.0, end - fallback_duration)
            elif end is None and start is not None:
                end = start + fallback_duration

            try:
                start_f = float(start)
            except Exception:
                start_f = 0.0
            try:
                end_f = float(end)
            except Exception:
                end_f = start_f + fallback_duration

            start_ts = format_timestamp(start_f).replace(",", ".")
            end_ts = format_timestamp(end_f).replace(",", ".")
            text = seg.get("text", "").strip()
            vtt.write(f"{start_ts} --> {end_ts}\n{text}\n\n")

    print(f"✅ Saved chunk transcription: {txt_path}")
    return txt_path


def summarize_text(full_text, base_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_file = os.path.join(SUMMARY_DIR, f"{base_name}_summary_{timestamp}.txt")

    max_chunk_size = 2000
    chunks = [full_text[i:i + max_chunk_size] for i in range(0, len(full_text), max_chunk_size)]
    summary = ""
    for chunk in chunks:
        s = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        summary += s + "\n"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary.strip())

    print(f"🧠 Summary saved to {summary_file}")


# -----------------------------
# MAIN REAL-TIME FUNCTION
# -----------------------------
def record_and_transcribe():
    print("🎙️  Starting live recording. Press Ctrl+C to stop.")
    base_name = "realtime_audio"
    collected_text = ""

    try:
        while True:
            print(f"\n⏺️  Recording {RECORD_SECONDS}s chunk...")
            audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            wav_file = os.path.join(OUTPUT_DIR, f"{base_name}_temp.wav")
            write(wav_file, SAMPLE_RATE, (audio * 32767).astype(np.int16))

            print("🔍 Transcribing...")
            result = whisper_pipe(wav_file, generate_kwargs={"task": "transcribe"})
            collected_text += " " + result["text"]
            save_transcripts(result, base_name)

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped.")
        print("Generating summary...")
        summarize_text(collected_text, base_name)


if __name__ == "__main__":
    record_and_transcribe()
