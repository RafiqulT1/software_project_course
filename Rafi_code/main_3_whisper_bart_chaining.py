import os
import torch
import json
from datetime import datetime, timedelta
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
audio_file = "eng_female.mp3"  # 🎧 Change this to any audio filename

# Output directories
output_dir = "output"
summary_dir = "summarized"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# Base name extraction (e.g., harri.mp3 → harri)
base_name = os.path.splitext(os.path.basename(audio_file))[0]

# Output file paths
txt_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
srt_path = os.path.join(output_dir, f"{base_name}_transcript.srt")
vtt_path = os.path.join(output_dir, f"{base_name}_transcript.vtt")

# -----------------------------
# STAGE 1: SPEECH-TO-TEXT (Whisper)
# -----------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

print(f"Loading Whisper model on {device}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
    return_timestamps=True,
)

print(f"Transcribing {audio_file}...")
result = pipe(audio_file, generate_kwargs={"task": "transcribe"})

# Save text + JSON
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Helper function for time formatting
def format_timestamp(seconds: float, always_include_hours=False):
    td = timedelta(seconds=round(seconds, 3))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    hours_str = f"{hours:02d}:" if (always_include_hours or hours > 0) else ""
    return f"{hours_str}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

segments = result.get("chunks") or result.get("segments", [])

# Save SRT
with open(srt_path, "w", encoding="utf-8") as srt:
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["timestamp"][0], always_include_hours=True)
        end = format_timestamp(seg["timestamp"][1], always_include_hours=True)
        srt.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")

# Save VTT
with open(vtt_path, "w", encoding="utf-8") as vtt:
    vtt.write("WEBVTT\n\n")
    for seg in segments:
        start = format_timestamp(seg["timestamp"][0]).replace(",", ".")
        end = format_timestamp(seg["timestamp"][1]).replace(",", ".")
        vtt.write(f"{start} --> {end}\n{seg['text'].strip()}\n\n")

print(f"✅ Transcription complete! Files saved in '{output_dir}'")

# -----------------------------
# STAGE 2: SUMMARIZATION (BART)
# -----------------------------
print("Loading summarization model (facebook/bart-large-cnn)...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Read the transcribed text
with open(txt_path, "r", encoding="utf-8") as f:
    text_to_summarize = f.read().strip()

# Break long texts into smaller parts (BART limit ~1024 tokens)
max_chunk_size = 2000  # chars, adjustable
chunks = [text_to_summarize[i:i + max_chunk_size] for i in range(0, len(text_to_summarize), max_chunk_size)]

summary = ""
for chunk in chunks:
    summary_chunk = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    summary += summary_chunk + "\n"

# Timestamp for output filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_filename = os.path.join(summary_dir, f"{base_name}_summary_{timestamp}.txt")

# Save the summary
with open(summary_filename, "w", encoding="utf-8") as f:
    f.write(summary.strip())

print(f"🧠 Summary saved to: {summary_filename}")
