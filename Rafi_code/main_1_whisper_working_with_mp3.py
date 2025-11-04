import torch
import json
from datetime import timedelta
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True  # ✅ Enables timestamps for SRT/VTT/JSON export
)

# Input audio file
audio_file = "eng_female.mp3"

# Transcribe
result = pipe(audio_file, generate_kwargs={"task": "transcribe"})

# -----------------------------
# Save plain text
# -----------------------------
text = result["text"]
with open("harri_transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)

# -----------------------------
# Save JSON
# -----------------------------
with open("harri_transcript.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# -----------------------------
# Helper function for timestamp formatting
# -----------------------------
def format_timestamp(seconds: float, always_include_hours=False):
    td = timedelta(seconds=round(seconds, 3))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    hours_str = f"{hours:02d}:" if (always_include_hours or hours > 0) else ""
    return f"{hours_str}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# -----------------------------
# Save SRT
# -----------------------------
if "chunks" in result:
    segments = result["chunks"]
else:
    # Transformers pipeline returns timestamps under 'chunks' or 'segments' depending on version
    segments = result.get("segments", [])

with open("harri_transcript.srt", "w", encoding="utf-8") as srt:
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["timestamp"][0], always_include_hours=True)
        end = format_timestamp(seg["timestamp"][1], always_include_hours=True)
        text = seg["text"].strip()
        srt.write(f"{i}\n{start} --> {end}\n{text}\n\n")

# -----------------------------
# Save VTT
# -----------------------------
with open("harri_transcript.vtt", "w", encoding="utf-8") as vtt:
    vtt.write("WEBVTT\n\n")
    for seg in segments:
        start = format_timestamp(seg["timestamp"][0]).replace(",", ".")
        end = format_timestamp(seg["timestamp"][1]).replace(",", ".")
        text = seg["text"].strip()
        vtt.write(f"{start} --> {end}\n{text}\n\n")

print("✅ Transcription complete!")
print("Files saved: harri_transcript.txt, harri_transcript.json, harri_transcript.srt, harri_transcript.vtt")
