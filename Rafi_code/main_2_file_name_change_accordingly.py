import os
import torch
import json
from datetime import timedelta
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Input audio file (change this to any file name)
audio_file = "eng_female.mp3"  # Example: replace with "meeting_audio.mp3"

# 🧠 Automatically generate file names based on audio file name
base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

txt_path  = os.path.join(output_dir, f"{base_name}_transcript.txt")
json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
srt_path  = os.path.join(output_dir, f"{base_name}_transcript.srt")
vtt_path  = os.path.join(output_dir, f"{base_name}_transcript.vtt")

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

# Load model and processor
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

# Transcribe
result = pipe(audio_file, generate_kwargs={"task": "transcribe"})

# Save text
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(result["text"])

# Save JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Timestamp formatter
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

print(f"✅ Transcription complete!\nFiles saved in '{output_dir}' as:")
print(f" - {os.path.basename(txt_path)}")
print(f" - {os.path.basename(json_path)}")
print(f" - {os.path.basename(srt_path)}")
print(f" - {os.path.basename(vtt_path)}")
