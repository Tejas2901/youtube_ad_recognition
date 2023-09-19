import whisper

model = whisper.load_model("base")
result = model.transcribe("videoplayback.mp3")
print(result["text"])