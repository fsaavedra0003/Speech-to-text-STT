import whisper
from pydub import AudioSegment
import os

def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio(file_path):
    model = whisper.load_model("base")  # or "medium", "large"
    result = model.transcribe(file_path)
    return result["text"]


#This is the path for the audio that will be transcribed
if __name__ == "__main__":
    mp3_path = "audio_samples/example.mp3"
    wav_path = "audio_samples/example.wav"

    mp3_to_wav(mp3_path, wav_path)
    text = transcribe_audio(wav_path)

    with open("transcriptions/example.txt", "w") as f:
        f.write(text)
