import os
import wave
import asyncio
import tempfile
import subprocess
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class AudioTranscriber:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not set in .env or environment variables!")
        self.client = Groq(api_key=self.api_key)
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']

    def _convert_to_supported_format(self, input_file: str) -> str:
        ext = os.path.splitext(input_file)[1].lower()
        if ext in self.supported_formats:
            return input_file

        output_file = tempfile.mktemp(suffix='.wav')
        subprocess.run([
            'ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', '-y', output_file
        ], check=True, capture_output=True)
        return output_file

    def _split_audio_if_large(self, audio_file: str, max_size_mb: int = 25) -> list:
        file_size = os.path.getsize(audio_file) / (1024 * 1024)
        if file_size <= max_size_mb:
            return [audio_file]

        chunks = []
        with wave.open(audio_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate

        num_chunks = int(file_size / max_size_mb) + 1
        chunk_duration = duration / num_chunks

        for i in range(num_chunks):
            chunk_file = tempfile.mktemp(suffix='.wav')
            start = i * chunk_duration
            subprocess.run([
                'ffmpeg', '-i', audio_file, '-ss', str(start),
                '-t', str(chunk_duration), '-y', chunk_file
            ], check=True, capture_output=True)
            chunks.append(chunk_file)

        return chunks

    def transcribe_audio(self, audio_file: str, language: str = "en") -> str:
        converted_file = self._convert_to_supported_format(audio_file)
        audio_chunks = self._split_audio_if_large(converted_file)

        full_transcript = ""
        for i, chunk_file in enumerate(audio_chunks):
            print(f"🔍 Transcribing chunk {i + 1}/{len(audio_chunks)}...")
            with open(chunk_file, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    file=(os.path.basename(chunk_file), f.read()),
                    model="whisper-large-v3",
                    language=language,
                    response_format="text"
                )
                full_transcript += result + " "
            if chunk_file != converted_file:
                os.remove(chunk_file)

        if converted_file != audio_file:
            os.remove(converted_file)

        return full_transcript.strip()

    def save_transcription(self, transcription: str, audio_file: str):
        base = os.path.splitext(audio_file)[0]
        output_file = f"{base}_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"💾 Transcription saved to {output_file}")

    def find_latest_audio(self, folder: str = "recordings") -> Optional[str]:
        if not os.path.exists(folder):
            print("❌ 'recordings/' folder not found.")
            return None
        wav_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]
        if not wav_files:
            print("❌ No .wav files found in recordings/")
            return None
        latest = max(wav_files, key=os.path.getmtime)
        print(f"📂 Using latest file: {os.path.basename(latest)}")
        return latest


def main():
    try:
        transcriber = AudioTranscriber()
        latest_audio = transcriber.find_latest_audio()

        if not latest_audio:
            return

        print("🎯 Transcribing...")
        transcript = transcriber.transcribe_audio(latest_audio)

        if transcript:
            print("\n📝 Transcription:")
            print("=" * 50)
            print(transcript)
            print("=" * 50)
            transcriber.save_transcription(transcript, latest_audio)
        else:
            print("❌ No transcription returned.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
