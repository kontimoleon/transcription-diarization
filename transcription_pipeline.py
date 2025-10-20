# transcription_pipeline.py
import os
import math
import ffmpeg
import whisper
import pandas as pd
import logging
from datetime import datetime
from settings import INPUT_DIR, OUTPUT_DIR, LOG_DIR, LOG_LEVEL, MODEL_SIZE


# Logging Setup
def setup_logging(file_name="main_final"):
    """
    Configures logging for the application.
    Logs are written both to a file in LOG_DIR and to the console.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = f"{LOG_DIR}/{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging initialized. Writing logs to {log_filename}")


# Transcription Pipeline
class TranscriptionPipeline:
    def __init__(self, input_dir, output_dir, model_size):
        """
        End-to-end transcription pipeline using local Whisper.

        Args:
            input_dir (str): Directory containing video files.
            output_dir (str, optional): Directory for output audio and transcripts.
            model_size (str): Whisper model to use ('tiny', 'base', 'small', 'medium', 'large').
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_size = model_size
        self.model = whisper.load_model(model_size)

        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized pipeline with model 'whisper-{self.model_size}'.")

    @staticmethod
    def format_time(seconds):
        """Convert seconds to SRT-compatible timestamp."""
        hours = math.floor(seconds / 3600)
        seconds %= 3600
        minutes = math.floor(seconds / 60)
        seconds %= 60
        milliseconds = round((seconds - math.floor(seconds)) * 1000)
        seconds = math.floor(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def extract_audio(self, input_video):
        """Extract audio from a video file to WAV format."""
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        extracted_audio = os.path.join(self.output_dir, f"{base_name}.wav")

        try:
            stream = ffmpeg.input(input_video)
            stream = ffmpeg.output(stream, extracted_audio, acodec="pcm_s16le", ar="44100", ac="2")
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            self.logger.info(f"Extracted audio: {extracted_audio}")
            return extracted_audio
        except Exception as e:
            self.logger.error(f"Failed to extract audio from {input_video}: {e}")
            raise

    def transcribe(self, audio_file):
        """Transcribe the given audio file using Whisper."""
        self.logger.info(f"Transcribing: {os.path.basename(audio_file)}")

        try:
            result = self.model.transcribe(audio_file, verbose=False)
            segments = result.get("segments", [])

            transcripts = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                for seg in segments
            ]

            self.logger.info(f"Transcription complete: {os.path.basename(audio_file)}")
            return transcripts

        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_file}: {e}")
            raise

    def generate_subtitle_file(self, segments, output_file_name):
        """Generate an .srt subtitle file from transcription segments."""
        subtitle_path = os.path.join(self.output_dir, f"{output_file_name}.srt")
        try:
            with open(subtitle_path, "w", encoding="utf-8") as f:
                for index, segment in enumerate(segments, start=1):
                    start = self.format_time(segment["start"])
                    end = self.format_time(segment["end"])
                    f.write(f"{index}\n{start} --> {end}\n{segment['text']}\n\n")

            self.logger.info(f"Subtitle file generated: {subtitle_path}")
            return subtitle_path

        except Exception as e:
            self.logger.error(f"Failed to write subtitle file for {output_file_name}: {e}")
            raise

    def run(self):
        """Run the transcription pipeline for all videos in the input directory."""
        self.logger.info(f"Scanning input directory: {self.input_dir}")

        videos = [
            os.path.join(self.input_dir, f)
            for f in os.listdir(self.input_dir)
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        ]

        if not videos:
            self.logger.warning(f"No video files found in {self.input_dir}")
            return

        for video in videos:
            base_name = os.path.splitext(os.path.basename(video))[0]
            self.logger.info(f"Processing video: {base_name}")

            try:
                audio_path = self.extract_audio(video)
                segments = self.transcribe(audio_path)
                self.generate_subtitle_file(segments, base_name)
            except Exception as e:
                self.logger.error(f"Error processing {video}: {e}")
                continue

        self.logger.info(f"Processing input directory complete. Outputs saved in: {self.output_dir}")


if __name__ == "__main__":
    setup_logging("transcription_pipeline")

    pipeline = TranscriptionPipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_size=MODEL_SIZE
    )
    pipeline.run()
