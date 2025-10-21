"""
mre.py: Minimal reproducible example to test the transcription pipeline.

"""
import os
import whisper
import logging
from datetime import datetime
from settings import INPUT_DIR, OUTPUT_DIR, LOG_DIR, LOG_LEVEL, MODEL_SIZE


# Logging Setup
def setup_logging(file_name):
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

def whisper_paradigm(input_file):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Loading whisper model '{MODEL_SIZE}'")
    model = whisper.load_model(MODEL_SIZE)
    logging.info(f"Initialized pipeline with model 'whisper-{MODEL_SIZE}'.")

    result = model.transcribe(input_file)
    text = result.get("text", "")
    logging.info(f"Successfully transcribed {input_file}")


    output_file = os.path.join(OUTPUT_DIR, input_file.split('/')[-1].split('.')[0]+"output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    logging.info(f"Transcript written to {output_file}")


if __name__ == "__main__":
    setup_logging(file_name="mre")
    audios = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
    ]

    for audio in audios:
        whisper_paradigm(audio)