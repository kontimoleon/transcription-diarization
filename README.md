# Transcription & Speaker Diarization

## Instalation Guide

1. Mount project. Video folder must be called `interview-videos`. TODO: how?  
2. Clone the repo and navigate to the repo directory
3. Create a new virtual environment  
`python3 -m venv .venv`
4. Activate the virtual environment  
`source .venv/bin/activate`
5. Install required packages  
`pip install -r requirements.txt`
6. Install CUDA 13.0-compatible PyTorch Build
`pip3 install torch --index-url https://download.pytorch.org/whl/cu130`
7. Install ffmpeg distribution (Note: sudo not available to all users)  
`sudo apt update && sudo apt install ffmpeg`
8. Extract audio from video  
`audio_extraction.ps1`
9. Run the transcription pipeline  
`python3 transcription_pipeline.py`
