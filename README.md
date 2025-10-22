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
6. Install CUDA 13.0-compatible PyTorch Build for openai-whisper.
`pip3 install torch --index-url https://download.pytorch.org/whl/cu130`  
OR  
Install cuBLAS and cuDNN for faster-whisper (only for Linux!)
    ```
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
    export LD_LIBRARY_PATH="$(
        .venv/bin/python - <<'PY'
        import os, nvidia.cublas, nvidia.cudnn
        cublas_lib = os.path.join(nvidia.cublas.__path__[0], 'lib')
        cudnn_lib = os.path.join(nvidia.cudnn.__path__[0], 'lib')
        print(f"{cublas_lib}:{cudnn_lib}")
        PY
        ):$LD_LIBRARY_PATH"
    ```
7. Extract audio from video  
`audio_extraction.ps1`
8. Run the transcription pipeline  
`python3 transcription_pipeline.py`
