# Transcription & Speaker Diarization

## Installation Guide

### On `icr-s05.ethz.ch` (CUDA 13.0, RTX 6000 Ada)

1. Clone the repo and navigate to the repo directory

2. Create a new virtual environment  
   `python3 -m venv .venv`

3. Activate the virtual environment  
   `source .venv/bin/activate`

4. Install PyTorch for CUDA 13.0 (must be done before the rest, as `+cu130` is not on PyPI)  
   `pip install torch==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130`

5. Install remaining required packages  
   `pip install -r requirements.txt`

6. Install cuBLAS and cuDNN shared libraries required by faster-whisper's ctranslate2 backend  
   ```bash
   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   export LD_LIBRARY_PATH="$(python -c "import os, nvidia.cublas, nvidia.cudnn; print(os.path.join(nvidia.cublas.__path__[0], 'lib') + ':' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))"):$LD_LIBRARY_PATH"
   ```
   Add the `export` line to your `~/.bashrc` (or re-run it each session before running scripts).

---

### On Euler (CUDA 12.x)

1. Clone the repo and navigate to the repo directory

2. Load the Python CUDA stack  
   `module load stack/2024-06 python_cuda/3.11.6`  
   or  
   `module load stack/2024-06 python_cuda/3.9.18`

3. Create a new virtual environment  
   `python -m venv .venv`

4. Activate the virtual environment  
   `source .venv/bin/activate`

5. Install PyTorch for CUDA 12 (check your exact version with `nvcc --version` after loading the module)  
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`

6. Install remaining required packages  
   `pip install -r requirements.txt`

7. Install cuBLAS and cuDNN shared libraries required by faster-whisper's ctranslate2 backend  
   ```bash
   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   export LD_LIBRARY_PATH="$(python -c "import os, nvidia.cublas, nvidia.cudnn; print(os.path.join(nvidia.cublas.__path__[0], 'lib') + ':' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))"):$LD_LIBRARY_PATH"
   ```
   Add the `export` line to your `~/.bashrc` (or re-run it each session before running scripts).

---

## Running

7. Transfer video/audio files.

8. Extract audio if your input is in video format (you might need to adjust the input/output directories in the bash script).
   - Make the script executable: `chmod +x extract_audio.sh`
   - Run audio extraction: `./extract_audio.sh`
   - Adjust `INPUT_DIR` and `OUTPUT_DIR` in `settings.py` accordingly.

9. Run the minimal reproducible example to verify GPU is detected and transcription works  
   `python3 mre.py`

10. Run the full transcription pipeline  
    `python3 transcription_pipeline.py`

---

## Notes

- On `icr-s05`, both GPUs may be in use by other processes. Check free VRAM with `nvidia-smi` before running.  
  If GPU 0 is low on memory, force GPU 1 with: `CUDA_VISIBLE_DEVICES=1 python3 mre.py`
- `large-v3` with `float16` needs ~3 GB VRAM.
