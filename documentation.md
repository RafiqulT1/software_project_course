
# Table of Contents

1.  [Project Installation & Setup Guide (Jetson Orin / JetPack 6.0)](#org6362594)
    1.  [1. Overview](#org772c973)
    2.  [2. System Prerequisites (APT Packages)](#org923d2a1)
    3.  [3. ONNX Runtime GPU Wheel Location](#org3f0a75c)
        1.  [Pitfalls:](#orgc26af37)
    4.  [4. Create and Activate the virtual Environment](#org98f607f)
    5.  [5. Install Python dependencies from requirements.txt](#org21d7a2c)
        1.  [Important Version couplings & Pitfalls](#org435de22)
    6.  [6. Download NLTK Data (for text2emotion)](#org6e67ab4)
        1.  [Pitfall:](#org8d6d54e)
    7.  [7. Sanity Checking Key Components](#orgfdc667e)
        1.  [7.1 ONNX Runtime](#orgc902e2a)
        2.  [7.2 Whisper](#org68675f2)
        3.  [7.3 text2emotion](#org98bf16e)
        4.  [7.4 Transformers & Torch](#orge9171a9)
    8.  [8. Running the project](#org35506a7)
    9.  [9. Common gotchas & how to fix them](#org8ef998c)
        1.  [9.1 ModuleNotFoundError: No module named 'X'](#org97abe08)
        2.  [9.2 free(): invalid next size (fast) on exit](#orgb546417)
    10. [10. Recreating the environment in the future](#org245bad4)


<a id="org6362594"></a>

# Project Installation & Setup Guide (Jetson Orin / JetPack 6.0)

This document describes how to set up and run the "Finnish spoken robot dialogue - local LLM based implementation" -project:

-   **Device:** Jetson Orin NX / reComputer J4012
-   **OS:** JetPack 6.0 (Ubuntu 22.04 base)

The project uses a mix of ONNX Runtime GPU, Whisper, Piper-TTS, Transformers, Librosa, and text2emotion. Some packages have tight version coupling.


<a id="org772c973"></a>

## 1. Overview

The project includes:

-   **ONNX Runtime GPU** with a **Jetson-specific wheel**
-   **Whisper** (*openai-whisper*) for speech-to-text
-   **Piper-TTS** for text-to-speech
-   **Transformers**, **PyTorch**, **scikit-learn**, **librosa** for NLP/audio processing
-   **text2emotion + NLTK** for emotion analysis from text


<a id="org923d2a1"></a>

## 2. System Prerequisites (APT Packages)

Some packages for this project are installed systemwide. Some packages are installed to Python virtual environment to make sure that the versions are correct.   

Install system-level dependencies first:

    sudo apt update
    sudo apt install python3-venv python3-dev build-essential libsndfile1 portaudio19-dev ffmpeg


<a id="org3f0a75c"></a>

## 3. ONNX Runtime GPU Wheel Location

You can download the correct wheel for this project with this command:

    wget https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl -O onnxruntime_gpu-1.18.0-cp310-linux_aarch64.whl

In `requirements.txt` there is an entry similar to:

    onnxruntime-gpu @ file:///home/user/Documents/software_project_course/onnxruntime_gpu-1.18.0-cp310-linux_aarch64.whl

You must ensure:

1.  That wheel actually exists at the referenced path, **or**
2.  You edit `requirements.txt` and update that path to wherever the wheel is stored.


<a id="orgc26af37"></a>

### Pitfalls:

-   If the path is wrong, `pip install -r requirments.txt` will fail with a "file not found" error.
-   Do **not** replace this with `onnxruntime-gpu` from PyPi on  Jetson Orin. Many generic ARM builds crash on Orin with a `std::vector` assertion and core dump.


<a id="org98f607f"></a>

## 4. Create and Activate the virtual Environment

Create a dedicated virtual environment:

    python3 -m venv ~/venvs/jetson-onnx

Activate it:

    source ~/venvs/jetson-onnx/bin/activate

Upgrade `pip`:

    python -m pip install --upgrade pip

If you forget to activate the venv, packages will install into the **system Python**. When you install packages with pip systemwide, it is difficult to make them all be specific versions.


<a id="org21d7a2c"></a>

## 5. Install Python dependencies from requirements.txt

With the venv \*activated and the ONNX Runtime wheel in the right place:

    cd /path/to/the/project
    pip install -r requirements.txt

This will install:

-   numpy==1.23.5
-   onnxruntime-gpu (from the local Jetson wheel)
-   openai-whisper (e.g. 20250625)
-   piper-tts
-   transformers
-   torch
-   text2emotion
-   emoji
-   nltk
-   librosa
-   And other helper libraries


<a id="org435de22"></a>

### Important Version couplings & Pitfalls

1.  NumPy & ONNX Runtime

    -   Our ONNX Runtime wheel is compiled agains **NumPy 1.x**, so we use *numpy==1.23.5*
    -   If NumPy ever upgrades to 2.x (by installing some other package), you may see:
    
        A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
        ImportError: import numpy failed
    
    1.  Fix:
    
            pip install "numpy==1.23.5" --force-reinstall

2.  ONNX Runtime GPU on Jetson

    -   You must use the **Jetson-specific** `onnxruntime-gpu` **wheel** (1.18.0).
    -   If you accidentally install `onnxruntime` / `onnxruntime-gpu` with pip, you may see on import:
    
        onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
        /opt/.../stl_vector.h:...
        Assertion '__n < this->size()' failed.
        Aborted (core dumped)
    
    1.  Fix:
    
            pip uninstall -y onnxruntime onnxruntime-gpu
            pip install /path/to/onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl

3.  emoji & text2emotion

    -   `text2emotion` expects the old `emoji.UNICODE_EMOJI` API.
    -   This only exists in `emoji` 1.x, not in 2.x.
    -   If `emoji` upgrades to 2.x, you'll see:
    
        AttributeError: module 'emoji' has no attribute 'UNICODE_EMOJI'
    
    -   The requirements should pin `emoji==1.7.0`.

4.  Torch on Jetson

    -   Jetson works best with nvidia's official L4T PyTorch wheels.
    -   If `pip install -r requirements.txt` tries to compile `torch` from source (very slow/may fail), consider:
        1.  Install nvidia's torch wheel first (per Jetson docs).
        2.  Remove `torch==...` from `requirements.txt` or install the rest with:
    
        pip install -r requirements.txt --no-deps


<a id="org6e67ab4"></a>

## 6. Download NLTK Data (for text2emotion)

`text2emotion` uses NLTK tokenizers and corpora. With our pinned `nltk` version, we need at least:

-   punkt
-   punkt<sub>tab</sub>
-   stopwords
-   wordnet

Install them once (**with the venv active**):

    python -m nltk.downloader punkt punkt_tab stopwords wordnet


<a id="org8d6d54e"></a>

### Pitfall:

-   If you skip this, you may see errors like:

    LookupError: 
      Resource punkt_tab not found.
      Please use the NLTK Downloader to obtain the resource:
      >>> import nltk
      >>> nltk.download('punkt_tab')

Quick test:

    python -c 'import text2emotion as te; print(te.get_emotion("I am very happy today"))'

Expected output:

    {'Happy': 1.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.0}

If you see something like this, `text2emotion` is fully functional.


<a id="orgfdc667e"></a>

## 7. Sanity Checking Key Components

With the **venv activated**, run the following checks.


<a id="orgc902e2a"></a>

### 7.1 ONNX Runtime

    python -c "import onnxruntime as ort; print('ORT:', ort.__version__, ort.get_device())"

Expected output:

    ORT: 1.18.0 GPU

Notes:

-   A line like:

    onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0

&#x2026;is just a warning and can be ignored.

If you get the `std::vector` assertion or a core dump, your `onnxruntime` installation is wrong, and you should reinstall the Jetson wheel.


<a id="org68675f2"></a>

### 7.2 Whisper

    python -c "import whisper; print('Whisper:', whisper.__version__)"

Expect a version string (something like: `20250625`)


<a id="org98bf16e"></a>

### 7.3 text2emotion

(Assuming NLTK data & correct emoji):

    python -c 'import text2emotion as te; print(te.get_emotion("I am very happy today"))'


<a id="orge9171a9"></a>

### 7.4 Transformers & Torch

    python -c "import torch, transformers; print('Torch:', torch.__version__, 'Transformers:', transformers.__version__)"

Expect both version numbers without errors.


<a id="org35506a7"></a>

## 8. Running the project

Once everything is installed and sanity checks pass:

1.  Activate the venv:

    source ~/venvs/jetson-onnx/bin/activate

1.  Go to the project folder:

    cd /path/to/the/project

1.  Run the main script:

    python ollamarunner.py


<a id="org8ef998c"></a>

## 9. Common gotchas & how to fix them


<a id="org97abe08"></a>

### 9.1 ModuleNotFoundError: No module named 'X'

This means:

-   You don't have the module 'X' installed in your venv, **or**
-   You're not in the correct venv, **or**
-   Installation failed earlier, **or**
-   `requirements.txt` was changed and doesn't include `X`.

1.  Fix:

    1.  Activate the venv:
    
        source ~/venvs/jetson-onnx/bin/activate
    
    1.  Install the missing package:
    
        pip install X
    
    or re-run:
    
        pip install -r requirements.txt


<a id="orgb546417"></a>

### 9.2 free(): invalid next size (fast) on exit

If after importing some library you see:

    free(): invalid next size (fast)
    Aborted (core dumped)

It's typically a buggy or mismatched C++ extension, most often `onnxruntime` on Jetson.

-   First, check which ORT you have:

    python -c "import onnxruntime as ort; print(ort.__version__, ort.get_device())"

-   If it's not the Jetson-specific `1.18.0` wheel you expect, reinstall your wheel:

    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install /path/to/onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl


<a id="org245bad4"></a>

## 10. Recreating the environment in the future

To recreate this environment on the same or on another Jetson:   

1.  Install system dependencies:

    sudo apt update
    sudo apt install -y python3-venv python3-dev build-essential libsndfile1 portaudio19-dev ffmpeg

1.  Copy the project folder, including `requirements.txt` and the `ONNX Runtime wheel`.

2.  Ensure the `onnxruntime-gpu @ file:///...` path in `requirements.txt` points to the correct wheel location.

3.  Create and activate the venv, and upgrade pip for it:

    python3 -m venv ~/venvs/jetson-onnx
    source ~/venvs/jetson-onnx/bin/activate
    python -m pip install --upgrade pip

1.  Install Python dependencies:

    pip install -r requirements.txt

1.  Download NLTK data:

    python -m nltk.downloader punkt punkt_tab stopwords wordnet

1.  Run sanity checks to confirm everything is working.

2.  Run the main script:

    python ollamarunner.py

