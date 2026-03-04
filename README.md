# DER Evaluation Harness

General-purpose **Diarization Error Rate (DER)** evaluation for any diarization
backend against any HuggingFace speech dataset with gold speaker annotations.

# Background
Accurate speaker diarization is crucial for downstream ASR tasks like speaker-attributed transcription, meeting summarization, and conversational analytics — especially in low-resource languages like Icelandic. This harness allows fair, offline comparison of local models (pyannote) vs cloud APIs vs external baselines on the same dataset, with resumable runs, failure isolation, and diagnostic plots.
---

## API support notes.
Cloud APIs are supported provided you have valid credentials, sufficient quota, and compatible audio formats. Minor configuration adjustments may be required depending on your environment.

## Supported backends

| Key | System | Needs GPU? |
|-----|--------|-----------|
| `pyannote_local` | pyannote/speaker-diarization-3.1 (local inference) | Recommended |
| `rttm` | Any external system — supply pre-computed RTTM files | No |
| `aws` | AWS Transcribe (speaker diarization job) | No |
| `revai` | Rev.ai (speaker diarization job) | No |

---

## Project structure

```
project/
├── der_eval/
│   ├── __init__.py      # package version
│   ├── config.py        # CONFIG defaults + argparse
│   ├── audio.py         # materialise dataset audio to WAV files
│   ├── backends.py      # backend init + per-backend diarize()
│   ├── metrics.py       # reference annotation + evaluation loop
│   ├── report.py        # console summary + scatter plot
│   └── __main__.py      # wires all modules together
├── eval_der.py          # thin top-level entry point
└── requirements.txt
```

---

## Installation

### 1. Prerequisites

Install **system ffmpeg** (required by pyannote for audio decoding):

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Conda (any platform)
conda install -c conda-forge ffmpeg
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install PyTorch

PyTorch must be installed **before** the rest of the requirements because pip
cannot apply `--index-url` on a per-package basis from a requirements file.

```bash
# CPU only
pip install torch==2.4.0 torchaudio==2.4.0

# CUDA 12.1  (RTX 30xx / 40xx)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8  (older GPUs)
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify

```bash
python -c "import torch, torchaudio, pyannote.audio; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('pyannote OK')"
```

---

## HuggingFace authentication

The `pyannote/speaker-diarization-3.1` model is gated. You must:

1. Create a token at <https://huggingface.co/settings/tokens>
2. Accept the model terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>
3. Log in:

```bash
huggingface-cli login
# paste your hf_... token when prompted
```

Alternatively, pass the token directly at runtime:

```bash
python eval_der.py --hf-token hf_yourtoken
```

---

## Configuration

All defaults live in `der_eval/config.py` under the `DEFAULTS` dict.
Edit that block to permanently change any setting. Every key can also be
overridden at runtime via CLI arguments (see below).

Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `pyannote_local` | Which diarization system to use |
| `dataset_id` | `palli23/Spjallromur-AB-NoOverlap-v3` | HuggingFace dataset |
| `split` | `train` | Dataset split, supports slice syntax |
| `audio_col` | `audio` | Name of the audio column |
| `segments_col` | `segments` | Name of the gold segments column |
| `duration_col` | `duration` | Name of the duration column |
| `audio_out_dir` | `diar_eval_audio` | Where materialised WAVs are written |
| `results_csv` | `diarization_eval_results.csv` | Per-file DER results |
| `plot_png` | `der_vs_duration.png` | Output scatter plot |

---

## Usage

### Basic run

```bash
python eval_der.py
```

### Quick smoke test (2 files)

```bash
python eval_der.py --split "train[:2]"
```

### Switch backend

```bash
# Pre-computed RTTM files
python eval_der.py --backend rttm --rttm-dir ./my_rttm_files

# AWS Transcribe
python eval_der.py --backend aws

# Rev.ai
python eval_der.py --backend revai
```

### Custom dataset

```bash
python eval_der.py --dataset my_org/my_dataset --split "test"
```

### Custom output paths

```bash
python eval_der.py --results my_results.csv --plot my_plot.png
```

### All CLI options

```
--backend     {pyannote_local,rttm,aws,revai}
--dataset     HuggingFace dataset ID
--split       Dataset split, e.g. "train" or "train[:50]"
--hf-token    HuggingFace access token
--rttm-dir    Directory containing pre-computed RTTM files
--audio-dir   Directory where WAV files are written
--results     Output CSV path
--plot        Output PNG path
```

You can also run the package directly:

```bash
python -m der_eval --split "train[:2]"
```

---

## RTTM backend

If you have diarization output from an external system, export it to RTTM
format and use the `rttm` backend. Place all files in a single directory
named with a zero-padded index prefix matching the dataset row:

```
rttm_output/
    0000_<anything>.rttm
    0001_<anything>.rttm
    ...
```

Each file must follow the standard 10-field RTTM format:

```
SPEAKER <file_id> 1 <start_sec> <duration_sec> <NA> <NA> <speaker_id> <NA> <NA>
```

Example:
```
SPEAKER conv_001 1 0.000 4.500 <NA> <NA> spk_00 <NA> <NA>
SPEAKER conv_001 1 4.800 3.200 <NA> <NA> spk_01 <NA> <NA>
```

---

## Output

### Console summary

```
=============================================
  Diarization Evaluation Summary
=============================================
  Backend   : pyannote_local
  Dataset   : palli23/Spjallromur-AB-NoOverlap-v3  (train)
  Processed : 30  |  DER computed: 30
  Avg DER (unweighted)   : 21.40%
  Avg DER (dur-weighted) : 20.87%
  Std deviation          : 5.12%
  Results CSV : diarization_eval_results.csv
=============================================
```

### CSV columns

| Column | Description |
|--------|-------------|
| `index` | Dataset row index |
| `backend` | Backend used |
| `dataset` | HuggingFace dataset ID |
| `split` | Split used |
| `audio_path` | Path to materialised WAV |
| `duration` | Clip duration in seconds |
| `num_segments` | Number of gold speaker segments |
| `der` | DER as a fraction (0–1+) |
| `der_percent` | DER as a percentage |

Failed rows are written to a separate `*_failures.csv` file.

### Plot

A DER-vs-duration scatter plot is saved to `der_vs_duration.png`.

---

## AWS Transcribe setup

Set credentials either in `der_eval/config.py` or as environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

The IAM user needs `transcribe:StartTranscriptionJob`,
`transcribe:GetTranscriptionJob`, `s3:PutObject` permissions.

---

## Rev.ai setup

```bash
export REVAI_ACCESS_TOKEN=...
```

Or set `revai_access_token` in `der_eval/config.py`.