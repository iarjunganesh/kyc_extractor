# KYC Document Extractor

A complete pipeline for extracting Know Your Customer (KYC) information from documents using a Mistral 7B/8B LLM (via **Hugging Face Transformers** or **Ollama**) and PaddleOCR.

## Setup (Windows / PowerShell)

This repo is tested with **Python 3.12**. If you have multiple Python versions installed (e.g. 3.14 + 3.12), use a 3.12 virtual environment.

### 1) Create + activate a venv

```powershell
cd C:\Users\arjunganesh\ws\kyc_extractor
py -3.12 -m venv .venv-ocr
.\.venv-ocr\Scripts\Activate.ps1
python -m pip install -U pip
```

### 2) Install dependencies

```powershell
python -m pip install transformers torch accelerate faker pillow
python -m pip install paddleocr
```

### 3) (Recommended) Enable PaddleOCR GPU on Windows

PaddleOCR GPU on Windows requires **Paddle GPU**. Install the CUDA 12.9 build (tested):

```powershell
python -m pip uninstall -y paddlepaddle-gpu
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

This pulls CUDA/cuDNN runtime wheels into the venv (large downloads), and enables PaddleOCR GPU on modern GPUs.

## Run scripts (in order)

#### Step 1: Test Basic Extraction (No Images Needed)
```powershell
python 01_test_basic.py
```
This tests the model with sample text. Good for verifying everything works.

#### Step 2: Generate Synthetic Documents
```powershell
python 02_generate_synthetic_docs.py
```
Creates 3 sample passport images in `sample_documents/` folder for testing.

#### Step 3: Run Complete Pipeline
```powershell
python 03_kyc_extractor.py
```
Processes both text and images, extracts structured KYC data.

## Running with GPU vs CPU

Script 3 supports a few environment variables to control GPU/CPU behavior.

### GPU (default)

```powershell
.\.venv-ocr\Scripts\Activate.ps1

# LLM uses GPU if `torch.cuda.is_available()`
# OCR tries GPU if available, and falls back to CPU if it fails
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:KYC_OCR_USE_GPU -ErrorAction SilentlyContinue

$env:KYC_USE_4BIT='0'
$env:KYC_MAX_NEW_TOKENS='128'
$env:KYC_VERBOSE='1'

python 03_kyc_extractor.py
```

## LLM backend: Hugging Face vs Ollama

Script 3 supports two LLM backends:

- `hf` (default): loads a Hugging Face model locally with PyTorch
- `ollama`: calls a local Ollama server (faster startup and often much faster generation on Windows)

### Use Hugging Face (default)

```powershell
.b.venv-ocr\Scripts\Activate.ps1

$env:KYC_LLM_BACKEND='hf'
$env:KYC_MODEL_ID='mistralai/Mistral-7B-v0.1'
$env:KYC_MAX_NEW_TOKENS='128'
$env:KYC_VERBOSE='1'

python 03_kyc_extractor.py
```

### Use Ollama

Make sure Ollama is installed and running, then pull a model:

```powershell
ollama pull mistral:latest
```

Run Script 3 using the Ollama backend:

```powershell
.b.venv-ocr\Scripts\Activate.ps1

$env:KYC_LLM_BACKEND='ollama'
$env:KYC_OLLAMA_MODEL='mistral:latest'
$env:KYC_OLLAMA_URL='http://127.0.0.1:11434'
$env:KYC_OLLAMA_FORMAT_JSON='1'

$env:KYC_MAX_NEW_TOKENS='128'
$env:KYC_VERBOSE='1'

python 03_kyc_extractor.py
```

When `KYC_VERBOSE=1`, the script prints Ollama timing breakdown fields (`load_duration`, `prompt_eval_duration`, `eval_duration`, token counts, tokens/sec) to help diagnose first-run vs warm-run performance.

### Force CPU (OCR + LLM)

```powershell
.\.venv-ocr\Scripts\Activate.ps1

$env:CUDA_VISIBLE_DEVICES='-1'
$env:KYC_OCR_USE_GPU='0'

$env:KYC_USE_4BIT='0'
$env:KYC_MAX_NEW_TOKENS='128'

python 03_kyc_extractor.py
```

### Force OCR CPU (keep LLM on GPU)

```powershell
.\.venv-ocr\Scripts\Activate.ps1

$env:KYC_OCR_USE_GPU='0'
python 03_kyc_extractor.py
```

## Features

- **Text Extraction**: Extracts text from document images using PaddleOCR
- **Field Extraction**: Uses Mistral-7B to identify:
  - Full Name
  - Date of Birth
  - Passport/ID Number
  - Nationality
  - Issue & Expiry Dates
  - Document Type

- **Structured Output**: Returns extracted data in JSON format
- **Handles Both**: Text input OR image files

## Notes

- Script 3 defaults to **Mistral 7B** (`mistralai/Mistral-7B-v0.1`). Override with:
  - `KYC_MODEL_ID` (example: `mistralai/Mistral-7B-Instruct-v0.2`)
- Generation speed is dominated by token count. Tune with:
  - `KYC_MAX_NEW_TOKENS` (default is 128; try 64/96 for faster runs)
- On Windows, this repo runs PaddleOCR in a **subprocess** to avoid CUDA DLL conflicts between PyTorch (LLM) and Paddle (OCR).

### Environment variables (Script 3)

- `KYC_LLM_BACKEND`: `hf` (default) or `ollama`
- `KYC_MODEL_ID`: Hugging Face model id (used when backend is `hf`)
- `KYC_USE_4BIT`: `1` to attempt 4-bit quantization (requires `bitsandbytes`), else `0`
- `KYC_MAX_NEW_TOKENS`: generation length (default 128)
- `KYC_VERBOSE`: `1` for extra timing/debug output
- `KYC_OCR_USE_GPU`: `1` / `0` / `auto`
- `KYC_OLLAMA_MODEL`: Ollama model name (used when backend is `ollama`)
- `KYC_OLLAMA_URL`: Ollama base URL (default `http://127.0.0.1:11434`)
- `KYC_OLLAMA_FORMAT_JSON`: `1` to request strict JSON output when supported

## Output Format

```json
{
  "name": "Per Hammarstrom",
  "date_of_birth": "1957-12-18",
  "passport_number": "825832605",
  "nationality": "Sweden",
  "issue_date": "2019-09-08",
  "expiry_date": "2027-02-26",
  "document_type": "PASSPORT"
}
```

## Troubleshooting

**OCR GPU fails / slow OCR:**
- Install Paddle GPU as described above (`paddlepaddle-gpu==3.2.0` / `cu129`).
- Force CPU OCR: `KYC_OCR_USE_GPU=0`

**LLM memory issues:**
- Script 3 defaults to non-quantized loading.
- You can *attempt* 4-bit quantization by setting `KYC_USE_4BIT=1`, but this requires `bitsandbytes` (often fragile on Windows).
- If you hit VRAM limits, reduce `KYC_MAX_NEW_TOKENS` or try a smaller Mistral variant.

**HF generation is very slow:**
- Try `KYC_LLM_BACKEND=ollama` (often much faster on Windows).
- Reduce `KYC_MAX_NEW_TOKENS` to 64/96.

**GPU not detected (LLM):**
- Check `torch.cuda.is_available()` in the active venv.
- Ensure you installed a CUDA-enabled PyTorch build.

## Current defaults

- **Model**: `mistralai/Mistral-7B-v0.1`
- **OCR**: PaddleOCR (GPU if available; CPU fallback)
- **Quantization**: off by default (`KYC_USE_4BIT=0`)

## Next Steps

1. Test with your own documents
2. Fine-tune prompts in `03_kyc_extractor.py` for specific document types
3. Add validation rules for extracted fields
4. Build a batch processing pipeline
