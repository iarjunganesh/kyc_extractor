"""
Step 3: Complete KYC extraction pipeline
Extracts text from documents using OCR and then processes with the model
"""

import os
import importlib.util
import importlib.metadata
import subprocess
import sys
import time
import urllib.request
import urllib.error

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# For OCR (install with: pip install paddleocr)
# NOTE: We intentionally avoid importing paddleocr/paddle at module import time.
# On Windows, importing both torch (CUDA) and paddle (CUDA) in the same process can
# lead to DLL conflicts (different CUDA/cuDNN builds). We run OCR in a subprocess.
OCR_AVAILABLE = importlib.util.find_spec("paddleocr") is not None
if not OCR_AVAILABLE:
    print("Warning: paddleocr not installed. OCR features disabled.")
    print("Install with: pip install paddleocr")

class KYCExtractor:
    def __init__(self):
        verbose_env = os.getenv("KYC_VERBOSE", "0").strip().lower()
        self._verbose = verbose_env in {"1", "true", "yes", "y"}

        if self._verbose:
            print(f"Python: {sys.version.split()[0]}")
            print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                try:
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                except Exception:
                    pass

        # Small, safe performance knobs for NVIDIA GPUs.
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self._llm_backend = os.getenv("KYC_LLM_BACKEND", "hf").strip().lower()
        if self._llm_backend not in {"hf", "ollama"}:
            raise ValueError("KYC_LLM_BACKEND must be 'hf' or 'ollama'")

        # Ollama backend: no HuggingFace model load.
        if self._llm_backend == "ollama":
            self._ollama_model = os.getenv("KYC_OLLAMA_MODEL", "mistral")
            self._ollama_url = os.getenv("KYC_OLLAMA_URL", "http://localhost:11434")

            if self._verbose:
                print(f"LLM backend: ollama | model: {self._ollama_model} | url: {self._ollama_url}")

            self.model = None
            self.tokenizer = None
            self._input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            print("Loading model...")
            model_load_start = time.perf_counter()

        if self._llm_backend == "hf":
            def _is_accelerate_available() -> bool:
                return importlib.util.find_spec("accelerate") is not None

            def _is_bitsandbytes_available() -> bool:
                if importlib.util.find_spec("bitsandbytes") is None:
                    return False
                try:
                    importlib.metadata.version("bitsandbytes")
                    return True
                except importlib.metadata.PackageNotFoundError:
                    return False

            # Control knobs (defaults chosen for Windows reliability)
            # - KYC_USE_4BIT=1 enables 4-bit (only if bitsandbytes is available)
            # - KYC_USE_4BIT=0 forces non-quantized load
            use_4bit_env = os.getenv("KYC_USE_4BIT")
            if use_4bit_env is None:
                use_4bit = False
            else:
                use_4bit = use_4bit_env.strip().lower() in {"1", "true", "yes", "y"}

            # Default model: Mistral 7B. Override with KYC_MODEL_ID if needed.
            # Example: $env:KYC_MODEL_ID='mistralai/Mistral-7B-Instruct-v0.2'
            model_id = os.getenv("KYC_MODEL_ID", "mistralai/Mistral-7B-v0.1")
            quantization_config = None

            if use_4bit:
                if not _is_bitsandbytes_available():
                    print("KYC_USE_4BIT=1 was set, but bitsandbytes is not installed; falling back to non-quantized load.")
                elif not torch.cuda.is_available():
                    print("KYC_USE_4BIT=1 was set, but CUDA is not available; falling back to non-quantized load.")
                else:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

            load_kwargs = {}
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                # Quantized models effectively require CUDA.
                load_kwargs["device_map"] = "auto" if _is_accelerate_available() else None
            else:
                # Non-quantized path: try to use GPU when available, otherwise CPU.
                if torch.cuda.is_available():
                    # transformers prefers `dtype` over `torch_dtype` in newer versions.
                    load_kwargs["dtype"] = torch.float16
                    if _is_accelerate_available():
                        load_kwargs["device_map"] = "auto"

            # Remove None entries to avoid transformers complaining.
            load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            # If we didn't use accelerate/device_map and CUDA exists, put the model on GPU by default.
            if quantization_config is None and torch.cuda.is_available() and not _is_accelerate_available():
                self.model = self.model.to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Avoid generation warnings on models without a pad token.
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            try:
                self.model.eval()
            except Exception:
                pass

            model_load_end = time.perf_counter()
            if self._verbose:
                print("LLM backend: hf")
                print(f"Model: {model_id}")
                try:
                    device = next(self.model.parameters()).device
                    print(f"Model device: {device}")
                except Exception:
                    pass
                print(f"Model load time: {model_load_end - model_load_start:.2f}s")

            # Pick a reasonable device to place inputs on.
            # When device_map="auto" is used, model may be sharded; CUDA:0 is usually correct.
            # Otherwise, follow the model's parameter device.
            try:
                first_param = next(self.model.parameters())
                self._input_device = first_param.device
            except StopIteration:
                self._input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # OCR runs in a subprocess (see extract_text_from_document) to avoid
        # torch/paddle CUDA DLL conflicts on Windows.
        if OCR_AVAILABLE:
            print("OCR available (runs in subprocess)...")
        self.ocr = None
        self._ocr_use_gpu = False
        
        print("KYC Extractor ready!\n")
    
    def extract_text_from_document(self, image_path):
        """Extract text from document image using OCR"""
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR not available. Install paddleocr: pip install paddleocr")
        
        print(f"Extracting text from: {image_path}")
        ocr_start = time.perf_counter()

        # Default behavior: try GPU when available, otherwise CPU.
        # Override with:
        #   KYC_OCR_USE_GPU=0  -> force CPU
        #   KYC_OCR_USE_GPU=1  -> try GPU (fallback to CPU if init fails)
        ocr_use_gpu_env = os.getenv("KYC_OCR_USE_GPU", "auto").strip().lower()
        if ocr_use_gpu_env in {"0", "false", "no", "n"}:
            ocr_try_gpu = False
        elif ocr_use_gpu_env in {"1", "true", "yes", "y"}:
            ocr_try_gpu = True
        else:
            ocr_try_gpu = torch.cuda.is_available()

        ocr_code = "\n".join(
            [
                "import os",
                "import json",
                "import sys",
                "import warnings",
                "",
                "warnings.filterwarnings('ignore', message=r'No ccache found.*', category=UserWarning)",
                "os.environ.setdefault('GLOG_minloglevel', '2')",
                "os.environ.setdefault('FLAGS_minloglevel', '2')",
                "",
                "from paddleocr import PaddleOCR",
                "",
                "image_path = sys.argv[1]",
                "try_gpu = sys.argv[2].strip().lower() in {'1','true','yes','y'}",
                "used_gpu = False",
                "",
                "if try_gpu:",
                "    try:",
                "        ocr = PaddleOCR(lang='en', use_gpu=True, show_log=False)",
                "        used_gpu = True",
                "    except Exception:",
                "        ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)",
                "else:",
                "    ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)",
                "",
                "try:",
                "    result = ocr.ocr(image_path)",
                "except Exception:",
                "    if used_gpu:",
                "        ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)",
                "        used_gpu = False",
                "        result = ocr.ocr(image_path)",
                "    else:",
                "        raise",
                "",
                "if not result:",
                "    text = ''",
                "else:",
                "    text = '\\n'.join([",
                "        item[1][0]",
                "        for line in result",
                "        for item in (line or [])",
                "        if item and len(item) > 1 and item[1] and len(item[1]) > 0",
                "    ])",
                "",
                "print(json.dumps({'text': text, 'used_gpu': used_gpu}, ensure_ascii=False))",
            ]
        )

        completed = subprocess.run(
            [sys.executable, "-c", ocr_code, image_path, "1" if ocr_try_gpu else "0"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            combined_err = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            raise RuntimeError(f"OCR subprocess failed (exit {completed.returncode}). Output was:\n{combined_err}")
        combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        # Paddle/PaddleOCR may still emit warnings; extract the JSON object from output.
        json_start = combined.rfind("{")
        json_end = combined.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise RuntimeError(f"OCR subprocess did not return JSON. Output was:\n{combined}")
        payload = json.loads(combined[json_start:json_end])
        extracted_text = payload.get("text", "")
        self._ocr_use_gpu = bool(payload.get("used_gpu", False))

        ocr_end = time.perf_counter()
        if self._verbose:
            backend = "GPU" if self._ocr_use_gpu else "CPU"
            print(f"OCR backend used: {backend} | OCR time: {ocr_end - ocr_start:.2f}s")

        return extracted_text
    
    def extract_kyc_fields(self, document_text):
        """Use LLM to extract structured KYC fields"""

        # Keep the prompt short to reduce latency.
        prompt = (
            "Extract KYC fields and return ONLY valid JSON with keys: "
            "name, date_of_birth, passport_number, nationality, issue_date, expiry_date, document_type. "
            "Dates must be YYYY-MM-DD.\n\n"
            f"DOCUMENT:\n{document_text}\n\nJSON:"
        )
        
        print("Processing with model...")
        gen_start = time.perf_counter()

        if self._llm_backend == "ollama":
            result = self._extract_kyc_fields_ollama(prompt)
            gen_end = time.perf_counter()
            if self._verbose:
                max_new_tokens_env = os.getenv("KYC_MAX_NEW_TOKENS")
                max_new_tokens = 128
                if max_new_tokens_env:
                    try:
                        max_new_tokens = int(max_new_tokens_env)
                    except ValueError:
                        pass
                print(f"Generation time: {gen_end - gen_start:.2f}s | max_new_tokens={max_new_tokens}")
            return result

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)

        max_new_tokens_env = os.getenv("KYC_MAX_NEW_TOKENS")
        max_new_tokens = 128
        if max_new_tokens_env:
            try:
                max_new_tokens = int(max_new_tokens_env)
            except ValueError:
                print(f"Warning: invalid KYC_MAX_NEW_TOKENS={max_new_tokens_env!r}; using {max_new_tokens}.")

        with torch.inference_mode():
            if self._input_device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=False,
                        use_cache=True,
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    use_cache=True,
                )
        gen_end = time.perf_counter()
        if self._verbose:
            print(f"Generation time: {gen_end - gen_start:.2f}s | max_new_tokens={max_new_tokens}")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to parse JSON from response
        try:
            # Extract JSON part
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {"error": "Could not parse response", "raw": response}
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON response", "raw": response}
        
        return result

    def _extract_kyc_fields_ollama(self, prompt: str):
        """Call Ollama local server and return extracted fields as dict."""

        max_new_tokens_env = os.getenv("KYC_MAX_NEW_TOKENS")
        max_new_tokens = 128
        if max_new_tokens_env:
            try:
                max_new_tokens = int(max_new_tokens_env)
            except ValueError:
                pass

        # Try to enforce JSON output when supported.
        format_json_env = os.getenv("KYC_OLLAMA_FORMAT_JSON", "1").strip().lower()
        use_format_json = format_json_env in {"1", "true", "yes", "y"}

        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": max_new_tokens,
            },
        }
        if use_format_json:
            payload["format"] = "json"

        url = self._ollama_url.rstrip("/") + "/api/generate"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"Ollama HTTP error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Could not reach Ollama at http://localhost:11434. "
                "Ensure the Ollama app/service is running."
            ) from e

        data = json.loads(raw)
        response_text = (data.get("response") or "").strip()

        if self._verbose:
            # Ollama returns durations in nanoseconds.
            def _ns_to_s(value):
                try:
                    return float(value) / 1e9
                except Exception:
                    return None

            load_s = _ns_to_s(data.get("load_duration"))
            prompt_eval_s = _ns_to_s(data.get("prompt_eval_duration"))
            eval_s = _ns_to_s(data.get("eval_duration"))
            total_s = _ns_to_s(data.get("total_duration"))
            prompt_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")

            parts = []
            if load_s is not None:
                parts.append(f"load={load_s:.2f}s")
            if prompt_eval_s is not None:
                parts.append(f"prompt_eval={prompt_eval_s:.2f}s")
            if eval_s is not None:
                parts.append(f"eval={eval_s:.2f}s")
            if total_s is not None:
                parts.append(f"total={total_s:.2f}s")

            tok_parts = []
            if isinstance(prompt_tokens, int):
                tok_parts.append(f"prompt_tokens={prompt_tokens}")
            if isinstance(output_tokens, int):
                tok_parts.append(f"output_tokens={output_tokens}")
                if eval_s and eval_s > 0:
                    tok_parts.append(f"tok/s={output_tokens / eval_s:.2f}")

            print(
                "Ollama timings: "
                + (", ".join(parts) if parts else "(no timing fields)")
                + (" | " + ", ".join(tok_parts) if tok_parts else "")
            )
            print(f"Ollama prompt chars: {len(prompt)}")

        # If Ollama format=json worked, response_text should already be JSON.
        # Otherwise, fall back to extracting JSON object from text.
        try:
            return json.loads(response_text)
        except Exception:
            pass

        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(response_text[json_start:json_end])
            except Exception:
                return {"error": "Invalid JSON response", "raw": response_text}

        return {"error": "Could not parse response", "raw": response_text}
    
    def process_document(self, image_path_or_text):
        """Main processing function"""
        
        # Check if input is a file path or text
        if os.path.isfile(image_path_or_text):
            # It's an image file
            if not OCR_AVAILABLE:
                raise RuntimeError("Image processing requires paddleocr. Install: pip install paddleocr")
            document_text = self.extract_text_from_document(image_path_or_text)
        else:
            # Assume it's text
            document_text = image_path_or_text
        
        print(f"\nExtracted text preview:\n{document_text[:200]}...\n")
        
        # Extract fields
        kyc_data = self.extract_kyc_fields(document_text)
        
        return kyc_data

def main():
    # Initialize extractor
    extractor = KYCExtractor()
    
    # Example 1: Test with text (no OCR needed)
    print("="*60)
    print("EXAMPLE 1: Extract from text")
    print("="*60)
    
    # Swedish-style sample text (matches the synthetic image examples)
    sample_text = """
    PASSPORT
    Name: Per Hammarstrom
    Date of Birth: 1957-12-18
    Passport Number: 825832605
    Nationality: Sweden
    Issue Date: 2019-09-08
    Expiry Date: 2027-02-26
    """
    
    result = extractor.process_document(sample_text)
    print("\nExtracted KYC Data:")
    print(json.dumps(result, indent=2))
    
    # Example 2: Test with image (requires OCR)
    print("\n" + "="*60)
    print("EXAMPLE 2: Extract from image")
    print("="*60)
    
    image_path = "sample_documents/passport_1.png"
    if os.path.exists(image_path):
        try:
            result = extractor.process_document(image_path)
            print("\nExtracted KYC Data:")
            print(json.dumps(result, indent=2))
        except RuntimeError as e:
            print(f"Note: {e}")
            msg = str(e)
            if "cudnn" in msg.lower() or "dynamic library" in msg.lower():
                print("Tip: PaddleOCR GPU dependencies are missing; set KYC_OCR_USE_GPU=0 to force CPU, or fix CUDA/cuDNN on PATH.")
            else:
                print("Tip: Ensure paddleocr is installed: pip install paddleocr")
    else:
        print(f"Image not found: {image_path}")
        print("Run 02_generate_synthetic_docs.py first to create sample documents")

if __name__ == "__main__":
    main()
