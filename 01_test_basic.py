"""
Step 1: Test KYC extraction with sample text (no documents needed)
Run this first to verify the model works
"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
print("Model loaded!")

# Sample document text (simulating OCR output)
sample_document_text = """
PASSPORT
Name: JOHN MICHAEL SMITH
Nationality: UNITED STATES OF AMERICA
Date of Birth: 15 MAY 1990
Passport Number: 123456789
Issue Date: 10 JAN 2020
Expiry Date: 09 JAN 2030
"""

prompt = f"""Extract KYC information from this document and return as JSON:

{sample_document_text}

JSON Format:
{{
  "name": "",
  "date_of_birth": "",
  "passport_number": "",
  "nationality": "",
  "issue_date": "",
  "expiry_date": ""
}}

Response:"""

print("\nExtracting information...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs, 
    max_length=300, 
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n" + "="*50)
print("EXTRACTION RESULT:")
print("="*50)
print(response)
