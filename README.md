# # ReasonForge-3B: Fine-Tuning Llama-3.2-3B-Instruct

This project details the complete end-to-end pipeline for fine-tuning, evaluating, exporting, and running a reasoning model using **Unsloth**, **QLoRA**, and **Ollama**.

Check out the [Download the Fine-Tuned Model on Hugging Face](https://huggingface.co/mihirpatel1111/ReasonForge-3B).


## 🚀 Project Overview
- **Base Model**: `unsloth/Llama-3.2-3B-Instruct` (Quantized 4-bit)
- **Fine-Tuned Model**: `ReasonForge-3B`
- **Dataset**: `ServiceNow-AI/R1-Distill-SFT` (v0)
- **Inference & Deployment Engine**: Ollama (using GGUF conversion)

---

## 🛠️ Step-by-Step Workflow

### 1. Installation & Environment Setup
We initialize our environment by installing Unsloth, which accelerates training by up to 2x and significantly reduces memory usage:
```bash
pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 2. Loading the Model & Tokenizer
We load a 4-bit quantized version of the Llama-3.2-3B model using Unsloth's `FastLanguageModel` class:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
```

### 3. Applying PEFT/LoRA Adapters
To optimize parameter-efficient fine-tuning (QLoRA), we attach LoRA adapters onto targeted linear projection layers:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 4. Dataset Structuring & Fine-Tuning
We map our raw reasoning data using an custom `r1_prompt` template and train the adapter using `SFTTrainer` for 30 steps:
```python
trainer_stats = trainer.train()
```

### 5. Native Inference & GGUF Conversion
We test the model natively after patching the chat template for inference. The fine-tuned weights are saved locally and exported directly to a GGUF format `Llama-3.2-3B-Instruct.Q8_0.gguf` using Unsloth's packaging features:
```python
model.save_pretrained_gguf("ReasonForge-3BF", tokenizer)
```

### 6. Local Running via Ollama
We set up and launch Ollama within Colab, load the generated GGUF file via a custom `Modelfile`, and run the model natively through the command line:
```bash
# Creating and launching the ollama service
ollama create reasonforge -f ReasonForge-3BF_gguf/Modelfile
ollama run reasonforge "How many r's are in strawberry?"
```
