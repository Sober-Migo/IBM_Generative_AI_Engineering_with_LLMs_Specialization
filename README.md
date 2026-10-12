# IBM Generative AI Engineering with LLMs Specialization

**IBM · Coursera**  
*Generative AI Engineering with Large Language Models*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2%20%2B%20CUDA%2012.1-ee4c2c.svg)](https://pytorch.org/)
[![Coursera](https://img.shields.io/badge/Coursera-Specialization-0056D2.svg)](https://www.coursera.org/)
[![IBM](https://img.shields.io/badge/IBM-watsonx-054ADA.svg)](https://www.ibm.com/watsonx)

<p align="center">
  <img src="GenAI%20Certificate.png" alt="IBM Generative AI Engineering with LLMs — Certificate" width="720"/>
</p>

<p align="center"><em>Specialization certificate — IBM Generative AI Engineering with LLMs (Coursera)</em></p>

Labs, quizzes, readings, and project materials for the **IBM Generative AI Engineering with LLMs** specialization (7 courses): architecture & data prep → foundational NLP models → transformers → fine-tuning / PEFT / LoRA / QLoRA → instruction tuning, PPO & DPO → RAG & LangChain agents → capstone QA bot.

> Personal solutions and notes for educational reference only. Attempt labs yourself before consulting solutions.

---

## Environment Setup & Installation

> **Why this is at the top:**  
> This specialization mixes **large pretrained weights**, **pinned LangChain stacks**, **bitsandbytes**, and **Windows CUDA builds**. Version drift breaks labs quickly. The steps below recreate a **tested** environment (Python **3.10**, PyTorch **2.2.2+cu121**, Transformers **4.43.4**, LangChain **0.2.7**) so notebooks run without dependency archaeology.

### Important: large models are **not** in Git

Lab weights and datasets are too large for GitHub. They are **downloaded on demand** with `download_assets.py` (see `URL_MAPPING.txt`). After clone, run that script (or choose “yes” inside `setup_env.py`).

**GloVe** is handled differently: notebooks load it via `torchtext.vocab.GloVe` into a shared cache under **`TORCH_HOME`** (not into the repo working directory).

### Recommended: interactive setup

```bash
git clone https://github.com/Sober-Migo/IBM_Generative_AI_Engineering_with_LLMs_Specialization.git
cd IBM_Generative_AI_Engineering_with_LLMs_Specialization

# Conda recommended (matches course notes; env name: llms_env)
python setup_env.py
```

The script will:

1. Create **Conda** (`llms_env`) or **venv**
2. Install the full locked stack (PyTorch, Transformers, PEFT, TRL, LangChain, Chroma, spaCy, …)
3. Set **`KERAS_BACKEND=torch`**
4. Optionally set **`HF_HOME`** and **`TORCH_HOME`** (put caches on a large disk)
5. Register a Jupyter kernel
6. Optionally run **`download_assets.py`**

### Manual Conda path (same pins)

```bash
conda create -n llms_env -c conda-forge python=3.10 -y
conda activate llms_env

# Persist backend + optional cache dirs (adjust paths)
conda env config vars set KERAS_BACKEND=torch
# conda env config vars set HF_HOME=D:/hf_cache
# conda env config vars set TORCH_HOME=D:/torch_cache
conda deactivate && conda activate llms_env

# Then run: python setup_env.py
# or: pip install -r requirements.txt  (after installing CUDA PyTorch — see requirements.txt header)
```

### GloVe cache (torchtext)

By default, `torchtext` looks for `.vector_cache/` in the **current working directory**, which can clutter every lab folder. Notebooks in this repo override that:

```python
import os
from torchtext.vocab import GloVe

torch_home = os.getenv("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
glove_cache = os.path.join(torch_home, "text", ".vector_cache")
glove_vectors_6B = GloVe(name="6B", dim=100, cache=glove_cache)
# Optional larger set: GloVe(name="840B", dim=300, cache=glove_cache)
```

| Setting | Effect |
|---------|--------|
| **`TORCH_HOME` unset** | Vectors land in `~/.cache/torch/text/.vector_cache` |
| **`TORCH_HOME` set** (recommended) | Vectors land in `$TORCH_HOME/text/.vector_cache` — reusable across projects, keeps repo dirs clean |

Set `TORCH_HOME` once in the Conda env (or during `setup_env.py`) and every GloVe lab reuses the same files.

### Download large lab assets (weights / datasets)

```bash
python download_assets.py           # missing files only
python download_assets.py --list    # show URLs + paths
python download_assets.py --force   # re-download all
```

Full URL → path table: **[URL_MAPPING.txt](URL_MAPPING.txt)**.

| Asset | Example path |
|-------|----------------|
| C02 RNN model | `C02 .../assets/lab3_RNN-TR-model.pt` |
| C03 decoder / BERT / transformer | `C03 .../assets/lab*.pt` / `.zip` |
| C04 adapters / LoRA / fine-tuned heads | `C04 .../assets/lab*.pth` |
| C05 PPO archives | `C05 .../assets/lab1_ppo-*.tar.gz` |

GloVe is **not** required from this script when using the notebook pattern above (torchtext downloads into `TORCH_HOME`).

Files marked `__local__` in course materials are **generated inside notebooks** (not downloaded).

### Verify

```bash
python -c "import torch, transformers, peft, langchain; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, langchain.__version__)"
```

---

## About the Specialization

Seven courses covering the full generative-AI engineering path with IBM tooling and open-source libraries:

1. **Architecture & data preparation** for LLMs  
2. **Foundational models** for NLP (n-grams, Word2Vec, seq2seq)  
3. **Transformers** (attention, BERT, GPT-style decoders, translation)  
4. **Fine-tuning** with Hugging Face / PyTorch (adapters, LoRA, QLoRA)  
5. **Advanced fine-tuning** (instruction tuning, reward models, PPO, DPO)  
6. **RAG & LangChain** agents, tools, and prompt patterns  
7. **Capstone**: document loaders, vector stores, retrievers, QA bot (Gradio)

---

## Repository Structure

```text
IBM_Generative_AI_Engineering_with_LLMs_Specialization/
├── GenAI Certificate.png     # Specialization certificate
├── C01 - Generative AI and LLMs Architecture and Data Preparation/
│   └── Module 1–2/   Labs · Quizzes · Readings
├── C02 - Gen AI Foundational Models for NLP & Language Understanding/
│   └── Module 1–2/   Labs · assets · Quizzes · Readings
├── C03 - Generative AI Language Modeling with Transformers/
│   └── Module 1–2/   Labs · assets · Quizzes · Readings
├── C04 - Generative AI Engineering and Fine-Tuning Transformers/
│   └── Module 1–2/   Labs · PEFT / LoRA / QLoRA · Quizzes
├── C05 - Generative AI Advanced Fine-Tuning for LLMs/
│   └── Module 1–2/   Instruction tuning · PPO · DPO
├── C06 - Fundamentals of AI Agents Using RAG and LangChain/
│   └── Module 1–3/   LangChain · RAG · agents · prompts
├── C07 - Project Generative AI Applications with RAG and LangChain/
│   └── Module 1–3/   Loaders · embeddings · retrievers · QA bot
├── setup_env.py              # Interactive env + HF/TORCH cache options
├── download_assets.py        # Fetch large lab models/datasets (CDN)
├── URL_MAPPING.txt           # URL → local path for large assets
├── requirements.txt          # Locked pip pins (see header for PyTorch)
├── LICENSE
├── .gitignore
└── README.md
```

---

## Course Overview

### C01 — Architecture and Data Preparation
Tokenization, data loaders, GenAI architecture overview, hallucinations, watsonx governance readings.

### C02 — Foundational Models for NLP
Document classification, n-gram LMs, feed-forward LMs, Word2Vec (CBOW / Skip-gram), sequence-to-sequence translation.

### C03 — Language Modeling with Transformers
Attention & positional encoding, classification with transformers, GPT-like causal LMs, Baby BERT, data prep for BERT, translation transformers.

### C04 — Engineering and Fine-Tuning Transformers
Hugging Face inference, pre-training (optional), full fine-tuning, **adapters**, **LoRA**, **QLoRA**.

### C05 — Advanced Fine-Tuning for LLMs
Instruction fine-tuning, reward modeling, **PPO**, **DPO**.

### C06 — AI Agents with RAG and LangChain
LangChain tools, RAG (HF + PyTorch), in-context learning, prompt templates, agents from scratch, document summarization with RAG.

### C07 — Capstone Project
Document loaders, text splitters, watsonx embeddings, vector stores, retrievers, Gradio QA bot.

---

## Large Assets (links only)

Run `python download_assets.py` after setup. Full mapping: [URL_MAPPING.txt](URL_MAPPING.txt).

Do **not** force-commit multi‑hundred‑MB weights unless you use **Git LFS**. Specific large paths are listed in `.gitignore`.

---

## Tech Stack

| Area | Packages (pinned in setup) |
|------|----------------------------|
| Runtime | Python 3.10, JupyterLab |
| Deep learning | PyTorch 2.2.2 (+ CUDA 12.1), torchtext, torchdata |
| Hugging Face | transformers 4.43.4, tokenizers, datasets, accelerate, peft, trl, sentence-transformers |
| RAG / agents | LangChain 0.2.7 family, Chroma, FAISS, langchain-ibm, ibm-watsonx-ai |
| NLP classic | spaCy, NLTK, gensim, **GloVe via torchtext** (`TORCH_HOME/text/.vector_cache`) |
| Docs | pypdf, pdfplumber, pymupdf, unstructured |
| Quantization | bitsandbytes 0.43.1 |
| Config | python-dotenv, `KERAS_BACKEND=torch`, optional `HF_HOME` / `TORCH_HOME` |

---

## Certificate

Specialization completion certificate:

![IBM Generative AI Engineering with LLMs Certificate](GenAI%20Certificate.png)

---

## Disclaimer

Educational use only. Course materials belong to **IBM** and **Coursera**. Respect the Honor Code; use this repo as a reference after attempting the work yourself.

---

## License

MIT — see [LICENSE](LICENSE).

---

⭐ If this helps you, consider starring the repo.
