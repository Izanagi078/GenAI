<p align="center">
  <h1 align="center">glosser</h1>
  <p align="center">
    <strong>Automatically annotate research PDFs with citation titles and abbreviation expansions, right in the margins.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/glosser/"><img alt="PyPI" src="https://img.shields.io/pypi/v/glosser?color=blue&logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/glosser/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/glosser?color=green&logo=python&logoColor=white"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  </p>
</p>

---

## 🤔 What is Glosser?

Ever read a dense research paper and had to constantly flip to the references section just to see what **[17]** is? Or wondered what **LSTM** stands for on page 8?

**Glosser** solves this. It reads your PDF, identifies every citation marker (like `[1]`, `[2]`, …) and every uppercase abbreviation, then writes their full titles and expansions **directly into the page margins**, producing a new, self-contained PDF you can read without jumping around.

### Before → After

| Without Glosser | With Glosser |
|---|---|
| `...as shown in [14]...` | `...as shown in [14]...` **← margin: *"Attention Is All You Need (2017)"*** |
| `...using the BERT model...` | `...using the BERT model...` **← margin: *"Bidirectional Encoder Representations from Transformers"*** |

---

## ✨ Features

- 📄 **Citation Annotations**: Extracts titles and years from the references section using LLM and writes them next to each `[n]` marker in the margins.
- 🔤 **Abbreviation Expansion**: Finds uppercase abbreviations (3–5 chars), looks up their full forms via RAG + LLM, and annotates them in the margins.
- 🎨 **Color-Coded**: Green for context-extracted definitions, red for LLM-inferred ones, so you know the confidence level at a glance.
- 🔑 **API Key Saved Locally**: Enter your Groq API key once; it's stored at `~/.glosser_config` for future runs.

---

## 🚀 Installation

```bash
pip install -q glosser
```

> **Note:** The `-q` flag keeps the install output clean. Glosser has ML dependencies (PyTorch, Transformers, FAISS) so the first install may take a few minutes.

### Requirements

- Python ≥ 3.9
- A free [Groq API key](https://console.groq.com/keys)

---

## 📖 Usage

### CLI (recommended)

Simply run:

```bash
glosser
```

### As a Python Library

```python
import asyncio
from glosser.main import annotate

async def main():
    out_path, count = await annotate(
        path="paper.pdf",
        GROQ_API_KEY="gsk_...",
        scaling=1.2,               # margin width multiplier (default: 1.2)
        find_references=True,      # annotate [n] citations
        find_abbreviation=True,    # annotate abbreviations
    )
    print(f"Saved {count} annotations to {out_path}")

asyncio.run(main())
```

---

## ⚙️ How It Works

```
PDF Input
   │
   ├─ 1. Scale pages horizontally to create side margins
   │
   ├─ 2. Extract references section → LLM extracts title + year for each [n]
   │
   ├─ 3. Find all [n] markers in body text → write citation info in nearest margin
   │
   ├─ 4. Find all uppercase abbreviations (≥2 occurrences)
   │
   ├─ 5. RAG pipeline: chunk PDF → embed with MiniLM → FAISS similarity search
   │     └─ LLM resolves full form from retrieved context
   │
   └─ 6. Write abbreviation expansions in margins → save new PDF
```

**Tech stack:** PyMuPDF · LangChain · Groq (Kimi K2) · HuggingFace Embeddings (MiniLM-L6-v2) · FAISS

---

## 🛠️ Configuration

| Option | Default | Description |
|---|---|---|
| `scaling` | `1.2` | How much to widen pages for margins (1.0 = no extra margin) |
| `find_references` | `True` | Annotate `[n]` citation markers |
| `find_abbreviation` | `True` | Annotate uppercase abbreviations |

Your Groq API key is stored locally at `~/.glosser_config` after the first run.  
To reset it, simply delete that file.

