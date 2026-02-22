<p align="center">
  <h1 align="center">Glosser</h1>
  <p align="center">
    <strong>Automatically annotate research PDFs with citation titles and abbreviation expansions, right in the margins.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/glosser/"><img alt="PyPI" src="https://img.shields.io/pypi/v/glosser?color=FFD700&style=for-the-badge&logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/glosser/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/glosser?color=3776AB&style=for-the-badge&logo=python&logoColor=white"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge"></a>
  </p>
</p>

---

## 🤔 What is Glosser?

Ever read a dense research paper and had to constantly flip to the references section just to see what **[17]** is? Or wondered what **LSTM** stands for on page 8?

**Glosser** solves this. It reads your PDF, identifies every citation marker (like `[1]`, `[2]`, …) and every uppercase abbreviation, then writes their full titles and expansions **directly into the page margins**, producing a new, self-contained PDF you can read without jumping around.

### Before → After

| Without Glosser | With Glosser |
|:--- |:--- |
| `...as shown in [14]...` | `...as shown in [14]...` **← margin: *"Attention Is All You Need (2017)"*** |
| `...using the BERT model...` | `...using the BERT model...` **← margin: *"Bidirectional Encoder Representations from Transformers"*** |

---

## ✨ Key Features

- 📄 **Citations in Margins**: Extracts titles and years from the bibliography using LLMs and places them exactly where they are cited.
- 🔤 **Context-Aware Abbreviations**: Identifies abbreviations and finds their definitions within the document using an intelligent RAG (Retrieval-Augmented Generation) pipeline.
- 🎨 **Visual Confidence**: Color-coded annotations (Green for high-confidence document matches, Red for LLM-inferred definitions).
- ⚡ **Zero-Config CLI**: A simple interactive CLI that remembers your API keys and preferences.

---

## 🚀 Quick Start

### Installation

```bash
pip install -q glosser
```

> **Note:** The `-q` flag keeps the install output clean. Glosser has ML dependencies (PyTorch, Transformers, FAISS) so the first install may take a few minutes.

### Requirements

- **Python**: 3.9 or higher
- **API Key**: A free [Groq API key](https://console.groq.com/keys)

---

## 📖 How to Use

### 1. Command Line Interface (CLI)
The easiest way to use Glosser is the built-in interactive CLI, simply run:

```bash
glosser
```
Follow the prompts to select your PDF and set your preferences. Your Groq API key will be safely stored locally for future use.

### 2. Python Library
Integrate Glosser directly into your own automation scripts:

```python
import asyncio
from glosser.main import annotate

async def main():
    out_path, count = await annotate(
        path="my_paper.pdf",
        GROQ_API_KEY="gsk_...",
        scaling=1.2,               # margin width multiplier
        find_references=True,      # toggle citation lookups
        find_abbreviation=True,    # toggle abbreviation expansion
    )
    print(f"Success! Saved {count} annotations to: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚙️ Technical Architecture

Glosser transforms a standard PDF into an augmented reading experience through a multi-stage pipeline:

1. **Page Scaling**: Re-layouts the PDF to create a dedicated annotation column on the right.
2. **Citation Mapping**: Parses the 'References' or 'Bibliography' section to map indices to titles.
3. **Internal RAG**: 
   - Chunks the PDF text.
   - Generates vector embeddings (HuggingFace MiniLM).
   - Performs similarity searches on abbreviations to find candidate definitions.
4. **LLM Refinement**: Uses state-of-the-art LLMs (via Groq) to finalize the most accurate definitions and titles.
5. **Annotation Layer**: Renders the final text onto the PDF using PyMuPDF.

---

## 🛠️ Configuration & Customization

| Variable | Default | Description |
|---|---|---|
| `scaling` | `1.2` | Page width multiplier. Increase for wider margins. |
| `find_references` | `True` | Whether to process citation markers like `[12]`. |
| `find_abbreviation` | `True` | Whether to use RAG to expand uppercase abbreviations. |

Your preferences and API keys are stored at `~/.glosser_config`.

---

<p align="center">
  Built with ❤️ for the research community.
</p>
