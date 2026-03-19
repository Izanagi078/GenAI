<p align="center">
  <h1 align="center">Glosser</h1>
  <p align="center">
    <strong>Automatically annotate research PDFs with citation titles, abbreviation expansions, and symbol definitions, right in the margins.</strong>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge"></a>
  </p>
</p>

---

## 🤔 What is Glosser?

Ever read a dense research paper and had to constantly flip to the references section just to see what **[17]** is? Or wondered what **LSTM** stands for on page 8? Or what the Greek letter **π** means in equation 3?

**Glosser** solves this. It reads your PDF, identifies every citation marker (like `[1]`, `[2]`, …), every uppercase abbreviation, and every mathematical symbol, then writes their full meanings **directly into the page margins** — producing a new, self-contained PDF you can read without jumping around. Everything runs **fully offline** using a local Small Language Model (SLM) via [Ollama](https://ollama.com/).

### Before → After

| Without Glosser | With Glosser |
|:--- |:--- |
| `...as shown in [14]...` | `...as shown in [14]...` **← margin: *"Attention Is All You Need (2017)"*** |
| `...using the BERT model...` | `...using the BERT model...` **← margin: *"Bidirectional Encoder Representations from Transformers"*** |
| `...policy π is optimized...` | `...policy π is optimized...` **← margin: *π = robot policy*** |

---

## ✨ Key Features

- 📄 **Citation Resolution** — Extracts titles and years from the bibliography and places them exactly where they are cited. Supports both numeric `[1]` and author-year `(Smith et al., 2020)` formats.
- 🔤 **Context-Aware Abbreviation Expansion** — Identifies uppercase abbreviations and finds their definitions within the document using a RAG (Retrieval-Augmented Generation) pipeline.
- 🔣 **Symbol Grounding** — Detects Greek letters, math operators, and LaTeX symbols from equation blocks (via LatexOCR), then uses the SLM to assign meanings from surrounding context. Rendered as small images using Matplotlib.
- 🎨 **Color-Coded Confidence** — Green annotations for high-confidence matches extracted from the document; red for SLM-inferred definitions. Low-confidence results are silently discarded.
- 🛡️ **SLM Critique Layer** — A second independent SLM call verifies abbreviation and symbol results before they are written to the PDF, filtering out hallucinations.
- 🔒 **Fully Offline** — Runs entirely on your machine using a local SLM (Qwen2.5:1.5b via Ollama). No data leaves your computer.

---

## 📋 Prerequisites

Before cloning and running Glosser, you need the following set up on your system:

### 1. Python 3.9+
Make sure you have Python 3.9 or higher installed.

### 2. Ollama (Local LLM Runtime)
Glosser uses **Qwen2.5:1.5b** as its local Small Language Model, served through [Ollama](https://ollama.com/).

**Install Ollama:**
- **Windows / macOS / Linux**: Download from [ollama.com](https://ollama.com/) and follow the install instructions.

**Pull the required model:**
```bash
ollama pull qwen2.5:1.5b
```

**Start the Ollama server** (if it isn't already running):
```bash
ollama serve
```

> **Note:** Ollama must be running in the background before you use Glosser. The model is ~1 GB and runs on CPU — no GPU required.

---

## 🚀 Getting Started

### Clone the Repository
```bash
git clone https://github.com/Izanagi078/GenAI.git
cd GenAI
```

### Install Dependencies
```bash
cd package
pip install .
```

> **Note:** Glosser has ML dependencies (PyTorch, Transformers, FAISS, sentence-transformers) so the first install may take a few minutes.

---

## 📖 How to Use

The easiest way to use Glosser is through its interactive command-line interface. Simply run:

```bash
glosser
```

Follow the prompts to select your PDF and set your preferences. Your settings (and API keys, if used) will be remembered for future runs.

> [!TIP]
> **No local SLM?** If you can't run Ollama locally, you can choose to use the cloud-based **Groq API** instead. Just provide your [Groq API key](https://console.groq.com/keys) when prompted by the CLI.

---

## 🏗️ Architecture — 4-Phase Pipeline

Glosser transforms a standard PDF into an augmented reading experience through four distinct phases:

### Phase 1 — Parsing
The PDF page width is scaled by 1.2× using PyMuPDF, creating blank side margins while keeping original content centered. Three parsers then run on the original document:
- **Citations** — Scans backward from the last page to find the References section. Extracts entries and matches them against citation tags in the body (`[1]`, `(Smith et al., 2020)`).
- **Abbreviations** — A regex picks up all-caps tokens appearing at least twice, filtering out author-list and bibliography noise.
- **Symbols** — Two layers: a Unicode range scan for Greek letters and math operators, plus a heuristic equation-block detector that crops the block and runs LatexOCR to extract LaTeX symbol strings.

### Phase 2 — SLM Extraction
For each unique item, the local SLM (Qwen2.5:1.5b via Ollama) is queried:
- **Citations** — The SLM parses raw reference text and returns title + year as JSON. Regex handles simple cases; the SLM handles harder author-year entries.
- **Abbreviations (RAG)** — The PDF is chunked (500 chars, 100 overlap), embedded with `all-MiniLM-L6-v2`, and stored in a FAISS index (cached to disk). For each abbreviation, the top-3 relevant chunks are retrieved and given to the SLM as context.
- **Symbols** — The SLM receives the symbol plus surrounding context and returns a short meaning, a description, and a source tag (extracted / inferred / not found).

### Phase 3 — SLM Critique
Before writing anything to the PDF, a second independent SLM call verifies each abbreviation and symbol result. Given the item, proposed expansion/meaning, and context, it returns **HIGH**, **MEDIUM**, or **LOW** confidence. HIGH and MEDIUM pass through; LOW is silently discarded. This prevents hallucinated definitions from appearing on the final PDF. Citations skip this stage since they are verified directly against the references section.

### Phase 4 — Margin Rendering
Each annotation target has a bounding box. Glosser checks which half of the page it falls in and assigns it to the left or right margin. Before placing text, it checks for spatial collisions — if the area is already taken, that occurrence is skipped. Annotations are rendered at 5pt font using PyMuPDF's `insert_textbox`. For math symbols, the LaTeX is rendered as a small PNG via Matplotlib and inserted as an image alongside the meaning text. **Green** = high confidence, **Red** = inferred.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `scaling` | `1.2` | Page width multiplier. Increase for wider margins. |
| `find_references` | `True` | Whether to annotate citation markers like `[12]`. |
| `find_abbreviation` | `True` | Whether to use RAG to expand uppercase abbreviations. |
| `find_symbols` | `True` | Whether to detect and explain math symbols. |
| `use_local_llm` | `True` | Use local Qwen2.5:1.5b via Ollama. Set `False` to use Groq cloud API. |
| `GROQ_API_KEY` | `None` | Groq API key (only needed when `use_local_llm=False`). |

---

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| PDF read/write | PyMuPDF |
| Local SLM | Qwen2.5:1.5b via Ollama |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector search | FAISS |
| Symbol OCR | pix2tex (LatexOCR) |
| Symbol rendering | Matplotlib |
| Cloud LLM (optional) | Groq |

---

<p align="center">
  Built with ❤️ for the research community.
</p>
