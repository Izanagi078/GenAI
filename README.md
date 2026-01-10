# Paper Annotator

## Overview

**Paper Annotator** is a comprehensive tool designed to enhance the reading and analysis of research papers. By leveraging Generative AI and PDF processing libraries, it automatically annotates PDFs with helpful context—such as resolved references and abbreviation definitions—directly in the margins.

## Features

- **Smart PDF Annotation**: automatically allows for scaling of PDF pages to create side margins, preventing text overlap.
- **Reference Resolution**: Detects citations (e.g., `[1]`) within the text, looks them up in the bibliography, and uses an LLM to extract and annotate the Title and Year in the margin.
- **Abbreviation Expansion**: Identifies abbreviations, finds their definitions within the document using RAG (Retrieval-Augmented Generation), and annotates them.

## Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/), Uvicorn
- **PDF Processing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- **LLM & Orchestration**: [LangChain](https://www.langchain.com/), Google Generative AI (Gemini)
- **Embeddings & Vector Store**: HuggingFace Embeddings, FAISS
- **Language**: Python 3.x

## Project Structure

```
GenAI/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application entry point
│   │   └── config.py        # Configuration (API keys, etc.)
│   ├── models/              # Pydantic models (Schemas)
│   ├── services/            # Core logic
│   │   ├── definitions.py   # RAG & LLM logic for definitions
│   │   ├── parser.py        # Text parsing logic
│   │   └── pdf_transform.py # PDF manipulation (scaling, annotation)
│   └── requirements.txt     # Python dependencies
├── docs/                    # Documentation & Plans
├── uploads/                 # Temporary storage for uploads & indices
└── README.md
```

## Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd GenAI
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment.

   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the `backend/app` directory (or where `config.py` expects it) and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   ```

## Usage

1. **Run the Server**:
   From the root directory of the project:

   ```bash
   uvicorn backend.app.main:app --reload
   ```

2. **Access the API**:
   Open your browser and navigate to the Swagger UI:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

3. **Endpoints**:
   - **POST /annotate/**: Upload a PDF to receive an annotated version with references and abbreviations resolved in the margins.
