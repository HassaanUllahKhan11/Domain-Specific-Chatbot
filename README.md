# 📚 Management Study Assistant

A local Retrieval-Augmented Generation (RAG) chatbot that answers questions from management textbooks and other PDF study material.

The original project used Google Colab, Google Drive paths, LangChain, FAISS, Sentence Transformers, Llama 2 7B GGUF, and Streamlit. This version turns that notebook workflow into a standalone GitHub-ready application.

## Features

- 📄 Upload one or multiple PDF textbooks
- 🔎 Extract and chunk PDF text
- 🧠 Generate semantic embeddings with `all-MiniLM-L6-v2`
- ⚡ Store and search embeddings with FAISS
- 🤖 Generate answers with a local Llama GGUF model
- 💬 Streamlit chat interface
- 📚 Show retrieved source pages
- 🔒 Runs locally; uploaded documents and the LLM can remain on your machine
- 🖥️ Optional GPU acceleration when `llama-cpp-python` is installed with GPU support

## Architecture

```text
PDF files
   │
   ▼
PyPDF2 text extraction
   │
   ▼
Chunking
   │
   ▼
Sentence Transformers
(all-MiniLM-L6-v2)
   │
   ▼
FAISS vector index
   │
   ├──────────────► semantic retrieval
   │                         │
User question ───────────────┘
                             ▼
                    Retrieved context
                             │
                             ▼
                    Llama 2 GGUF (local)
                             │
                             ▼
                         Answer + sources
                             │
                             ▼
                       Streamlit UI
```

## Project structure

```text
management-rag-chatbot/
├── app.py                  # Streamlit frontend
├── rag.py                  # PDF, embeddings, FAISS and LLM pipeline
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep            # Keep PDFs out of Git
├── models/
│   └── .gitkeep            # Keep GGUF model files out of Git
└── vectorstore/
    └── .gitkeep            # Generated FAISS files
```

## Requirements

- Python 3.10 or 3.11 recommended
- 8 GB+ RAM recommended for a 7B GGUF model
- CPU works, but local LLM generation can be slow
- NVIDIA GPU is optional

## 1. Clone the repository

```bash
git clone https://github.com/HassaanUllahKhan11/management-rag-chatbot.git
cd management-rag-chatbot
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Windows note for `llama-cpp-python`

If the normal installation fails because a C/C++ build environment is missing, install the package using a compatible prebuilt wheel or follow the official `llama-cpp-python` installation instructions for your platform.

For NVIDIA CUDA acceleration, install a CUDA-enabled build appropriate for your CUDA version. The application itself supports GPU layers through the sidebar.

## 4. Add a compatible GGUF model

The application expects a local GGUF model.

Place it under:

```text
models/
```

For example:

```text
models/
└── llama-2-7b-chat.Q5_K_M.gguf
```

Do **not** commit the GGUF model to GitHub. The `.gitignore` already excludes it.

The original notebook used:

```text
TheBloke/Llama-2-7b-Chat-GGUF
llama-2-7b-chat.Q5_K_M.gguf
```

Use a model you are legally permitted to download and redistribute/use.

## 5. Start the application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

## 6. Build the knowledge base

1. Open the application.
2. Upload your PDF textbook(s).
3. Click **Build / Rebuild Index**.
4. Wait for the FAISS index to finish.
5. Enter questions in the chat box.

The generated files are stored in:

```text
vectorstore/
├── index.faiss
└── index.pkl
```

These files are intentionally excluded from Git because they can be regenerated from the source PDFs.

## Configuration

You can provide the model path through an environment variable:

Windows PowerShell:

```powershell
$env:LLM_MODEL_PATH="models/llama-2-7b-chat.Q5_K_M.gguf"
streamlit run app.py
```

macOS/Linux:

```bash
export LLM_MODEL_PATH="models/llama-2-7b-chat.Q5_K_M.gguf"
streamlit run app.py
```

Or simply enter the path in the Streamlit sidebar.

## Original notebook workflow

The original implementation performed these major steps:

1. Read the management textbook with PyPDF2.
2. Split extracted content into passages.
3. Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Store embeddings in FAISS.
5. Load a local Llama 2 7B GGUF model.
6. Retrieve relevant passages for a user question.
7. Generate an answer using the retrieved document context.
8. Run the interface through Streamlit.

The standalone version preserves that core RAG approach while removing Google Colab and hard-coded Google Drive dependencies.

## Important limitations

- PDF extraction quality depends on the source PDF. Scanned/image-only PDFs require OCR.
- A local 7B model may be slow on CPU.
- The chatbot is designed to answer from retrieved textbook context; it is not a general-purpose factual assistant.
- The generated answer can still contain LLM errors, so textbook material should be checked for important academic work.
- The FAISS index must be rebuilt when the underlying PDFs change.

## Privacy

The intended deployment is local. PDFs, embeddings, and the GGUF model are stored locally unless you explicitly move or deploy the project elsewhere.

## License

Add the license appropriate for your own code and project requirements. Do not assume that the textbook or downloaded LLM model is covered by your project's license.
