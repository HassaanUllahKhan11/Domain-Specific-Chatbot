import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "index.faiss"
META_FILE = "index.pkl"


def get_pdf_files(directory: str = "data") -> List[str]:
    return sorted(str(p) for p in Path(directory).glob("*.pdf"))


def extract_pdf_pages(pdf_path: str):
    """Extract page-level text so retrieved chunks can be traced to pages."""
    documents = []

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())

            if text:
                documents.append(
                    {
                        "source": Path(pdf_path).name,
                        "page": page_number,
                        "text": text,
                    }
                )

    return documents


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 300):
    """Character-based chunking that keeps enough context for a 7B local LLM."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_documents(pdf_paths: List[str]):
    documents = []

    for pdf_path in pdf_paths:
        for page in extract_pdf_pages(pdf_path):
            for chunk in chunk_text(page["text"]):
                documents.append(
                    {
                        "text": chunk,
                        "source": page["source"],
                        "page": page["page"],
                    }
                )

    if not documents:
        raise ValueError("No readable text was found in the supplied PDF files.")

    return documents


def build_vector_store(
    pdf_paths: List[str],
    output_dir: str = "vectorstore",
    model_name: str = EMBEDDING_MODEL_NAME,
):
    os.makedirs(output_dir, exist_ok=True)

    documents = build_documents(pdf_paths)
    encoder = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in documents]
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(output_dir, INDEX_FILE))

    import pickle

    with open(os.path.join(output_dir, META_FILE), "wb") as f:
        pickle.dump(
            {
                "documents": documents,
                "embedding_model": model_name,
            },
            f,
        )

    return output_dir


def load_vector_store(
    vector_dir: str = "vectorstore",
    model_name: str = EMBEDDING_MODEL_NAME,
):
    index_path = os.path.join(vector_dir, INDEX_FILE)
    meta_path = os.path.join(vector_dir, META_FILE)

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "FAISS index not found. Upload a PDF and build the index first."
        )

    import pickle

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    encoder = SentenceTransformer(model_name)

    return {
        "index": index,
        "documents": metadata["documents"],
        "encoder": encoder,
    }


class RAGEngine:
    def __init__(
        self,
        vector_store,
        model_path: str,
        use_gpu: bool = False,
        top_k: int = 4,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}\n"
                "Download a compatible Llama GGUF model and set LLM_MODEL_PATH "
                "or enter its path in the sidebar."
            )

        self.index = vector_store["index"]
        self.documents = vector_store["documents"]
        self.encoder = vector_store["encoder"]
        self.top_k = top_k

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1 if use_gpu else 0,
            temperature=0.2,
            top_p=0.9,
            max_tokens=200,
            verbose=False,
        )

    def retrieve(self, question: str):
        query_vector = self.encoder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vector, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[int(idx)].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    @staticmethod
    def _make_prompt(question: str, contexts):
        context_text = "\n\n".join(
            f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
            for item in contexts
        )

        return f"""You are a study assistant for a management textbook.

Answer the user's question ONLY using the supplied textbook context.
- Be direct and factual.
- Keep the answer under 100 words unless a short list is necessary.
- Do not invent facts.
- If the context does not contain the answer, say:
  "I cannot provide an answer based on the provided document."
- Do not ask the user a follow-up question.

TEXTBOOK CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:"""

    def ask(self, question: str) -> Tuple[str, List[str]]:
        contexts = self.retrieve(question)

        if not contexts:
            return (
                "I cannot provide an answer based on the provided document.",
                [],
            )

        prompt = self._make_prompt(question, contexts)

        result = self.llm(
            prompt,
            stop=["</s>", "[/INST]"],
        )

        answer = result["choices"][0]["text"].strip()

        if not answer:
            answer = "I cannot provide an answer based on the provided document."

        sources = [
            f"{item['source']} — page {item['page']} "
            f"(similarity: {item['score']:.3f})"
            for item in contexts
        ]

        return answer, sources


def create_rag_engine(
    vector_store,
    model_path: str,
    use_gpu: bool = False,
):
    return RAGEngine(
        vector_store=vector_store,
        model_path=model_path,
        use_gpu=use_gpu,
    )
