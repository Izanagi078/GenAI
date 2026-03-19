import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["VERBOSITY"] = "ERROR"

import json
import logging
import traceback
import warnings
import re
from typing import Optional, List

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

for logger_name in ["langchain_huggingface", "transformers", "huggingface_hub", "sentence_transformers", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import transformers
transformers.utils.logging.set_verbosity_error()
try:
    import transformers.models.bert.modeling_bert
    logging.getLogger("transformers.models.bert.modeling_bert").setLevel(logging.ERROR)
except ImportError:
    pass

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_groq import ChatGroq
import ollama
from dotenv import load_dotenv
load_dotenv()

class OllamaLLM(Runnable):
    def __init__(self, model="qwen2.5:1.5b"):
        self.model = model
    def invoke(self, input_data, config=None):
        msg = input_data.to_messages()[0].content if hasattr(input_data, 'to_messages') else str(input_data)
        return ollama.chat(model=self.model, messages=[{'role': 'user', 'content': msg}]).message.content

_cached_embeddings = None
_cached_vectorstores = {}


def _clean_reference_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\s*(?:\[\d+\]|\d+[\.]|\d+[\)])\s*", "", text)
    return text


def _extract_title_year_from_reference_regex(reference_text: str) -> dict:
    cleaned = _clean_reference_text(reference_text)
    year_match = re.search(r"\b((?:19|20)\d{2})(?:[a-z])?\b", cleaned, flags=re.IGNORECASE)
    year = year_match.group(1) if year_match else None

    title = None
    if year_match:
        tail = cleaned[year_match.end():]
        tail = re.sub(r"^[\s\)\]\.\,:;\-]+", "", tail)
        parts = [p.strip(" \t\n\r\"'") for p in re.split(r"\.\s+", tail) if p.strip()]
        if parts:
            title = parts[0]

    if (not title or len(title) < 4) and cleaned:
        parts = [p.strip(" \t\n\r\"'") for p in re.split(r"\.\s+", cleaned) if p.strip()]
        if len(parts) >= 2 and year and year in parts[0]:
            title = parts[1]
        elif len(parts) >= 3:
            title = parts[1]

    if title:
        title = re.sub(r"\s+", " ", title).strip(" .;:,-")
        if len(title) < 4:
            title = None

    return {"title": title, "year": year}

def get_embeddings():
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return _cached_embeddings

def get_llm(use_local_llm: bool, groq_api_key: Optional[str] = None):
    if use_local_llm:
        return OllamaLLM()
    if not groq_api_key:
        return None
    return ChatGroq(
        model="moonshotai/kimi-k2-instruct-0905",
        temperature=0,
        api_key=groq_api_key,
    )

def get_vectorstore(pdf_path, groq_api_key):
    """
    Load or create a FAISS vector store for the given PDF.
    Caches the vector store in memory for repeated lookups.
    """
    if pdf_path in _cached_vectorstores:
        return _cached_vectorstores[pdf_path]

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    vectorstore_path = f"{pdf_path}.faiss"
    embeddings = get_embeddings()
    
    vectorstore = None
    try:
        if os.path.exists(vectorstore_path):
            vectorstore = FAISS.load_local(
                vectorstore_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(vectorstore_path)
    except Exception:
        # Recreate if loading fails
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vectorstore_path)

    if vectorstore:
        _cached_vectorstores[pdf_path] = vectorstore
    return vectorstore

def extract_title_year_from_reference(reference_text: str, groq_api_key: Optional[str] = None, target_author: Optional[str] = None, target_year: Optional[str] = None, use_local_llm: bool = False) -> Optional[dict]:
    """
    Extract title and year from a reference citation text using LLM.
    """
    try:
        api_key = groq_api_key
        # Regex is too inaccurate for titles (often picks authors or journals).
        fallback = _extract_title_year_from_reference_regex(reference_text)
        
        hint = ""
        if target_author and target_year:
            hint = f"Specifically for the reference authored by '{target_author}' in '{target_year}'."

        template = f"""You are a precise bibliographic data extractor. Extract the main paper title and publication year from the given reference text.
{hint}

Instructions:
1. Return ONLY a JSON object with "title" and "year" keys.
2. The "title" should be the full, exact title of the paper/article.
3. The "year" should be a 4-digit number (e.g., 2020).
4. If a field is not found, use "NOT_FOUND".
5. Do not include any citations, authors, or journal names in the title field.

Reference:
{{reference_text}}

Response Format:
{{{{ "title": "...", "year": "..." }}}}"""

        prompt = ChatPromptTemplate.from_template(template)

        llm = get_llm(use_local_llm, api_key)
        
        if not llm:
            return fallback if fallback.get("title") or fallback.get("year") else None

        response = (prompt | llm | StrOutputParser()).invoke({"reference_text": reference_text})
        response = response.strip()

        if response.startswith("```json"): response = response[7:]
        elif response.startswith("```"): response = response[3:]
        if response.endswith("```"): response = response[:-3]

        import json
        try:
            res_json = json.loads(response.strip())
            title = res_json.get("title")
            year = str(res_json.get("year"))

            if title == "NOT_FOUND": title = None
            if year == "NOT_FOUND": year = None

            # Final cleanup
            if not title: title = fallback.get("title")
            if not year: year = fallback.get("year")
            else:
                y_match = re.search(r"\b((?:19|20)\d{2})\b", year)
                year = y_match.group(1) if y_match else fallback.get("year")

            if title or year:
                return {"title": title, "year": year}
        except (json.JSONDecodeError, TypeError):
            pass

        return fallback if fallback.get("title") or fallback.get("year") else None

    except Exception as e:
        print(f"Error extracting title/year from reference: {e}")
        fallback = _extract_title_year_from_reference_regex(reference_text)
        return fallback if fallback.get("title") or fallback.get("year") else None


def find_full_form(abbr: str, pdf_path: str, groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> dict:
    try:
        vectorstore = get_vectorstore(pdf_path, groq_api_key)
        if not vectorstore:
            return {"ans": "Error: Could not initialize vector store.", "using_llm": False}

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(f"What is the full form or definition of {abbr}?")
        context = "\n\n".join(d.page_content for d in docs)

        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return {"ans": "Error: LLM not available.", "using_llm": False}

        template = """
        You are an information extraction system.

        Task: Find the FULL FORM of a given abbreviation.

        Return ONLY a valid JSON object. No extra text.

        Output format:
        {{
            "full_form": "<only the full form or empty string>",
            "source": "extracted" or "inferred"
        }}

        STRICT RULES:
        - Output EXACTLY one JSON object. Nothing before or after.
        - "full_form" must be a SHORT phrase, not a sentence.
        - Do NOT include explanations, examples, or extra words.
        - Do NOT repeat the abbreviation.
        - If multiple candidates exist, choose the most relevant one.
        - If full form is explicitly written in the context → "extracted"
        - If you reasonably guess it → "inferred"
        - "full_form" must contain ONLY the expansion words, never a sentence fragment or verb phrase leading into the term.

        GOOD OUTPUT EXAMPLE:
        {{
            "full_form": "Scanning Electron Microscope",
            "source": "extracted"
        }}

        Context:
        {context}

        Abbreviation: {abbreviation}
        """

        prompt = ChatPromptTemplate.from_template(template)
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "abbreviation": abbr
        })

        try:
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            parsed = json.loads(clean.strip())
            ans = parsed.get("full_form", "NOT_FOUND")
            source = parsed.get("source", "inferred")
        except Exception:
            ans = response.strip()
            source = "inferred"

        return {
            "ans": ans,
            "using_llm": source != "extracted",
            "context": context,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ans": f"An error occurred: {e}", "using_llm": False, "context": ""}


def find_symbol_meaning(symbol: str, context: str, pdf_path: str = "", groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> dict:
    try:
        rag_context = ""
        if pdf_path:
            vectorstore = get_vectorstore(pdf_path, groq_api_key)
            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(f"What does the symbol {symbol} represent or mean?")
                rag_context = "\n\n".join(d.page_content for d in docs)

        combined_context = ""
        if rag_context:
            combined_context += "Relevant passages from the paper:\n" + rag_context + "\n\n"
        if context:
            combined_context += "Local context where the symbol appears:\n" + context

        if not combined_context.strip():
            return {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}

        template = """You are extracting mathematical symbol definitions from a research paper.

Return ONLY a valid JSON object. No extra text.

Output format:
{{
  "meaning": "<1-4 word name for the symbol>",
  "description": "<one sentence describing what it represents>",
  "source": "extracted" or "inferred"
}}

STRICT RULES:
- Read ALL the provided context carefully to understand what the symbol represents.
- Do NOT just pick the word that appears immediately before or after the symbol. That adjacent word is often unrelated to the symbol's true meaning.
- Look for explicit definitions like "X denotes ...", "X represents ...", "X is the ...", "where X is ...".
- "meaning" must be a SHORT noun phrase (1-4 words), like "learning rate", "reward function", "robot policy".
- "description" must explain what the symbol represents in the paper.
- If the meaning is explicitly defined in the context → source = "extracted"
- If you reasonably infer it → source = "inferred"
- If you cannot determine the meaning → set meaning to "NOT_FOUND"

Context:
{context}

Symbol: {symbol}
"""

        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}

        prompt = ChatPromptTemplate.from_template(template)

        response = (prompt | llm | StrOutputParser()).invoke({
            "symbol": symbol,
            "context": combined_context
        }).strip()

        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            res = json.loads(response)
            return {
                "meaning": res.get("meaning", "NOT_FOUND"),
                "description": res.get("description", "NOT_FOUND"),
                "source": res.get("source", "inferred")
            }
        except:
            return {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}

    except Exception:
        traceback.print_exc()
        return {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}


def critique_abbr(abbr: str, expansion: str, context: str, groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> str:
    """
    Stage-2 critique: a second SLM call that acts as an independent judge,
    evaluating whether `expansion` is correct for `abbr` in context.
    Returns "HIGH", "MEDIUM", or "LOW".
    """
    try:
        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return "MEDIUM"

        template = """You are verifying whether a proposed abbreviation expansion is correct.

Abbreviation: {abbr}
Proposed expansion: {expansion}
Document context:
{context}

Is "{expansion}" the correct full form for "{abbr}" in this context?
Return ONLY a JSON object with keys "confidence" (HIGH, MEDIUM, or LOW) and "reason" (one sentence).
- HIGH: expansion is clearly correct and consistent with the context.
- MEDIUM: expansion is plausible but uncertain or not directly confirmed in context.
- LOW: expansion is likely wrong, irrelevant, or cannot be verified from context.

{{"confidence": "...", "reason": "..."}}"""

        prompt = ChatPromptTemplate.from_template(template)
        response = (prompt | llm | StrOutputParser()).invoke({
            "abbr": abbr,
            "expansion": expansion,
            "context": context,
        }).strip()

        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        parsed = json.loads(response.strip())
        confidence = parsed.get("confidence", "MEDIUM").upper()
        return confidence if confidence in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
    except Exception:
        return "MEDIUM"


def critique_sym(symbol: str, meaning: str, context: str, groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> str:
    """
    Stage-2 critique: a second SLM call that acts as an independent judge,
    evaluating whether `meaning` is correct for `symbol` in context.
    Returns "HIGH", "MEDIUM", or "LOW".
    """
    try:
        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return "MEDIUM"

        template = """You are verifying whether a proposed symbol meaning is correct.

Symbol: {symbol}
Proposed meaning: {meaning}
Document context:
{context}

Is "{meaning}" the correct meaning for "{symbol}" in this context?
Return ONLY a JSON object with keys "confidence" (HIGH, MEDIUM, or LOW) and "reason" (one sentence).
- HIGH: meaning is clearly stated or strongly implied in the context.
- MEDIUM: meaning is plausible but not explicitly confirmed in context.
- LOW: meaning is likely wrong or cannot be verified from context.

{{"confidence": "...", "reason": "..."}}"""

        prompt = ChatPromptTemplate.from_template(template)
        response = (prompt | llm | StrOutputParser()).invoke({
            "symbol": symbol,
            "meaning": meaning,
            "context": context,
        }).strip()

        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        parsed = json.loads(response.strip())
        confidence = parsed.get("confidence", "MEDIUM").upper()
        return confidence if confidence in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
    except Exception:
        return "MEDIUM"