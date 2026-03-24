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
from typing import Optional, List, Dict

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
    def __init__(self, model="gemma3:4b"):
        self.model = model
 
    def invoke(self, input_data, config=None):
        if hasattr(input_data, 'to_messages'):
            msg = input_data.to_messages()[0].content
        elif hasattr(input_data, 'content'):
            msg = input_data.content
        else:
            msg = str(input_data)
        return ollama.chat(model=self.model, messages=[{'role': 'user', 'content': msg}]).message.content
 
    def chat(self, prompt_text: str) -> str:
        return ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt_text}]).message.content

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


def find_full_form_batch(
    abbrs: List[str],
    pdf_path: str,
    groq_api_key: Optional[str] = None,
    use_local_llm: bool = False,
    batch_size: int = 10
) -> Dict[str, dict]:
    """
    Find full forms for multiple abbreviations in batched LLM calls.

    VIS Performance Optimization: Reduces LLM calls from O(n) to O(n/batch_size).
    For typical papers with 50 abbreviations:
    - Sequential: 50 calls × 2s = 100 seconds
    - Batched (10): 5 calls × 3s = 15 seconds
    - Speedup: 6.7×

    Args:
        abbrs: List of abbreviations to expand
        pdf_path: Path to PDF for RAG context
        groq_api_key: Groq API key (optional)
        use_local_llm: Whether to use local LLM
        batch_size: Number of abbreviations per LLM call

    Returns:
        Dictionary mapping abbreviation to result dict
    """
    try:
        vectorstore = get_vectorstore(pdf_path, groq_api_key)
        if not vectorstore:
            return {abbr: {"ans": "Error: Could not initialize vector store.", "using_llm": False} for abbr in abbrs}

        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return {abbr: {"ans": "Error: LLM not available.", "using_llm": False} for abbr in abbrs}

        results = {}

        # Process in batches
        for i in range(0, len(abbrs), batch_size):
            batch = abbrs[i:i + batch_size]

            # Retrieve context for all abbreviations in batch
            batch_contexts = []
            for abbr in batch:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced k for batching
                docs = retriever.invoke(f"What is the full form or definition of {abbr}?")
                context = "\n".join(d.page_content for d in docs)
                batch_contexts.append((abbr, context))

            # Format batch data
            batch_str = ""
            for abbr, ctx in batch_contexts:
                batch_str += f"\n\nAbbreviation: {abbr}\nContext: {ctx[:400]}"

            # Build prompt directly (avoids LangChain escaping issues with inline JSON)
            prompt_text = (
                'Extract the full form for each abbreviation using the context provided.\n\n'
                'Return ONLY valid JSON like this example:\n'
                '{"CNN": {"full_form": "Convolutional Neural Network", "source": "extracted"}, '
                '"RNN": {"full_form": "Recurrent Neural Network", "source": "inferred"}}\n\n'
                'Rules: source="extracted" if found in context, "inferred" if guessed, '
                'empty string for unknown.\n\n'
                f'{batch_str}\n\nJSON:'
            )

            # Invoke LLM directly (use chat() for OllamaLLM, invoke() for Groq)
            if hasattr(llm, 'chat'):
                response = llm.chat(prompt_text)
            else:
                from langchain_core.messages import HumanMessage
                result = llm.invoke(HumanMessage(content=prompt_text))
                response = result.content if hasattr(result, 'content') else str(result)

            try:
                clean = response.strip()
                if clean.startswith("```json"):
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]

                parsed = json.loads(clean.strip())

                # Extract results for each abbreviation in batch
                for abbr in batch:
                    if abbr in parsed:
                        ans = parsed[abbr].get("full_form", "NOT_FOUND")
                        source = parsed[abbr].get("source", "inferred")
                        results[abbr] = {
                            "ans": ans,
                            "using_llm": source != "extracted",
                            "context": next((ctx for a, ctx in batch_contexts if a == abbr), "")
                        }
                    else:
                        results[abbr] = {"ans": "NOT_FOUND", "using_llm": True, "context": ""}

            except Exception as e:
                # Fallback to individual processing if batch fails
                print(f"Batch processing failed, falling back to individual: {e}")
                for abbr in batch:
                    results[abbr] = find_full_form(abbr, pdf_path, groq_api_key, use_local_llm)

        return results

    except Exception as e:
        traceback.print_exc()
        return {abbr: {"ans": f"An error occurred: {e}", "using_llm": False, "context": ""} for abbr in abbrs}


_SKIP_WORDS = frozenset({'a', 'an', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'by',
                         'and', 'or', 'nor', 'with', 'from', 'into', 'via', 'de', 'du'})


def _word_initials(words: list, skip_function_words: bool = True) -> str:
    """
    Get initials from a list of words, splitting on hyphens too.
    Skips common function words so 'Association for Computational Linguistics' → 'ACL'.
    """
    initials = []
    for w in words:
        if skip_function_words and w.lower() in _SKIP_WORDS:
            continue
        parts = re.split(r'[-]', w)
        for p in parts:
            if p and p[0].isalpha():
                initials.append(p[0].upper())
    return ''.join(initials)


def _match_initials_to_abbr(words: list, abbr: str) -> bool:
    """Check if the initial letters of words match the abbreviation (in order)."""
    return _word_initials(words) == abbr.upper()


def _is_subsequence(shorter: str, longer: str) -> bool:
    """Check if every character of `shorter` appears in `longer` in order."""
    it = iter(longer)
    return all(c in it for c in shorter)


def _trim_to_abbr_words(candidate: str, abbr: str) -> Optional[str]:
    """
    Find the shortest contiguous suffix of candidate whose word-initials
    match the abbreviation. Falls back to subsequence matching for
    cases like 'Aggression Intensity' → AGI (AI is subsequence of AGI).
    """
    words = [w for w in re.split(r'\s+', candidate.strip()) if w]
    n = len(abbr)
    best_fuzzy = None

    for length in range(max(2, n - 2), min(n + 4, len(words) + 1)):
        for start in range(max(0, len(words) - length), len(words) - length + 1):
            subset = words[start:start + length]
            initials = _word_initials(subset)
            if initials == abbr.upper():
                return ' '.join(subset)  # Exact match — return immediately
            # Fuzzy: initials are subsequence of abbr AND differ by at most 1 char (e.g. AI ⊆ AGI)
            if best_fuzzy is None and len(initials) >= len(abbr) - 1 and _is_subsequence(initials, abbr.upper()):
                best_fuzzy = ' '.join(subset)

    return best_fuzzy


_LIGATURE_MAP = str.maketrans({
    "\uFB00": "ff", "\uFB01": "fi", "\uFB02": "fl",
    "\uFB03": "ffi", "\uFB04": "ffl", "\uFB05": "st",
    "\u2013": "-", "\u2014": "-", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
})


def extract_abbr_definitions_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Scan PDF text for explicit 'Full Form (ABBR)' patterns.
    Returns dict mapping abbreviation → cleaned full form.
    Highest-accuracy path — no LLM required.

    Handles:
    - Unicode ligatures (fi, fl, ffi, ffl → ASCII) common in typeset PDFs
    - Hyphenated line-breaks ("classi-\\nfication" → "classification")
    - Compound-word abbreviations (Socio-Temporal → S+T counts as 2 initials)
    - Function-word skipping (of, in, for, the, a, an, with)
    - Fuzzy initial matching (allows ±1 character for compound expansions)
    """
    import pymupdf
    definitions_map = {}
    try:
        doc = pymupdf.open(pdf_path)
        full_text = " ".join(page.get_text() for page in doc)
        doc.close()

        # Normalize Unicode ligatures → ASCII (critical for typeset PDFs)
        full_text = full_text.translate(_LIGATURE_MAP)

        # Rejoin hyphenated line-breaks: "In-\ntegration" → "Integration"
        full_text = re.sub(r'(\w+)-\s*\n\s*(\w)', lambda m: m.group(1) + m.group(2), full_text)
        collapsed = re.sub(r'\s+', ' ', full_text)

        # Capture up to 10 words before (ABBR) — we'll trim to the right subset
        # Also handle e.g. patterns (e.g., "called X (ABBR)" with preceding punctuation)
        pattern = re.compile(
            r'((?:[A-Za-z\-]+\s+){1,10})\(([A-Z]{2,8})\)'
        )
        for m in pattern.finditer(collapsed):
            candidate, abbr = m.group(1).strip(), m.group(2)
            if abbr in definitions_map:
                continue
            matched = _trim_to_abbr_words(candidate, abbr)
            if matched:
                # Normalize: capitalize content words, keep function words lowercase
                parts = matched.split()
                normalized = [
                    w.lower() if w.lower() in _SKIP_WORDS and i > 0 else w.capitalize()
                    for i, w in enumerate(parts)
                ]
                definitions_map[abbr] = ' '.join(normalized)

    except Exception:
        pass
    return definitions_map


def find_full_form(abbr: str, pdf_path: str, groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> dict:
    try:
        # --- Fast path: regex extraction from raw PDF text (highest accuracy) ---
        regex_map = extract_abbr_definitions_from_pdf(pdf_path)
        if abbr in regex_map:
            return {
                "ans": regex_map[abbr],
                "using_llm": False,
                "context": "",
            }

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
        except Exception:
            ans = response.strip()

        # LLM fallback is always "inferred" — never trust self-reported source from the model
        return {
            "ans": ans,
            "using_llm": True,
            "context": context,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ans": f"An error occurred: {e}", "using_llm": False, "context": ""}


def find_symbol_meaning_batch(
    symbols_with_context: List[tuple],
    pdf_path: str = "",
    groq_api_key: Optional[str] = None,
    use_local_llm: bool = False,
    batch_size: int = 8
) -> Dict[str, dict]:
    """
    Find meanings for multiple symbols in batched LLM calls.

    VIS Performance Optimization: Similar to abbreviation batching.

    Args:
        symbols_with_context: List of (symbol, context) tuples
        pdf_path: Path to PDF for RAG context
        groq_api_key: Groq API key (optional)
        use_local_llm: Whether to use local LLM
        batch_size: Number of symbols per LLM call (smaller than abbr due to context length)

    Returns:
        Dictionary mapping symbol to result dict
    """
    try:
        llm = get_llm(use_local_llm, groq_api_key)
        if not llm:
            return {sym: {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}
                    for sym, _ in symbols_with_context}

        results = {}

        # Process in batches
        for i in range(0, len(symbols_with_context), batch_size):
            batch = symbols_with_context[i:i + batch_size]

            # Build RAG context for batch if PDF provided
            batch_with_rag = []
            if pdf_path:
                vectorstore = get_vectorstore(pdf_path, groq_api_key)
                if vectorstore:
                    for symbol, local_context in batch:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                        docs = retriever.invoke(f"What does the symbol {symbol} represent or mean?")
                        rag_context = "\n".join(d.page_content for d in docs[:2])
                        combined = ""
                        if rag_context:
                            combined += f"Relevant passages: {rag_context[:300]}...\n"
                        if local_context:
                            combined += f"Local context: {local_context}"
                        batch_with_rag.append((symbol, combined))
                else:
                    batch_with_rag = batch
            
            batch_str = ""
            for symbol, ctx in batch_with_rag:
                # Escape backslashes in symbol names so they don't break JSON in the response
                safe_symbol = symbol.replace('\\', '\\\\')
                batch_str += f"\n\nSymbol: {safe_symbol}\nContext: {(ctx[:400] if ctx else 'No context')}"

            # Build prompt directly (avoids LangChain escaping issues with inline JSON)
            prompt_text = (
                'Extract the meaning of each mathematical symbol from the paper context.\n\n'
                'Return ONLY valid JSON. Use simple alphanumeric keys (replace backslashes with nothing).\n'
                'Example: {"alpha": {"meaning": "learning rate", "description": "controls step size", "source": "extracted"}}\n\n'
                'Rules: meaning=1-4 words, source="extracted" if defined in context, "inferred" if guessed, '
                '"NOT_FOUND" if unknown.\n\n'
                f'{batch_str}\n\nJSON:'
            )

            if hasattr(llm, 'chat'):
                response = llm.chat(prompt_text)
            else:
                from langchain_core.messages import HumanMessage
                result = llm.invoke(HumanMessage(content=prompt_text))
                response = result.content if hasattr(result, 'content') else str(result)
            response = response.strip()

            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            try:
                parsed = json.loads(response.strip())

                for symbol, _ in batch:
                    # Try both original key and backslash-stripped key
                    safe_key = symbol.replace('\\', '')
                    data = parsed.get(symbol) or parsed.get(safe_key)
                    if data:
                        results[symbol] = {
                            "meaning": data.get("meaning", "NOT_FOUND"),
                            "description": data.get("description", "NOT_FOUND"),
                            "source": data.get("source", "inferred")
                        }
                    else:
                        results[symbol] = {
                            "meaning": "NOT_FOUND",
                            "description": "NOT_FOUND",
                            "source": "not_found"
                        }
            except Exception as e:
                print(f"Batch symbol processing failed, falling back: {e}")
                for symbol, context in batch:
                    results[symbol] = find_symbol_meaning(symbol, context, pdf_path, groq_api_key, use_local_llm)

        return results

    except Exception:
        traceback.print_exc()
        return {sym: {"meaning": "NOT_FOUND", "description": "NOT_FOUND", "source": "not_found"}
                for sym, _ in symbols_with_context}


def find_symbol_meaning(symbol: str, context: str, pdf_path: str = "", groq_api_key: Optional[str] = None, use_local_llm: bool = False) -> dict:
    try:
        combined_context = ""
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