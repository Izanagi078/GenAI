import re, json
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = PromptTemplate(
    input_variables=["words"],
    template=(
        "Give concise dictionary-style definitions (1–2 sentences) "
        "for each of the following words:\n\n{words}\n\n"
        "Return as JSON mapping word -> definition."
    ),
)

definition_chain = LLMChain(llm=llm, prompt=prompt)

def build_inline_definitions(pdf_text: str) -> dict[str, str]:
    """
    Scan the entire PDF text once and collect inline definitions.
    Example: 'Photosynthesis (the process by which plants...)'
    """
    inline_defs: dict[str, str] = {}
    pattern = r"\b([A-Za-z]+)\s*\(([^)]+)\)"
    for match in re.finditer(pattern, pdf_text):
        word, definition = match.groups()
        inline_defs[word.lower()] = definition.strip()
    return inline_defs

def batch_resolve_with_gemini(unresolved: list[str]) -> dict[str, str]:
    """
    Batch unresolved words through Gemini in one call.
    """
    if not unresolved:
        return {}
    words_str = ", ".join(unresolved)
    result = definition_chain.run(words=words_str)
    try:
        return json.loads(result)
    except Exception:
        # fallback: naive split if JSON parse fails
        return {}
