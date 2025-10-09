import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load OpenAI API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize GPT-4 model via LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

# Prompt template for resolving full forms
prompt = PromptTemplate.from_template(
    """Given the following document:\n{document}\nWhat does "{abbr}" stand for in this document?"""
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Function to resolve full forms from abbreviations
def resolve_full_forms(abbrevs, pages):
    full_text = "\n".join(pages)
    resolved = {}

    if not abbrevs:
        return resolved

    for abbr in abbrevs:
        print(f"Resolving: {abbr}")
        try:
            answer = chain.run(document=full_text, abbr=abbr).strip()
            resolved[abbr] = answer if answer else "Not found in document"
        except Exception as e:
            resolved[abbr] = f"Error: {str(e)}"

    return resolved
