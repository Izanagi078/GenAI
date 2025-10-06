import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load OpenAI API key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]

# Initialize GPT-4 model via LangChain
llm = ChatOpenAI(model="gpt-4", temperature=0)

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
    for abbr in abbrevs:
        answer = chain.run(document=full_text, abbr=abbr).strip()
        resolved[abbr] = answer if answer else "Not found in document"
    return resolved
