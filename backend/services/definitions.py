from typing import Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import traceback
import os


def get_title_for_reference_langchain(ref: dict, pdf_path: str, google_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get the reference title using LangChain-based semantic search with Gemini.
    """
    try:
        n = ref.get("number")
        if n:
            try:
                api_key = google_api_key 
                if not api_key:
                    print("Warning: No Google API key provided. Set GOOGLE_API_KEY...")
                    return None

                loader = PyMuPDFLoader(pdf_path)
                docs = loader.load()
                if not docs:
                    return None

                reference_docs = []
                found_references = False
                for doc in docs:
                    content = doc.page_content
                    if "References" in content or "Bibliography" in content or found_references:
                        reference_docs.append(doc)
                        found_references = True
                
                if not reference_docs:
                    reference_docs = docs

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                chunks = splitter.split_documents(reference_docs)
                if not chunks:
                    return None

                vectorstore_path = f"{pdf_path}.faiss"
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )

                try:
                    if os.path.exists(vectorstore_path):
                        print(f"Loading existing vector store from: {vectorstore_path}")
                        vectorstore = FAISS.load_local(
                            vectorstore_path, 
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                    else:
                        print("No existing vector store found. Creating a new one...")
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        print(f"Saving new vector store to: {vectorstore_path}")
                        vectorstore.save_local(vectorstore_path)
                except Exception as e:
                    print(f"Error loading or creating vector store: {e}. Recreating from scratch...")
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    print(f"Saving new vector store to: {vectorstore_path}")
                    vectorstore.save_local(vectorstore_path)

                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                
                template = """You are analyzing a References/Bibliography section of a research paper.
                Your task is to locate reference number [{n}] in the provided context and extract ONLY the paper's TITLE from that reference.

                Context from the paper:
                {context}

                Question: What is the paper title for reference [{n}]?

                Instructions:
                - Find reference number [{n}] in the formats [n], (n), or n.
                - Return ONLY the paper title exactly as it appears in the citation text.
                - Do NOT include authors, editors, journal or conference names, volume, pages, year, DOI, URLs, parentheses/brackets, the reference number, or any additional commentary.
                - Do not add quotes, punctuation, or explanatory text—return the plain title string.
                - If the citation contains a subtitle, include it as part of the title.
                - If you cannot confidently find the title, respond with NOT_FOUND.

                Answer:"""
                prompt = ChatPromptTemplate.from_template(template)
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0,
                    convert_system_message_to_human=True
                )
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {
                        "context": (lambda x: f"citation for reference number {x}") | retriever | format_docs,
                        "n": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(n)
                response = response.strip()
                
                if not response or "NOT_FOUND" in response.upper() or "cannot find" in response.lower():
                    return None
                
                cleanup_phrases = [
                    "The full citation text for reference",
                    f"Reference [{n}] is:",
                    f"Reference {n} is:",
                    "The citation is:",
                    "Answer:",
                ]
                for phrase in cleanup_phrases:
                    if response.startswith(phrase):
                        response = response[len(phrase):].strip()
                
                return response if response else None
            
            except Exception as e:
                print(f"Error in get_title_for_reference_langchain: {e}")
                traceback.print_exc()
                return None

    except Exception as e:
        print(f"LangChain lookup failed: {e}")
        pass
    
    return None