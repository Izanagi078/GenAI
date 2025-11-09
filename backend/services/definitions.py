from typing import Optional
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
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
                        print("references section found in the paper")
                        reference_docs.append(doc)
                        found_references = True
                    elif found_references:
                        # Continue adding pages until we hit a new section
                        if not any(section in content.lower() for section in ["appendix", "acknowledg", "credit authorship"]):
                            reference_docs.append(doc)
                        else:
                            break
                
                if not reference_docs:
                    reference_docs = docs[-3:]

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
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
                            allow_dangerous_deserialization=True         #make it false if in prod (for user uploads) and use alternative deserialization
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

                query_str = f"Full bibliographic reference entry for citation [{n}], including [{n}] authors, title, journal name, year, and publication details."

                retrieved_docs = retriever.invoke(query_str)

                print("\n--- Retrieved Chunks ---")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\nChunk {i+1}:\n{doc.page_content}\n")

                formatted_context = format_docs(retrieved_docs)

                inputs = {
                    "context": formatted_context,
                    "n": str(n)
                }
                response = (prompt | llm | StrOutputParser()).invoke(inputs)
                response = response.strip()
                
                if not response or "NOT_FOUND" in response.upper() or "cannot find" in response.lower():
                    return "NOT_FOUND"
                
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
                
                return response if response else "NOT_FOUND"
            
            except Exception as e:
                print(f"Error in get_title_for_reference_langchain: {e}")
                traceback.print_exc()
                return None

    except Exception as e:
        print(f"LangChain lookup failed: {e}")
        pass
    
    return None

def find_answer(pdf_path: str, query: str, google_api_key: Optional[str] = None) -> str:
    """
    Finds an answer to a query within a PDF document using a RAG pipeline.
    """
    try:
        api_key = google_api_key
        if not api_key:
            return "Error: Google API key not provided."

        # Load the PDF
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            return "Could not load the document."

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return "Could not split the document into chunks."

        # Create or load the vector store
        vectorstore_path = f"{pdf_path}.faiss"
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
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
        except Exception as e:
            # If loading fails, recreate from scratch
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(vectorstore_path)


        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Define the prompt template
        template = """
        Answer the following question based only on the provided context.
        If the context does not contain the answer, respond with "This PDF does not have the answer to this question."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the language model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Create the RAG chain
        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain and get the response
        response = rag_chain.invoke(query)
        return response.strip()

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}"


def extract_title_year_from_reference(reference_text: str, google_api_key: Optional[str] = None) -> Optional[dict]:
    """
    Extract title and year from a reference citation text using LLM.
    """
    try:
        api_key = google_api_key
        if not api_key:
            print("Warning: No Google API key provided.")
            return None

        template = """Extract the title and publication year from this reference citation.

Reference:
{reference_text}

Instructions:
- Return ONLY the title and year in this exact format:
Title: <exact title>
Year: <year>

- If title or year cannot be found, use "NOT_FOUND" for that field.
- Do not include any other text or explanations."""

        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905",
            temperature=0,
        )

        response = (prompt | llm | StrOutputParser()).invoke({"reference_text": reference_text})
        response = response.strip()

        title = None
        year = None

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Title:'):
                title = line[6:].strip()
            elif line.startswith('Year:'):
                year = line[5:].strip()

        if title and title != "NOT_FOUND":
            title = title
        else:
            title = None

        if year and year != "NOT_FOUND":
            year = year
        else:
            year = None

        if title or year:
            return {"title": title, "year": year}
        else:
            return None

    except Exception as e:
        print(f"Error extracting title/year from reference: {e}")
        return None


def find_full_form(abbr: str,pdf_path: str, google_api_key: Optional[str] = None) -> dict:
    """
    Finds full form of any abbreviation within a PDF document.
    """
    try:
        api_key = google_api_key
        if not api_key:
            return {"ans": "Error: Google API key not provided.", "using_llm": False}

        # Load the PDF
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            return {"ans": "Could not load the document.", "using_llm": False}

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return {"ans": "Could not split the document into chunks.", "using_llm": False}

        # Create or load the vector store
        vectorstore_path = f"{pdf_path}.faiss"
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  #cpu or cuda (for gpu)
        )
        
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
        except Exception as e:
            # If loading fails, recreate from scratch
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(vectorstore_path)


        # Create the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize the language model
        llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905",
            temperature=0,
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        query_str = f"What is the full form or definition of {abbr}?"
        retrieved_docs = retriever.invoke(query_str)
        formatted_context = format_docs(retrieved_docs)

        # First prompt: strict extraction
        first_template = """
        Your task is to find the full form for the abbreviation provided, using ONLY the context below.
        Extract and return ONLY the full, unabbreviated text.
        If the context does not contain the full form, respond with "NOT_FOUND"

        Context:
        {context}

        Abbreviation:
        {abbreviation}

        Full Form:
        """
        first_prompt = ChatPromptTemplate.from_template(first_template)
        first_response = (first_prompt | llm | StrOutputParser()).invoke({"context": formatted_context, "abbreviation": abbr})

        if first_response.strip() != "NOT_FOUND":
            return {"ans": first_response.strip(), "using_llm": False}
        else:
            print("couldn't find full form inside paper, using search agent to find")
            # Second prompt: infer suitable full form
            second_template = """
            Based on the provided context, provide a suitable full form or expansion for the abbreviation "{abbreviation}".

            If you can infer a reasonable full form from the context, it is not necessary that the full form will be in the context but you can use context to understand the topic/field/area in which you have to think to give the full form, provide it. Otherwise, respond with "NOT_FOUND".

            Context:
            {context}

            Full Form:
            """
            second_prompt = ChatPromptTemplate.from_template(second_template)
            second_response = (second_prompt | llm | StrOutputParser()).invoke({"context": formatted_context, "abbreviation": abbr})
            return {"ans": second_response.strip(), "using_llm": True}

    except Exception as e:
        traceback.print_exc()
        return {"ans": f"An error occurred: {e}", "using_llm": False}
