import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chromadb

from src.rag_utils import chunk_pages, retrieve_relevant_chunks, get_unique_sources

load_dotenv()

st.set_page_config(page_title= "DocuChat AI", page_icon= "💬", layout= "wide")

st.title("DocuChat AI")
st.subheader("Upload a PDF and chat with its contents")

st.markdown("Upload a pdf file")

if "messages" not in st.session_state:
    st.session_state.messages = []

# defined a function for the extract the text from the pdf.
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages_data = []
    #for loop version - page fata    
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages_data.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                    "source": uploaded_file.name, 
                }
            )
        
    return pages_data


# defined a function for embeddings. 
def build_vector_store(chunks):
    db_path = "chroma_db"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="dochuchat_ai")

    embedding_model = OpenAIEmbeddings()

    documents = [chunk["content"] for chunk in chunks]
    ids = [f"chunk-{chunk['chunk_id']}" for chunk in chunks]
    metadatas = [
        {
            "page": chunk["page"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(documents)

    collection.add(
        ids = ids,
        documents = documents,
        metadatas = metadatas,
        embeddings= embeddings,
    )

    return collection, len(documents)


# defined a function for final answer generation with citations.
def generate_answer(user_question, retrieved_chunks):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    context_text = "\n\n".join(
        [
            f"Source: {chunk['source']}, Page: {chunk['page']}\n{chunk['content']}"
            for chunk in retrieved_chunks
        ]
    )

    prompt = f"""
    You are an AI document assistant.

    Rules:
    1. Answer ONLY using the provided context.
    2. Do NOT use outside knowledge.
    3. If the answer is not in the context, say exactly:
        "I couldn't find that in the uploaded document."
    4. Keep the answer clear and concise.
    5. End with citations using source name and page number.


    User Question: 
    {user_question}

    Context:
    {context_text}
    """

    response = llm.invoke(prompt)
    return response.content

# defined a function for to show the previous chat messages on screen.
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPEN_API_KEY is missing. Please add it to your .env file before using DocuChat AI.")
    st.stop()


uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

collection = None
chunks = []


if uploaded_file is None:
    st.info("Please upload a PDF to begin")
else: 
    st.success(f"Uploaded file: {uploaded_file.name}")
    # Error Handling if pdf file, has any issues.
    try:
        pages_data = extract_text_from_pdf(uploaded_file)

        if not pages_data:
            st.warning("No readable text was found in this PDF.")
        else:
            chunks = chunk_pages(pages_data)
            st.markdown("Extracted PDF Context")    
            st.write(f"Total readable pages: {len(pages_data)}")
            st.write(f"Total chunks created: {len(chunks)}")

            with st.spinner("Creating embeddings and storing them in ChromaDB...."):
                collection, total_stored = build_vector_store(chunks)

            st.success("Embeddings created and stored successfully.")
            st.write(f"Total vector stored: {total_stored}")
    
    except Exception as error:
        st.error(f"Error while reading PDF: {error}")


st.markdown("Ask a Question")

display_chat_history()

user_question = st.chat_input("Type your question here.....")

if user_question:
    st.session_state.message.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.write(user_question)

    if uploaded_file is None:
        warning_message = "Please upload a PDF brfore asking a question."
        st.warning(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})

    elif collection is None: 
        warning_message = "The document is not ready."
        st.warning(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})

    else:
        try:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved_chunks = retrieve_relevant_chunks(collection, user_question, top_k=3)
            
            if not retrieved_chunks:
                no_result_message = "I couldn't find relevant information in the uploaded document."
                st.warning(no_result_message)
                st.session_state.messages.append({"role": "assistant", "content": no_result_message})

            else:
                with st.spinner("Generating grounded answer...."):
                    final_answer = generate_answer(user_question, retrieved_chunks)
                    
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                with st.chat_message("assistant"):
                    st.markdown(" Answer")
                    st.write(final_answer)
                    
                    st.markdown("Sources")
                    for source in get_unique_sources(retrieved_chunks):
                        st.write(f"- {source}")
                        
                st.markdown("Retrieved Chunks")
                for chunk in retrieved_chunks:
                    with st.expander(f"Chunk {chunk['chunk_id']} | Page {chunk['page']}"):
                        st.write(chunk["content"])
        
        except Exception as error:
            error_message = f"Error while answering the question: {error}"
            st.error(error_message)
            st.session_state.messgaes.append({"role": "assistant", "content": error_message})

            