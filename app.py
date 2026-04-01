import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb


load_dotenv()

st.set_page_config(page_title= "DocuChat AI", page_icon= "💬", layout= "wide")

st.title("DocuChat AI")
st.subheader("Upload a PDF and chat with its contents")

st.markdown("Upload a pdf file")

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

# defined a function for chunking - breaking a large text into smaller pieces.
def chunk_pages(pages_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators = ["\n\n", "\n", " ", ""],
    )

    chunks = []
    chunk_id = 1

    for page_data in pages_data:
        split_texts = text_splitter.split_text(page_data["text"])

        for split_text in split_texts:
            cleaned_text = split_text.strip()

            if cleaned_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_data["page"],
                        "source": page_data["source"],
                        "content": split_text.strip(),
                    }
                )
                chunk_id += 1
    return chunks

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

# defined a function for the semantic retrieval.
def retrieve_relevant_chunks(collection, user_question, top_k=3):
    results = collection.query(
        query_texts = [user_question],
        n_results = top_k,
    )

    retrieved_chunks = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for document, metadata in zip(documents, metadatas):
        retrieved_chunks.append(
            {
                "content": document,
                "page": metadata["page"],
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
            }
        )
    return retrieved_chunks


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
user_question = st.chat_input("Type your question here.....")

if user_question:
    st.markdown("Your question")
    st.write(user_question)

    if uploaded_file is None:
        st.warning("Please upload a PDF before asking a question.")
    elif collection is None: 
        st.warning("The document is not ready for retrieval yet.")
    else:
        try:
            retrieved_chunks = retrieve_relevant_chunks(collection, user_question, top_k=3)

            if not retrieved_chunks:
                st.warning("No relevant chunks were found.")
            else:
                st.markdown("Top Retrieved Chunks")

            for chunk in retrieved_chunks:
                with st.expander(
                    f"Chunk {chunk['chunk_id']} | Page {chunk['page']}"
                    ):
                        st.write(chunk["content"])
        
        except Exception as error:
            st.error(f"Error while retrieving relevant chunks: {error}")

            