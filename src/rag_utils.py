from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# defined a function for chunking - breaking a large text into smaller pieces.
def chunk_pages(pages_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
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
                        "content": cleaned_text,
                    }
                )
                chunk_id += 1

    return chunks

# defined a function for the semantic retrieval.
def retrieve_relevant_chunks(collection, user_question, top_k=3):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embedding_model.embed_query(user_question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved_chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for document, metadata in zip(documents, metadatas):
        retrieved_chunks.append(
            {
                "content": document,
                "page": metadata.get("page"),
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
            }
        )

    return retrieved_chunks

#defined a function for avoids showing duplicate citations again and again.
def get_unique_sources(retrieved_chunks):
    unique_sources = []
    seen = set()

    for chunk in retrieved_chunks:
        label = f"{chunk['source']} - Page {chunk['page']}"
        if label not in seen:
            seen.add(label)
            unique_sources.append(label)

    return unique_sources