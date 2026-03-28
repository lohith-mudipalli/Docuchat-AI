import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title= "DocuChat AI", page_icon= "💬", layout= "wide")

st.title("DocuChat AI")
st.subheader("Upload a PDF and chat with its contents")

st.markdown("Upload a pdf file")

# defined a function for the extract the text from the pdf.
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages_data.append(
                {
                    "page": page_number,
                    "text": text,
                    "source": uploaded_file.name, 
                }
            )
        
    return pages_data



uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

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
            st.markdown("Extracted PDF Context")
            st.write(f"Total readable pages: {len(pages_data)}")

            for page_data in pages_data:
                with st.expander(f"Page {page_data['page']}"): 
                    st.write(page_data["text"])
    except Exception as error:
        st.error(f"Error while reading PDF: {error}")


st.markdown("Ask a Question")
user_question = st.chat_input("Type your question here.....")

if user_question:
    st.markdown("Your question")
    st.write(user_question)

    if uploaded_file is None:
        st.warning("Please upload a PDF before asking a question.")
    else: 
        st.info("Docuchat AI will answer for question")