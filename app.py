import streamlit as st

st.set_page_config(page_title= "DocuChat AI", page_icon= "💬",layout= "wide")

st.title("DocuChat AI")
st.subheader("Upload a PDF and chat with its contents")

st.markdown("Upload a pdf file")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a PDF to begin")
else: 
    st.success(f"Uploaded file: {uploaded_file.name}")

st.markdown("Ask a Question")
user_question = st.chat_input("Type your question here.....")

if user_question:
    st.markdown("Your question")
    st.write(user_question)

    if uploaded_file is None:
        st.warning("Please upload a PDF before asking a question.")
    else: 
        st.info("Docuchat AI will answer for question")