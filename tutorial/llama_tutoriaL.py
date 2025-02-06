import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("Build a RAG System with DeepSeek R1 & Ollama")

# File upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_extension = "pdf" if uploaded_file.type == "application/pdf" else "txt"
    temp_file_path = uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the file based on its type
    if uploaded_file.type == "application/pdf":
        loader = PDFPlumberLoader(temp_file_path)
    else:
        loader = TextLoader(temp_file_path, encoding='utf-8')
    
    docs = loader.load()

    # Split into chunks
    text_splitter = SpacyTextSplitter(
        chunk_size=300, 
        pipeline="ko_core_news_sm"
    )
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    # Create the vector store
    database = Chroma(
        persist_directory="C:\jupyter\RecipeAI\database",
        embedding_function=embedder
    )

    database.add_documents(documents)
    
    retriever = database.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define llm
    llm = Ollama(model="benedict/linkbricks-llama3.1-korean:8b")

    # Define the prompt
    prompt = """
    1. 주어진 컨텍스트를 사용하여 마지막 질문에 답변하세요.
    2. 답을 모르는 경우 답을 만들어내지 말고 "모르겠습니다"라고 답변하세요.
    3. 반드시 한국어로 답변하세요.
    4. 답변하고 나서 그렇게 말한 근거를 말해줘
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None)

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True)

    # User input
    user_input = st.text_input("Ask a question related to the uploaded file:")

    # Process user input
    if user_input:
        with st.spinner("Processing..."):
            response = qa(user_input)["result"]
            st.write("Response:")
            st.write(response)
else:
    st.write("Please upload a PDF or TXT file to proceed.")