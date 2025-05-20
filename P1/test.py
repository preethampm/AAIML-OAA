import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OllamaEmbeddings(model="llama3.2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    
    llm = OllamaLLM(model="llama3.2", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    return qa_chain

if __name__ == "__main__":
    qa_chain = setup_qa_system("paper5.pdf")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(query)
        print(f"Answer: {answer['result']}")
