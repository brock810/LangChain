import torch
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    cohere_api_key = "twlCjiIZjgFQBaNmICS9MOSEChnE5xwjHUzuKxxF"
    llm = ChatCohere(cohere_api_key=cohere_api_key)

    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()

    model_name = "distilbert-base-uncased"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    text_embeddings = []
    
    for doc in documents:
        if hasattr(doc, 'page_content'):
            page_content = doc.page_content
        else:
            page_content = ""

        encoded_inputs = tokenizer(page_content, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        text_embeddings.append((page_content, embeddings))
        
        print(f"Page Content: {page_content[:100]}...") 
        print(f"Embedding Shape: {embeddings.shape}")    

    print("Text Embeddings:")
    for idx, (content, embed) in enumerate(text_embeddings[:3]):
        print(f"Document {idx + 1} - Content Length: {len(content)}, Embedding Shape: {embed.shape}")

    texts = [page_content for page_content, _ in text_embeddings]
    embeddings = [embeddings for _, embeddings in text_embeddings]

    vectorstore = FAISS.from_embeddings(texts, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    input_text = "how can langsmith help with testing?"

    response = retrieval_chain.invoke({"input": input_text})

    if "answer" in response:
        print(response["answer"])
    else:
        print("No answer found.")

if __name__ == "__main__":
    main()
