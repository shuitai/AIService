from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

llm_model = "mistral:7b"


def load_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    # 将文档分割成小块
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts


# Step 2: 创建向量存储
def create_vector_store(texts):
    """
    使用 HuggingFace Embeddings 和 Chroma 创建向量存储。
    """
    embeddings = OllamaEmbeddings(model=llm_model)
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")  # 持久化存储
    return vector_store


# Step 3: 初始化 Ollama LLM
def init_ollama_llm():
    """
    初始化 Ollama 作为 LLM。
    """
    llm = OllamaLLM(base_url="http://localhost:11434", model=llm_model)  # 使用 LLaMA 2 模型
    return llm


# Step 4: 初始化 RAG 链
def init_rag_chain(vector_store, llm):
    """
    初始化 RAG 链，结合检索和生成。
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # 检索前 3 个相关文档
        return_source_documents=False  # 返回检索到的文档
    )

    return qa_chain


# Step 5: 运行 RAG
def run_rag(qa_chain, query):
    """
    运行 RAG 并返回结果。
    """
    result = qa_chain.invoke({"query": query})
    return result


def run_rag_stream(qa_chain, query):
    """
    使用 stream 方法逐步返回生成结果。
    """
    for chunk in qa_chain.stream({"query": query}):
        print(chunk)


if __name__ == "__main__":
    # 加载文档
    file_path = "knowledge_base2.txt"
    texts = load_documents(file_path)

    # 创建向量存储
    vector_store = create_vector_store(texts)

    # 初始化 Ollama LLM
    llm = init_ollama_llm()

    # 初始化 RAG 链
    qa_chain = init_rag_chain(vector_store, llm)

    # 运行 RAG
    query = "What is the capital of China?"
    run_rag_stream(qa_chain, query)
    # result = run_rag(qa_chain, query)
    #
    # # 打印结果
    # print("Answer:", result["result"])
    # print("Source Documents:")
    # for doc in result["source_documents"]:
    #     print(doc.page_content)
