from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQA

from config import settings


def load_all_documents(llm: OpenAI):
    load_pdf_documents(llm)
    load_txt_documents(llm)


def load_pdf_documents(llm: OpenAI):
    persist_directory = 'db/pdf_documents'
    embedding = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    if settings.RELOAD_DOCUMENTS:
        loader = PyPDFLoader('data/employee_handbook.pdf')
        pages = loader.load_and_split()

        # Embed and store the texts
        # Supplying a persist_directory will store the embeddings on disk
        vectordb = Chroma.from_documents(
            documents=pages, embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()
        vectordb = None

    # Now we can load the persisted database from disk, and use it as normal.
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    # vectordb.similarity_search("""what's innovation theme""", k=2)

    # retriever = vectordb.as_retriever()
    # retriever.get_relevant_documents("what's the innovation theme?")

    return VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=vectordb)

def load_txt_documents(llm: OpenAI):
    persist_directory = 'db/text_documents'
    embedding = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

    if settings.RELOAD_DOCUMENTS:
        loader = TextLoader('data/employee_data.txt')
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)


        vectordb = Chroma.from_documents(
            documents=texts, embedding=embedding,
            persist_directory=persist_directory)
        vectordb.persist()
        vectordb = None

    # Now we can load the persisted database from disk, and use it as normal.
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)

    retriever = vectordb.as_retriever()
    result = retriever.get_relevant_documents("what's Phi Huynh's birthday?")

    return VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=vectordb)