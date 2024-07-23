import os
import openai
import chainlit as cl
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

#Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

#Load PDF and split into chunks
loader = PyMuPDFLoader (
    "https://www.hillrom.com/content/dam/hillrom-aem/us/en/sap-documents/LIT/80026/80026025LITPDF.pdf"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200
)

documents = text_splitter.split_documents(documents)

#Load embeddings model - we'll use OpenAI's text-embedding-3-small
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

#Create QDrant vector store
qdrant_vector_store = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="WelchAllynVSMServiceManual",
)

#Create Retriever
retriever = qdrant_vector_store.as_retriever()

#Create Prompt Template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

#Choose LLM - we'll use gpt-4o.
primary_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

#Set up Chainlit
@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'HTMLLMBot'.
    """
    rename_dict = {
        "Assistant" : "HTMLLMBot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """
    retrieval_augmented_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | prompt | primary_llm
    )

    cl.user_session.set("retrieval_augmented_chain", retrieval_augmented_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    retrieval_augmented_chain = cl.user_session.get("retrieval_augmented_chain")

    msg = cl.Message(content="")

    async for chunk in retrieval_augmented_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk.content)

    await msg.send()