from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

_ = load_dotenv()

#variables
path = './chromadb'
name_collection = 'book_emb'

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
persistent_client = chromadb.PersistentClient(path=path)

vectordb = Chroma(
    client=persistent_client,
    collection_name=name_collection,
    embedding_function=instructor_embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

query= 'Who is our client?'

llm = ChatGoogleGenerativeAI(model="gemini-pro" ,temperature=0.8, convert_system_message_to_human=True)

memory = ConversationSummaryBufferMemory(llm=llm, memory_key='chat_history', input_key='question', output_key='answer')

qa = ConversationalRetrievalChain.from_llm(llm,
                                            vectordb.as_retriever(),
                                            memory=memory,
                                            return_source_documents=True)

result = qa.invoke(query)

print(result)