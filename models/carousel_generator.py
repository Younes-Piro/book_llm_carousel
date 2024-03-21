from langchain_community.vectorstores import Chroma
import chromadb
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
import numpy as np
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

_ = load_dotenv()

def generate_carousel(query):

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

    results = vectordb.max_marginal_relevance_search(query,k=5)

    ## rerank retreivers
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    retrieved_documents = [result.page_content for result in results]

    pairs = [[query, doc] for doc in retrieved_documents]

    scores = cross_encoder.predict(pairs)

    orders = []
    for o in np.argsort(scores)[::-1]:
        orders.append(o)

    # reorder my retreive docs following the reranking new order
    sorted_retrieved_documents = [retrieved_documents[i] for i in orders]

    ## generate carousel

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    template = """
    You are a specialized tool that build instragram carousel with only 5 slide. 
    Your goal is to start with a catchy scroll-sropping hook with title and then write slide by slide. 
    You should explain each slide with the exact content i should use, not instructions, so i can copy and paste each slide
    context: {context} 
    Answer:

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context"],
    )

    # load chain
    qa_chain = LLMChain(llm=llm, prompt=prompt)

    # list the informations
    result = qa_chain.invoke({'context': sorted_retrieved_documents[0:3]})

    return result['text']