from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from utils import helpers
from chromadb.utils import embedding_functions
import chromadb
import uuid

book = './data/Cutting-edge+digital+marketing.pdf'
path = './chromadb'
name_collection = 'book_emb'

# read the book
book_text = helpers.read_pdf_path(book)

# Define the pattern to find 
pattern = r"Alexei Vitchenko"

# take paragraphs after pattern
books_imp = helpers.skip_before_pattern(pattern, book_text)

book_docs = [
    Document(
        page_content=books_imp.strip(),
        metadata={"page": 0},
    )
]
## split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
chunks = text_splitter.split_documents(book_docs)
docs = [x.page_content for x in chunks]

## embed and store into chroma db
ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-base") 

## initialise chroma db client
client = chromadb.PersistentClient(path=path)
collection = client.get_or_create_collection(
            name=name_collection,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

print("## START EMBEDDING FUNCT")
collection.add(
            documents=docs,
            ids=[str(uuid.uuid4()) for _ in range(len(docs))]
        )
        
print("EMB DONE")