from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.docstore.document import Document
from utils import helpers
import re

book = './data/Cutting-edge+digital+marketing.pdf'

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
docs = text_splitter.split_documents(book_docs)

print(docs)