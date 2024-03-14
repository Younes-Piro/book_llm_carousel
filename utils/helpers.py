from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
import re


def read_pdf_path(book_path):
    # load cv file
    i_f = open(book_path,'rb')
    resMgr = PDFResourceManager()
    retData = io.StringIO()
    TxtConverter = TextConverter(resMgr,retData, laparams= LAParams())
    interpreter = PDFPageInterpreter(resMgr,TxtConverter)
    for page in PDFPage.get_pages(i_f):
        interpreter.process_page(page)
 
    txt = retData.getvalue()
    return txt

def skip_before_pattern(pattern, text):
    # Find the start of the desired section
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        start_position = match.start()
        desired_text = text[start_position:]
    else:
        desired_text = text

    return text