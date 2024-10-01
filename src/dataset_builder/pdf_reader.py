from io import BytesIO
from PIL import Image

import base64
import fitz


class PDFReader:

    def __init__(self, pdf_path):
        self.__pdf_path = pdf_path
        self.__images = None

    def get_images(self):
        if self.__images is None:
            self.__read_pdf()
        return self.__images
    
    def __read_pdf(self):
        # Initialize the list of images
        self.__images = []

        # Open the pdf file
        pdf_document = fitz.open(self.__pdf_path)

        # Set the DPI for the image rendering (96 is enough for vision to read the text and reduce token usage)
        dpi = 96

        # Iterate over all the pages in the PDF
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)  # Render the page to an image
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # Convert the image to a PIL-compatible format
            buffered = BytesIO()  # Create an in-memory bytes buffer
            image.save(buffered, format="PNG")  # Save the image to the buffer in PNG format
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Encode the buffer content to base64
            self.__images.append(img_str)  # Append the base64 string to the list
