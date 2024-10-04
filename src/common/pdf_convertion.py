import json

from src.common.azure_service import AzureService
from src.dataset_builder.pdf_reader import PDFReader


def convert_pdf_to_json(pdf_path):

    # Create the PDFReader object, and get the images
    pdf_reader = PDFReader(pdf_path)
    images = pdf_reader.get_images()
    if len(images) > 10:
        print("CV has more than 10 pages. Only the first 10 pages will be converted.")
        images = images[:10]

    # Get the JSON response from the API (as requested in the prompt)
    azure_service = AzureService()
    response = azure_service.call_vision_api(images)
    json_response = response.choices[0].message.content
    return json.loads(json_response)
