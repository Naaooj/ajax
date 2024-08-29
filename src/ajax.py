from azure_service import AzureService
from pdf_reader import PDFReader

import json
import os

def main():
    # Get the reference to the pdf file
    pdf_path = os.path.join(os.getcwd(), 'resumes/resume.pdf')

    # Create the PDFReader object, and get the images
    pdf_reader = PDFReader(pdf_path)
    images = pdf_reader.get_images()

    # Create the AzureService object and call the Vision API
    azure_service = AzureService()
    response = azure_service.call_vision_api(images)

    # Get the JSON response from the API (as requested in the prompt)
    json_response = response.choices[0].message.content
    print(json_response)

    # Save the response to a JSON file
    with open('output.json', 'w') as f:
        json.dump(json_response, f, indent=4)

if __name__ == '__main__':
    main()