import json
import os

from azure_service import AzureService
from pdf_reader import PDFReader


def convert_pdf(pdf_path, result_path):
    print(f"Start converting file {pdf_path}")

    # Create the PDFReader object, and get the images
    pdf_reader = PDFReader(pdf_path)
    images = pdf_reader.get_images()

    # Create the AzureService object and call the Vision API
    azure_service = AzureService()
    response = azure_service.call_vision_api(images)

    # Get the JSON response from the API (as requested in the prompt)
    json_response = response.choices[0].message.content

    # Save the response to a JSON file
    output_file = get_result_file(pdf_path, result_path)
    write_json(json_response, output_file)
    print(f"Saved result in file {output_file}")


def write_json(json_response, output_file):
    with open(output_file, 'w') as f:
        json.dump(json_response, f, indent=4)


def get_result_file(pdf_path, result_path):
    filename = os.path.basename(pdf_path)
    dot_position = filename.find('.')
    json_filename = filename[:dot_position] + '.json'
    json_filename = json_filename.replace(' ', '_')
    return os.path.join(result_path, json_filename)


def convert_folder(pdf_folder, result_folder):
    print(f"Start converting folder {pdf_folder}")
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                convert_pdf(pdf_path, result_folder)
            else:
                print(f"File {pdf_folder} is not a PDF")
    print(f"Folder {pdf_folder} has been converted")

