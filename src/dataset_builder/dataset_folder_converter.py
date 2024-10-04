import json
import os
import time

from src.common.pdf_convertion import convert_pdf_to_json


class DatasetFolderConverter:

    def __init__(self, pdf_folder, result_folder, override):
        self.pdf_folder = pdf_folder
        self.result_folder = result_folder
        self.override = override

    def run(self):
        print(f"Start converting folder '{self.pdf_folder}'")
        index = 0
        for root, _, files in os.walk(self.pdf_folder):
            total = len(files)
            for file in files:
                index += 1
                print(f"Processing file {index}/{total}: {file}")
                if file.endswith('.pdf'):
                    self.convert_pdf(os.path.join(root, file))
                else:
                    print("CV is not in a PDF format")
        print(f"Folder '{self.pdf_folder}' has been converted")

    def convert_pdf(self, pdf_path):
        output_file = self.get_result_file(pdf_path)
        if self.override is False and os.path.exists(output_file):
            print("Already converted and will be ignored.")
            return

        try:
            json_response = convert_pdf_to_json(pdf_path)
        except Exception as e:
            print("Cannot get JSON. IGNORED. File is removed.", e)
            os.remove(pdf_path)
            return

        # Save the response to a JSON file
        with open(output_file, 'w') as f:
            json.dump(json_response, f, indent=4)
        print(f"Result saved in file '{output_file}'. Waiting 60s before next conversion.")
        time.sleep(60)

    def get_result_file(self, pdf_path):
        filename = os.path.basename(pdf_path)
        dot_position = filename.find('.')
        json_filename = filename[:dot_position] + '.json'
        json_filename = json_filename.replace(' ', '_')
        return os.path.join(self.result_folder, json_filename)
