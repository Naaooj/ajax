import os

from src.dataset_builder.dataset_folder_converter import DatasetFolderConverter

class DatasetBuilder:

    def __init__(self, override):
        self.override = override

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hired_folder_path = os.path.join(current_dir, '../resumes/hired/')
        rejected_folder_path = os.path.join(current_dir, '../resumes/rejected/')
        hired_results_folder_path = os.path.join(current_dir, '../resumes/results/hired/')
        rejected_results_folder_path = os.path.join(current_dir, '../resumes/results/rejected/')

        hired_pdf_converter = DatasetFolderConverter(hired_folder_path, hired_results_folder_path, self.override)
        rejected_pdf_converter = DatasetFolderConverter(rejected_folder_path, rejected_results_folder_path, self.override)

        hired_pdf_converter.run()
        rejected_pdf_converter.run()
