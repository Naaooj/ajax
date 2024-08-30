from pdf_conversion import convert_folder
import os


def main():
    hired_folder_path = os.path.join(os.getcwd(), 'resumes/hired/')
    rejected_folder_path = os.path.join(os.getcwd(), 'resumes/rejected/')
    hired_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/hired/')
    rejected_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/rejected/')

    convert_folder(hired_folder_path, hired_results_folder_path)
    # convert_folder(rejected_folder_path, rejected_results_folder_path)


if __name__ == '__main__':
    main()
