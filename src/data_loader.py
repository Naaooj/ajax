from pdf_conversion import convert_folder
import sys
import os

def convert_folders(override):
    hired_folder_path = os.path.join(os.getcwd(), 'resumes/hired/')
    rejected_folder_path = os.path.join(os.getcwd(), 'resumes/rejected/')
    hired_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/hired/')
    rejected_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/rejected/')

    convert_folder(hired_folder_path, hired_results_folder_path, override)
    convert_folder(rejected_folder_path, rejected_results_folder_path, override)

def get_override_param():
    user_input = input("Would you like to override existing files? (y/N): ").lower()
    if user_input == "y":
        return True
    elif user_input == "n" or user_input == "N" or user_input == "":
        return False
    else:
        print("Invalid input. Please try again.")
        sys.exit()

if __name__ == '__main__':
    convert_folders(get_override_param())
