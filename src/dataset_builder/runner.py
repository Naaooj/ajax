import sys

from src.dataset_builder.dataset_builder import DatasetBuilder

def get_override_param():
    user_input = input("Would you like to override existing files? (y/N): ").lower()
    if user_input == "y":
        return True
    elif user_input in ["n", "N", ""]:
        return False
    else:
        print("Invalid input. Please try again.")
        sys.exit()

if __name__ == '__main__':
    override = get_override_param()
    DatasetBuilder(override).run()
