import os

from .utils import convert_all_tif_to_jpg


def verify_dataset_exists():
    data_path = os.path.join(os.curdir, "data")
    readme_path = os.path.join(data_path, "README.md")

    ground_truth_path = os.path.join(data_path, "ground-truth")
    image_path = os.path.join(data_path, "images")
    ocr_path = os.path.join(data_path, "ocr-text")

    for file_path in [ground_truth_path, image_path, ocr_path]:
        if not os.path.exists(file_path):
            error_string = f"'{file_path}' does not exist or is not named correctly. Follow instructions in '{readme_path}' to obtain dataset online, or if it is already downloaded ensure that it is named correctly and lives in the correct directory."
            raise FileNotFoundError(error_string)


def main():
    try:
        verify_dataset_exists()
        convert_all_tif_to_jpg()
        print("Execution successful")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
