import sys
import os
import subprocess
import warnings
import re
from tqdm import tqdm
import nltk
import nlpaug.augmenter.word as naw
import torch

# Function to check if running in Colab
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Function to check Python version and install Python 3.9 if necessary
def check_python_version():
    if is_running_in_colab() and (sys.version_info.major != 3 or sys.version_info.minor != 9):
        print("You are not running Python 3.9. Please install Python 3.9 and restart the runtime.")
        print("To install Python 3.9, run the following commands:")
        print("""
        !apt-get update -y
        !apt-get install python3.9 -y
        !apt-get install python3.9-distutils -y
        !wget https://bootstrap.pypa.io/get-pip.py
        !python3.9 get-pip.py
        !python3.9 -m pip install --upgrade pip
        """)
        sys.exit(1)

# Verify Python version
check_python_version()

# Continue with the rest of the code after Python 3.9 is verified
print(f"Current Python version: {sys.version}")

# Your code continues here...
nltk.download("punkt")

# Suppress the torch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Set the folder paths based on environment
if is_running_in_colab():
    print("Running in Google Colab, mounting Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive")
    FOLDER = "/content/drive/MyDrive/Colab Notebooks/back-translation-text-augmentation/data"
    OUTPUT_FOLDER = "/content/drive/MyDrive/Colab Notebooks/back-translation-text-augmentation/output"
else:
    print("Running in local environment...")
    FOLDER = "data"
    OUTPUT_FOLDER = "output"

# Additional functions for augmentation processing...
def get_txt_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.txt"))

def remove_repeated_punctuation(text):
    return re.sub(r"([!?.,])\1+", r"\1", text)

def augment_text_preserving_structure(file_path, augmenter):
    with open(file_path, "r") as f_input:
        text = f_input.read()

    # Split text into sentences with their trailing spaces
    sentences_with_spaces = split_text_with_spaces(text)

    augmented_text = ""
    for sentence, spaces in tqdm(sentences_with_spaces, desc="Processing Sentences"):
        if sentence.strip():  # Only process non-empty sentences
            augmented_sentence = augmenter.augment(sentence)[0]
            augmented_text += augmented_sentence + spaces
        else:
            augmented_text += spaces  # Preserve spaces for empty sentences

    return augmented_text

def split_text_with_spaces(text):
    pattern = re.compile(r'(.*?[\.\!\?]["\']?)(\s+|$)', re.DOTALL)
    return pattern.findall(text)

def write_augmented_file(augmented_text, output_file_path):
    with open(output_file_path, "w") as f_output:
        f_output.write(augmented_text)

def augment_files_in_folder(folder_path=FOLDER, output_folder=OUTPUT_FOLDER):
    folder_path = os.path.abspath(folder_path)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    txt_files = get_txt_files(folder_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    aug = naw.BackTranslationAug(
        from_model_name="facebook/wmt19-en-de",
        to_model_name="facebook/wmt19-de-en",
        name="BackTranslationAug",
        device=device,
        force_reload=False,
        verbose=0,
    )

    for file_path in tqdm(txt_files, desc="Augmenting Files"):
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}")

        # Process each file, preserving sentence and line structure
        augmented_text = augment_text_preserving_structure(file_path, aug)

        # Clean the augmented text
        cleaned_text = remove_repeated_punctuation(augmented_text)

        output_file_path = os.path.join(output_folder, "AUG_" + file_name)
        write_augmented_file(cleaned_text, output_file_path)

        print(f"{file_name} augmentation complete... Output saved to {output_file_path}")

def main():
    augment_files_in_folder()

if __name__ == "__main__":
    main()
