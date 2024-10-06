import sys
import os
import glob
import warnings
import re
import subprocess
import nltk
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch


def is_running_in_colab():
    """
    Check if the script is running in Google Colab.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_python_39_in_colab():
    """
    Install Python 3.9 and run setup.sh in Colab.
    """
    print("Installing Python 3.9 and necessary packages...")

    # Install Python 3.9 using subprocess
    subprocess.run(['apt-get', 'update', '-y'], check=True)
    subprocess.run(['apt-get', 'install', 'python3.9', '-y'], check=True)
    subprocess.run(['apt-get', 'install', 'python3.9-distutils', '-y'], check=True)

    # Check if get-pip.py already exists
    if not os.path.exists('get-pip.py'):
        print("Downloading get-pip.py...")
        subprocess.run(['wget', 'https://bootstrap.pypa.io/get-pip.py'], check=True)
    else:
        print("get-pip.py already exists. Skipping download.")

    subprocess.run(['python3.9', 'get-pip.py'], check=True)

    # Switch to using Python 3.9 explicitly
    subprocess.run(['update-alternatives', '--install', '/usr/bin/python3', 'python3', '/usr/bin/python3.9', '2'], check=True)
    subprocess.run(['update-alternatives', '--set', 'python3', '/usr/bin/python3.9'], check=True)

    # Run the setup.sh script instead of pip install requirements.txt directly
    subprocess.run(['bash', 'setup.sh'], check=True)

    print("Installation complete. Please restart the runtime and run the script again.")
    sys.exit()

def main():
    # Check if running in Colab and install Python 3.9 if necessary
    if is_running_in_colab() and (sys.version_info.major != 3 or sys.version_info.minor != 9):
        install_python_39_in_colab()

    print(f"Current Python version: {sys.version}")

    nltk.download("punkt")

    # Set the folder paths based on environment
    if is_running_in_colab():
        print("Running in Google Colab, mounting Google Drive...")
        from google.colab import drive
        drive.mount("/content/drive")
        folder_path = "/content/drive/MyDrive/Colab Notebooks/TextAugmentor/data"
        output_folder = "/content/drive/MyDrive/Colab Notebooks/TextAugmentor/output"
    else:
        print("Running in local environment...")
        folder_path = "data"
        output_folder = "output"

    # Continue with the rest of the script after the installation
    augment_files_in_folder(folder_path, output_folder)

def get_txt_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.txt"))

def remove_repeated_punctuation(text):
    return re.sub(r"([!?.,])\1+", r"\1", text)

def augment_text_preserving_structure(file_path, augmenter):
    with open(file_path, "r") as f_input:
        lines = f_input.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Processing Lines"):
        if line.strip():  # Non-empty line
            # Split the line into sentences, capturing the spaces
            sentences_with_spaces = split_line_with_spaces(line)

            augmented_line = ''
            for sentence, spaces in sentences_with_spaces:
                if sentence.strip():
                    augmented_sentence = augmenter.augment(sentence)[0]
                    augmented_line += augmented_sentence + spaces
                else:
                    augmented_line += spaces  # Preserve spaces for empty sentences

            augmented_lines.append(augmented_line)
        else:
            # Empty line (paragraph break), preserve as is
            augmented_lines.append(line)

    return ''.join(augmented_lines)

def split_line_with_spaces(line):
    """
    Splits a line into sentences, preserving original spacing (including double spaces and line breaks).
    """
    pattern = re.compile(r'(.*?[\.\!\?]["\']?|\S.+?)(\s+|$)', re.DOTALL)
    sentences_with_spaces = pattern.findall(line)
    return sentences_with_spaces

def write_augmented_file(augmented_text, output_file_path):
    with open(output_file_path, "w") as f_output:
        f_output.write(augmented_text)

def augment_files_in_folder(folder_path, output_folder):
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

if __name__ == "__main__":
    main()