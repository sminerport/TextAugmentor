import sys
import os
import glob
import warnings
import re
import nltk
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch

# Suppress FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

def is_running_in_colab():
    """
    Check if the script is running in Google Colab by checking environment variables and default Colab paths.
    """
    return 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ or os.path.exists('/content')

def main():
    print(f"Current Python version: {sys.version}")

    nltk.download("punkt")

    # Set the folder paths based on environment
    if is_running_in_colab():
        print("Running in Google Colab. Please manually connect Google Drive.")
        folder_path = "/content/drive/MyDrive/Code/TextAugmentor/data"
        output_folder = "/content/drive/MyDrive/Code/TextAugmentor/output"
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

def get_max_line_length(file_path):
    max_length = 0
    with open(file_path, "r") as f_input:
        for line in f_input:
            max_length = max(max_length, len(line.strip()))
    return max_length

def augment_text_preserving_structure(file_path, augmenter):
    max_line_length = get_max_line_length(file_path)
    with open(file_path, "r") as f_input:
        text = f_input.read()

    # Split the text into paragraphs, capturing the original newlines.
    paragraphs = re.split(r'(\n{2,})', text)  # Capture paragraphs and preserve original newlines

    augmented_text = ""
    for paragraph in tqdm(paragraphs, desc="Processing Paragraphs"):
        if re.match(r'^\n+$', paragraph):
            augmented_text += paragraph  # Preserve single or multiple newlines
            continue

        # Process each paragraph by splitting into sentences and preserving original spaces.
        sentences_with_spaces = split_text_with_spaces(paragraph)

        paragraph_text = ""
        for sentence, spaces in sentences_with_spaces:
            if sentence.strip():  # Only process non-empty sentences
                augmented_sentence = augmenter.augment(sentence.replace("\n", " "))[0]
                paragraph_text += augmented_sentence + spaces.replace("\n", "")
            else:
                paragraph_text += spaces  # Preserve spaces for empty sentences

        # Format text within each paragraph to max_line_length
        formatted_paragraph = format_text(paragraph_text, max_line_length).lstrip("\n")
        augmented_text += formatted_paragraph

    return augmented_text.rstrip()  # Ensure no extra newlines at the end

def split_text_with_spaces(text):
    """
    Splits text into sentences, preserving original spacing (single or double), excluding newlines.
    """
    pattern = re.compile(r'(.*?[.!?]["\']?)([ \t]+|$)', re.DOTALL)  # Capture spaces but not newlines
    return pattern.findall(text)

def format_text(text, max_line_length):
    """
    Format text to fit within a maximum line length, preserving spaces between sentences.
    """
    words = re.split(r'(\s+)', text)  # Split text into words and preserve spaces
    formatted_text = ""
    line = ""

    for word in words:
        # If adding the next word would exceed the max_line_length, add the line to formatted_text
        if len(line) + len(word) > max_line_length:
            formatted_text += line.rstrip() + "\n"
            line = ""

        # Avoid starting a new line with a space
        if not line and word.isspace():
            continue

        line += word

    # Add the last line
    if line:
        formatted_text += line.rstrip()

    return formatted_text

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

        # Ensure the header (like "CHAPTER 2") is preserved with its original spacing
        with open(file_path, "r") as original_file:
            original_lines = original_file.readlines()
            if original_lines:
                header = original_lines[0].rstrip("\n")
                if header:
                    # Preserve the original blank line after the header if it exists
                    cleaned_text = header + ("\n\n" if len(original_lines) > 1 and original_lines[1].strip() == "" else "\n") + cleaned_text.lstrip("\n")

        output_file_path = os.path.join(output_folder, "AUG_" + file_name)
        write_augmented_file(cleaned_text, output_file_path)

        print(f"{file_name} augmentation complete... Output saved to {output_file_path}")

if __name__ == "__main__":
    main()
