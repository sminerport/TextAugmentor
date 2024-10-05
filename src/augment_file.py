import os
import glob
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
import warnings
import nltk
from nltk.tokenize import sent_tokenize
import re

# Check if we're running in Google Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

nltk.download('punkt_tab')

# Suppress the torch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

# Set the folder paths based on environment
if IN_COLAB:
    print("Running in Google Colab, mounting Google Drive...")
    drive.mount('/content/drive')
    FOLDER = '/content/drive/MyDrive/Colab Notebooks/back-translation-text-augmentation/data'
    OUTPUT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/back-translation-text-augmentation/output'
else:
    print("Running in local environment...")
    FOLDER = 'data'
    OUTPUT_FOLDER = 'output'

def get_txt_files(folder_path):
    return glob.glob(os.path.join(folder_path, '*.txt'))

def clean_augmented_text(augmented_lines):
    cleaned_lines = []
    for line in augmented_lines:
        # Remove repeated punctuation and fix spacing
        line = remove_repeated_punctuation(line)

        # Tokenize the line into sentences and capitalize the first letter only of each sentence
        sentences = sent_tokenize(line)
        cleaned_sentences = [capitalize_sentence(sentence) for sentence in sentences]

        # Join sentences preserving original newline structure
        cleaned_lines.append(' '.join(cleaned_sentences) + '\n')

    return cleaned_lines

def remove_repeated_punctuation(line):
    # Remove repetitive punctuation (e.g., "!!!" or "...") with a single punctuation mark
    return re.sub(r'([!?.,])\1+', r'\1', line)

def capitalize_sentence(sentence):
    # Capitalize the first letter of each sentence, but leave the rest as is
    sentence = sentence.strip()  # Remove leading/trailing whitespace
    if sentence:
        return sentence[0].upper() + sentence[1:]
    return sentence

def augment_text_preserving_structure(file_path, augmenter):
    with open(file_path, 'r') as f_input:
        lines = f_input.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Processing Lines"):
        if line.strip():  # Only process non-empty lines
            # Tokenize line into sentences
            sentences = sent_tokenize(line.strip())

            augmented_sentences = []
            for sentence in sentences:
                augmented_sentence = augmenter.augment(sentence)
                augmented_sentences.append(augmented_sentence)

            # Join sentences back into a line, preserving original structure
            augmented_line = ' '.join(augmented_sentences) + '\n'
        else:
            augmented_line = '\n'  # Preserve empty lines (paragraph breaks)

        augmented_lines.append(augmented_line)

    return augmented_lines

def write_augmented_file(augmented_lines, output_file_path):
    # Write lines exactly as they were structured, preserving original line breaks and spaces
    with open(output_file_path, 'w') as f_output:
        f_output.writelines(augmented_lines)

def augment_files_in_folder(folder_path=FOLDER, output_folder=OUTPUT_FOLDER):
    folder_path = os.path.abspath(folder_path)
    output_folder = os.path.abspath(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    txt_files = get_txt_files(folder_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                 to_model_name='facebook/wmt19-de-en',
                                 name='BackTranslationAug',
                                 device=device,
                                 force_reload=False, verbose=0)

    for file_path in tqdm(txt_files, desc="Augmenting Files"):
        file_name = os.path.basename(file_path)
        print(f'Processing: {file_name}')

        # Process each file, preserving sentence and line structure
        augmented_lines = augment_text_preserving_structure(file_path, aug)

        # Clean the augmented text
        cleaned_lines = clean_augmented_text(augmented_lines)

        output_file_path = os.path.join(output_folder, 'AUG_' + file_name)
        write_augmented_file(cleaned_lines, output_file_path)

        print(f'{file_name} augmentation complete... Output saved to {output_file_path}')

def main():
    augment_files_in_folder()

if __name__ == "__main__":
    main()