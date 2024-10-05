import os
import glob
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import torch
import warnings
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Suppress the torch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

FOLDER = 'data'
OUTPUT_FOLDER = 'output'

def get_txt_files(folder_path):
    return glob.glob(os.path.join(folder_path, '*.txt'))

def clean_augmented_text(augmented_lines):
    cleaned_lines = []
    for line in augmented_lines:
        # Remove duplicate repetitions of phrases (e.g., "out of the crisis, out of the crisis")
        line = remove_repeated_phrases(line)

        # Capitalize the first letter of each sentence
        sentences = sent_tokenize(line)
        cleaned_sentences = [capitalize_sentence(sentence) for sentence in sentences]

        # Join back the cleaned sentences into a single line
        cleaned_lines.append(' '.join(cleaned_sentences))

    return cleaned_lines

def remove_repeated_phrases(line):
    words = line.split()
    new_words = []
    for i in range(len(words)):
        # If the current word is the same as the previous word, skip it
        if i > 0 and words[i] == words[i - 1]:
            continue
        new_words.append(words[i])
    return ' '.join(new_words)

def capitalize_sentence(sentence):
    if sentence:
        # Capitalize the first letter if it isn't already
        return sentence[0].upper() + sentence[1:]
    return sentence

def augment_text_by_sentence_with_line_breaks(file_path, augmenter):
    with open(file_path, 'r') as f_input:
        lines = f_input.readlines()

    augmented_lines = []
    for line in tqdm(lines, desc="Processing Lines"):
        # Tokenize each line into sentences
        sentences = sent_tokenize(line.strip())

        # Augment each sentence
        augmented_sentences = []
        if sentences:
            for sentence in sentences:
                augmented_sentence = augmenter.augment([sentence])[0]
                augmented_sentences.append(augmented_sentence)
            # Reconstruct the line from augmented sentences
            augmented_line = ' '.join(augmented_sentences)
        else:
            augmented_line = ''  # Handle empty lines

        augmented_lines.append(augmented_line + '\n')  # Maintain original newline

    return augmented_lines

def write_augmented_file(augmented_lines, output_file_path):
    # Write lines exactly as they were structured (preserving original line breaks)
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

        augmented_lines = augment_text_by_sentence_with_line_breaks(file_path, aug)

        # Clean augmented text for better formatting
        cleaned_lines = clean_augmented_text(augmented_lines)

        output_file_path = os.path.join(output_folder, 'AUG_' + file_name)
        write_augmented_file(cleaned_lines, output_file_path)

        print(f'{file_name} augmentation complete... Output saved to {output_file_path}')

def main():
    augment_files_in_folder()

if __name__ == "__main__":
    main()
