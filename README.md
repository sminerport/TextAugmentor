# Text Augmentation using Back-Translation

This repository contains a Python script that leverages the NLPAug library and round-trip translation (RTT) technique to augment text datasets. The script processes TXT files in the "data/" folder, translating text to another language and back, creating augmented versions. The augmented dataset enhances training data for natural language processing tasks like chatbot training or text classification.

## Overview of Text Augmentation Techniques

Text augmentation is a technique used to expand or modify existing text data in a way that increases the variety and quantity of training data for natural language processing tasks. Common text augmentation techniques include:

- Synonym replacement: Replacing words with their synonyms.
- Random insertion: Inserting random synonyms of words into the text.
- Random deletion: Deleting random words from the text.
- Random swap: Swapping the position of random words in the text.
- Back-translation: Translating the text to another language and back to the original language.

## Applications of Text Augmentation

Text augmentation can be used for various natural language processing tasks, including:

- Text classification
- Sentiment analysis
- Named entity recognition
- Machine translation
- Chatbot training
- Question-answering systems

## Back-Translation Augmenter

This repository uses the back-translation augmentation technique, which involves translating the text to another language and then translating it back to the original language. This process can introduce variations in the text while preserving the meaning, which helps create more diverse training data. The Python script in this repository uses the NLPAug library to perform back-translation on the input text files.

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/sminerport/back-translation-text-augmentation.git
    ```
2. Change to the project directory:

    ```bash
    cd text-augmentation-back-translation
    ```
3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
4. Run the Python script:

    ```bash
    python src/augment_file.py
    ```

The script will process each TXT file in the `data/` directory, performing round-trip translation, and save the augmented versions with an "AUG_" prefix in the same folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLPAug library: https://github.com/makcedward/nlpaug
- Round-trip translation technique inspiration
