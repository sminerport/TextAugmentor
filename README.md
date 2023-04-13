# Text Augmentation using Back-Translation

This repository contains a Python script that leverages the NLPAug library and round-trip translation (RTT) technique to augment text datasets. The script processes TXT files in the "data/" folder, translating text to another language and back, creating augmented versions. The augmented dataset enhances training data for natural language processing tasks like chatbot training or text classification. The repository includes a brief overview of text augmentation techniques, applications, and Python code for implementing the back-translation augmenter.

## Usage

1. Clone this repository:
    ```
    git clone https://github.com/sminerport/back-translation-text-augmentation.git
    ```
2. Change to the project directory:
    ```
    cd text-augmentation-back-translation
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the Python script:
    ```
    python src/augment_file.py
    ```

The script will process each TXT file in the `data/` directory, performing round-trip translation, and save the augmented versions with an "AUG_" prefix in the same folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLPAug library: https://github.com/makcedward/nlpaug
- Round-trip translation technique inspiration

