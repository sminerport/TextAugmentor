import pandas as pd
import numpy as np
import os
import glob
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

FOLDER = 'data'


def augment_file(folder_path=FOLDER):
    """
    This function takes a folder path as an argument and augments all the text files in the folder using the BackTranslationAug algorithm from the naw library.

    Parameters
    ----------
    folder_path : str
        The path of the folder containing the text files to be augmented.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the folder path is invalid.

    Examples
    --------
    >>> augment_file(folder_path='/path/to/folder')
    """

    folder_path += '' if folder_path[-1] == '/' else '/'
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, FOLDER, "")
    # get all text files
    txt_files = glob.glob(folder_path + '*.txt')

    for f in txt_files:
        aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                     to_model_name='facebook/wmt19-de-en',
                                     name='BackTranslationAug', device='cuda',
                                     force_reload=False, verbose=0)
        with open(f) as f_input:
            line = f_input.readline()
            cnt = 1
            file_name = os.path.basename(f)
            print(f'Augmenting File: {file_name}')
            print('----------------------', end='\n\n')
            while line:
                print(f'    Line {cnt}: {line}', end='')
                augment = aug.augment(line)
                print(f'AUG Line {cnt}: {augment}', end='\n\n')
                with open(os.path.join(folder_path, 'AUG_' + file_name), mode='a') as f_output:
                    f_output.write(' '.join(augment) + '\n')
                    line = f_input.readline()
                    cnt += 1
            print(f'{file_name} augmentation complete...', end='\n\n')


def main():
    augment_file()


if __name__ == "__main__":
    main()
