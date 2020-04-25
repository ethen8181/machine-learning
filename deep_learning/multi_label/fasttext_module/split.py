import random
from typing import Tuple


__all__ = ['train_test_split_file']


def train_test_split_file(input_path: str,
                          output_path_train: str,
                          output_path_test: str,
                          test_size: float=0.1,
                          random_state: int=1234,
                          encoding: str='utf-8') -> Tuple[int, int]:
    """
    Perform train and test split on a text file without reading the
    whole file into memory.

    Parameters
    ----------
    input_path : str
        Path to the original full text file.

    output_path_train : str
        Path of the train split.

    output_path_test : str
        Path of the test split.

    test_size : float, 0.0 ~ 1.0, default 0.1
        Size of the test split.

    random_state : int, default 1234
        Seed for the random split.

    encoding : str, default 'utf-8'
        Encoding for reading and writing the file.

    Returns
    -------
    count_train, count_test : int
        Number of record in the training and test set.
    """
    random.seed(random_state)

    # accumulate the number of records in the training and test set
    count_train = 0
    count_test = 0
    train_range = 1 - test_size

    with open(input_path, encoding=encoding) as f_in, \
         open(output_path_train, 'w', encoding=encoding) as f_train, \
         open(output_path_test, 'w', encoding=encoding) as f_test:

        for line in f_in:
            random_num = random.random()
            if random_num < train_range:
                f_train.write(line)
                count_train += 1
            else:
                f_test.write(line)
                count_test += 1

    return count_train, count_test
