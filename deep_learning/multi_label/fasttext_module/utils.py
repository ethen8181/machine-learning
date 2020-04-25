import os


__all__ = ['prepend_file_name']


def prepend_file_name(path: str, name: str) -> str:
    """
    Prepend the name to the base file name of the input path.
    e.g. data/cooking.stackexchange.txt, prepend 'train' to the base file name
    data/train_cooking.stackexchange.txt

    Parameters
    ----------
    path : str
        Path to a file.

    name : str
        Name that we'll prepend to the base file name.

    Returns
    -------
    prepended_file_name : str
    """
    directory = os.path.dirname(path)
    file_name = os.path.basename(path)
    return os.path.join(directory, name + '_' + file_name)
