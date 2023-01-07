import os
import tarfile
import zipfile
import requests
import subprocess
from tqdm import tqdm
from urllib.parse import urlparse


def download_file(url: str, directory: str):
    """
    Download the file at ``url`` to ``directory``.
    Extract to the file content ``directory`` if the original file
    is a tar, tar.gz or zip file.

    Parameters
    ----------
    url : str
        url of the file.

    directory : str
        Directory to download the file.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    content_len = response.headers.get('Content-Length')
    total = int(content_len) if content_len is not None else 0

    os.makedirs(directory, exist_ok=True)
    file_name = get_file_name_from_url(url)
    file_path = os.path.join(directory, file_name)

    with tqdm(unit='B', total=total) as pbar, open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                pbar.update(len(chunk))
                f.write(chunk)

    extract_compressed_file(file_path, directory)


def extract_compressed_file(compressed_file_path: str, directory: str):
    """
    Extract a compressed file to ``directory``. Supports zip, tar.gz, tgz,
    tar extensions.

    Parameters
    ----------
    compressed_file_path : str

    directory : str
        File will to extracted to this directory.
    """
    basename = os.path.basename(compressed_file_path)
    if 'zip' in basename:
        with zipfile.ZipFile(compressed_file_path, "r") as zip_f:
            zip_f.extractall(directory)
    elif 'tar.gz' in basename or 'tgz' in basename:
        with tarfile.open(compressed_file_path) as f:
            f.extractall(directory)


def get_file_name_from_url(url: str) -> str:
    """
    Return the file_name from a URL

    Parameters
    ----------
    url : str
        URL to extract file_name from

    Returns
    -------
    file_name : str
    """
    parse = urlparse(url)
    return os.path.basename(parse.path)



def create_translation_data(
    source_input_path: str,
    target_input_path: str,
    output_path: str,
    delimiter: str = "\t",
    encoding: str = "utf-8"
):
    """
    Creates the paired source and target dataset from the separated ones.
    e.g. creates `train.tsv` from `train.de` and `train.en`
    """
    with open(source_input_path, encoding=encoding) as f_source_in, \
         open(target_input_path, encoding=encoding) as f_target_in, \
         open(output_path, "w", encoding=encoding) as f_out:

        for source_raw in f_source_in:
            source_raw = source_raw.strip()
            target_raw = f_target_in.readline().strip()
            if source_raw and target_raw:
                output_line = source_raw + delimiter + target_raw + "\n"
                f_out.write(output_line)
