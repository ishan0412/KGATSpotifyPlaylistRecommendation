"""Utility methods for downloading the Spotify playlist and track datasets this project uses."""

import os
import zipfile

import requests
from dotenv import load_dotenv

load_dotenv()

def download_and_extract_url(url: str, name: str):
    """
    Downloads and uncompresses a dataset from a URL if it isn't already there.
    #### Parameters:
        - `url`: the URL to download the dataset from
        - `name`: the name to give the dataset .zip file
    #### Throws:
        - `RuntimeError`: if the download from the URL fails
    """
    # Makes a directory to download the dataset at:
    path_to_dataset = os.path.join(os.getenv('DATA_DIR'), name)
    path_to_dataset_zip_file = path_to_dataset + '.zip'
    # Checks if the dataset hasn't already been downloaded there:
    if os.path.exists(path_to_dataset):
        print(f'Dataset {name} has already been downloaded.')
        return

    # Obtains the dataset as an HTTP response from the URL:
    response = requests.get(url, stream=True, allow_redirects=True)
    if response.status_code == requests.codes.ok:
        with open(path_to_dataset_zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f'Downloaded to {path_to_dataset_zip_file}.')

        # Since the dataset has been downloaded as a .zip, extract it:
        with zipfile.ZipFile(path_to_dataset_zip_file, 'r') as zipref:
            zipref.extractall(path_to_dataset)
        print(f'Extracted at {path_to_dataset}.')
        os.remove(path_to_dataset_zip_file)
    else:
        raise RuntimeError(f'Failed to download the dataset at {url} with an HTTP ' 
                           + f'{response.status_code}: {response.text}.')
