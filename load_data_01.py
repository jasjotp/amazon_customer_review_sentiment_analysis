import os 
from kaggle.api.kaggle_api_extended import KaggleApi
from colorama import Fore, Style

def download_kaggle_dataset(storage_path, data_path, kaggle_path):
    '''
    Downloads and unzips a Kaggle dataset to the specified directory if the expected database file does not exist.

    Parameters:
    - storage_path (str): relative path to the local directory where the dataset should be stored.
    - data_path (str): relative path to the expected dataset file (e.g., 'database.sqlite').
    - kaggle_path (str): kaggle dataset path in the format 'owner/dataset-name'.

    Returns:
    - None
    '''

    api = KaggleApi()
    api.authenticate()

    db_file = os.path.join(storage_path, data_path)

    if not os.path.exists(db_file):
        print(f'Downloading dataset: {kaggle_path} to {storage_path}')
        api.dataset_download_files(kaggle_path, path = storage_path, unzip = True)
        
        # check if the file was downloaded (exists)
        if os.path.exists(db_file):
            print(f'{Fore.GREEN}✓ Dataset: {db_file} downloaded successfully.{Style.RESET_ALL}')
        else:
            print(f'{Fore.RED}✗ Download finished, but file not found. Please check manually.{Style.RESET_ALL}')

    else:
        print(f'{Fore.YELLOW}✓ Dataset already exists. Skipping download.{Style.RESET_ALL}')

if __name__ == '__main__':
    download_kaggle_dataset(storage_path = 'data', 
                            data_path = 'database.sqlite', 
                            kaggle_path = 'snap/amazon-fine-food-reviews')
