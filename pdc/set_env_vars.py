import os

def set_all_env_vars():
    """
    Set environment variable AKI_DATA_PDC to the path of the directory where
    you'd like to store data files. Make sure that it ends with "/" in order
    to work properly with code that references files within the directory.
    """
    os.environ['AKI_DATA_PDC'] = 'YOUR_DATA_DIRECTORY_PATH_HERE'