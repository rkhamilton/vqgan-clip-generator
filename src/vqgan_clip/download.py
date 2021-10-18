from torch.hub import download_url_to_file, get_dir
import os
from urllib.parse import urlparse

# Install downloaded files in a cache location.
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.getcwd()

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    Ref: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'models')
    else:
        model_dir = os.path.join(ROOT_DIR, model_dir)
    
    os.makedirs(model_dir, exist_ok=True)

    url_parts = urlparse(url)
    url_filename = os.path.basename(url_parts.path)
    if file_name is not None:
        url_filename = file_name
    if not url_filename:
        raise ValueError('The URL did not return a filename. Manually set a filename with argument file_name.')
    cached_file = os.path.abspath(os.path.join(ROOT_DIR, model_dir, url_filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file