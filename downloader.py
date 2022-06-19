import os
import wget


def download_fits_from_dr7(obs_id: int, save_to: str='./'):
    """
    Download the fits file from the DR7 website.
    """
    url = f'https://dr7.lamost.org/spectrum/fits/{obs_id:d}'
    filename = f'{obs_id:d}.fits'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, filename)
    wget.download(url, filepath)

