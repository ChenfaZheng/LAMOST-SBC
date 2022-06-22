import os
import wget
from toolkits import gen_model_spectrum_name


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


def download_model_spectrum(
        ins: int, 
        metal: float, 
        carbon: float, 
        alpha: float, 
        teff: float, 
        logg: float,
        save_to: str='./'
    ):
    """
    Download the model spectrum from BOSZ stellar model.

    Args should be already grided 
    (see Table 1 in Bohlin et al. 2017, AJ, 153, 234 for detail).
    """
    fname = gen_model_spectrum_name(ins, metal, carbon, alpha, teff, logg)
    url = f'http://archive.stsci.edu/missions/hlsp/bosz/fits/insbroad_{ins:06d}/metal_{metal:+.2f}/carbon_{carbon:+.2f}/alpha_{alpha:+.2f}/{fname}'
    print(f'Now downloading {url}')
    filepath = os.path.join(save_to, fname)
    wget.download(url, filepath)
