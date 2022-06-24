import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def read_spectrum(fpath: str):
    """
    Read a spectrum from a fits file.
    Return its wavelength and flux, and observation ID.
    """
    with fits.open(fpath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    N = data.shape[1]
    COEFF0 = float(header['COEFF0'])
    COEFF1 = float(header['COEFF1'])
    wl = data[2, :]
    flux = data[0, :]
    obs_id = int(header['OBSID'])
    return wl, flux, obs_id


def gen_model_spectrum_name(
        ins: int, 
        metal: float, 
        carbon: float, 
        alpha: float, 
        teff: float, 
        logg: float,
    ):
    """
    Generate the model spectrum name.
    Spectrum from BOSZ stellar model.

    Args should be already grided 
    (see Table 1 in Bohlin et al. 2017, AJ, 153, 234 for detail).
    """
    arg_metal = 'mp' if metal >= 0 else 'mm'
    arg_carbon = 'cp' if carbon >= 0 else 'cm'
    arg_alpha = 'op' if alpha >= 0 else 'om'
    round_away = lambda x: int(np.ceil(x)) if x >= 0 else int(np.floor(x))
    fname = ''.join([                               # see Appendix in Bohlin et al. 2017
        'a',                                        # the source is an ATLAS model
        f'{arg_metal}{abs(round_away(10 * metal)):02d}', # the metalicity
        f'{arg_carbon}{abs(round_away(10 * carbon)):02d}',    # the carbon content
        f'{arg_alpha}{abs(round_away(10 * alpha)):02d}',      # the alpha content
        f't{round_away(teff):d}',                        # the effective temperature
        f'g{round_away(10 * logg):02d}',                 # the surface gravity
        'v20',                                      # the microturbulent velocity, always 20 for 2 km/s
        'mod', 
        'rt0',                                      # the rotational broadening; vrot is always 0
        f'b{round_away(ins):d}',                         # the spectral resolution R
        'rs',                                       # the spectra are resampled at two points per resolution element
        '.fits',
    ])
    return fname


def read_model_spectrum(fpath: str):
    """
    Read a model spectrum.

    Note the flux in the fits file is surface brightness $H$.
    Hence $F = \pi * H$ (see eq.2 in Bohlin et al. 2017, AJ, 153, 234).
    """
    tab = Table.read(fpath)
    wl = np.array(tab['Wavelength'])
    flux = np.pi * np.array(tab['SpecificIntensity'])
    return wl, flux


def read_catalog(fpath: str):
    """
    Read the catalog.
    """
    return Table.read(fpath)


def get_spectrum_args_from_catalog(cata_path: str, obs_id: int):
    """
    Get the spectrum arguments from 
    LAMOST LRS Stellar Parameter Catalog of A, F, G and K Stars.
    
    The catalog is downloaded from the DR7 website.
    For detail see https://dr7.lamost.org/v2.0/doc/lr-data-production-description#s3

    TODO : Optimize the speed of this function.
    The catalog is readed every time.
    """
    cata = Table.read(cata_path)
    cataline = cata[cata['obsid'] == obs_id]
    metal = cataline['feh'][0]
    alpha = cataline['alpha_m'][0]
    teff = cataline['teff'][0]
    logg = cataline['logg'][0]
    return metal, alpha, teff, logg


def wave_cut(wl, flux, wl_min, wl_max):
    """
    Cut the spectrum in wavelength.
    """
    idx_min = np.argmin(np.abs(wl - wl_min))
    idx_max = np.argmin(np.abs(wl - wl_max))
    wl = wl[idx_min:idx_max]
    flux = flux[idx_min:idx_max]
    return wl, flux


def moving_average(x, n=3):
    """
    Moving average.
    """
    return np.convolve(x, np.ones(n), 'same') / n


def normalization(wl, flux, norm_type='mean', flux_raw=None):
    """
    Normalize the spectrum.
    """
    if norm_type in ['max']:
        max_val = np.max(flux)
        return wl, flux / max_val
    if norm_type in ['mean']:
        mean = np.mean(flux)
        std = np.std(flux)
        flux = (flux - mean) / std
        return wl, flux
    if norm_type in ['divide']:
        if flux_raw is None:
            raise ValueError('flux_raw is required for normalization type "devide"')
        return wl, flux_raw / flux
    raise ValueError(f'Unknown normalization type: {norm_type}')


def shift_with_velocity(wl, v):
    """
    Shift the wavelength with velocity.

    Velocity is in km/s.
    """
    return wl * (1 + v / 299792.458)


def plot_spectrum(wl, flux, title=None, xlabel=None, ylabel=None, save_to=None, show=False):
    """
    Plot a spectrum.
    """
    fig, ax = plt.subplots()
    ax.plot(wl, flux)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if save_to is not None:
        fig.savefig(save_to)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    pass