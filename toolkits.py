import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def read_spectrum(fpath: str, band: str):
    """
    Read a spectrum from a fits file.
    """
    assert band in ['u', 'g', 'r', 'i', 'z'], 'Band must be one of u, g, r, i, z.'
    band_idx_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4}
    with fits.open(fpath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    N = data.shape[1]
    COEFF0 = float(header['COEFF0'])
    COEFF1 = float(header['COEFF1'])
    wl = 10**(COEFF0 + COEFF1 * np.arange(N))
    flux = data[band_idx_dict[band], :]
    return wl, flux


def read_template_spectrum(band: str):
    """
    Read a template spectrum.

    To be implemented.
    For now, we use the ID: 2314120
    """
    fpath = './spectrums/2314120.fits'
    wl, flux = read_spectrum(fpath, band)
    return wl, flux


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


def normalization(wl, flux):
    """
    Normalize the spectrum.
    """
    mean = np.mean(flux)
    std = np.std(flux)
    flux = (flux - mean) / std
    return wl, flux


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