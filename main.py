from downloader import download_fits_from_dr7
from toolkits import (
    read_spectrum, 
    read_template_spectrum,
    wave_cut,
    moving_average,
    normalization, 
    shift_with_velocity,
)
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import os


def download_spectrum():
    save_dir = './spectrums/'
    obs_ids = [
        2314120, 
    ]
    for obs_id in obs_ids:
        download_fits_from_dr7(obs_id, save_dir)
    

if __name__ == '__main__':
    # download spectrums (already downloaded 2314120.fits)
    # download_spectrum()
    # target
    obs_id = 2314120
    save_dir = './spectrums/'
    fpath = os.path.join(save_dir, f'{obs_id:d}.fits')
    wl, flux = read_spectrum(fpath, band='g')   # wavelength (unit in A) and flux
    flux = moving_average(flux, n=51)
    wln, fluxn = normalization(wl, flux)        # 'n' for normalized
    # template
    wl0, flux0 = read_template_spectrum(band='g')
    flux0 = moving_average(flux0, n=51)
    wl0n, flux0n = normalization(wl0, flux0)
    v_shifts = np.arange(-1000, 1000, 5)    # km/s
    wl0ns = [shift_with_velocity(wl0n, v) for v in v_shifts]
    flux0ns = [flux0n] * len(wl0ns)
    # cut the spectrum
    wl_min = 3900
    wl_max = 9100
    wln, fluxn = wave_cut(wln, fluxn, wl_min, wl_max)
    for i in range(len(wl0ns)):
        wl0ns[i], flux0ns[i] = wave_cut(wl0ns[i], flux0ns[i], wl_min, wl_max)
    # interpolate the spectrum
    f_target = interp1d(wln, fluxn)
    f_templates = [interp1d(wl0ns[i], flux0ns[i]) for i in range(len(wl0ns))]
    # re-sample the spectrum
    wlr_min = 4000
    wlr_max = 9000
    wlr = np.arange(wlr_min, wlr_max, 1)
    # check the correlation
    corrs = []
    for f_template in f_templates:
        corrs.append(pearsonr(f_target(wlr), f_template(wlr))[0])
    corr_max_arg = np.argmax(corrs)
    v_shift = v_shifts[corr_max_arg]
    print(f'v_shift = {v_shift} km/s')
    
    # plot result
    fig_save_dir = './figures/'
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(211)
    ax1.plot(wln, fluxn, 'k.', alpha=0.8, label='target normalized')
    ax1.plot(wlr, f_target(wlr), 'r-', alpha=0.8, label='interpolated')
    ax1.set_xlabel('wavelength (A)')
    ax1.set_ylabel('flux')
    ax1.set_xlim(wl_min, wl_max)
    ax1.legend()
    ax2 = fig1.add_subplot(212)
    ax2.plot(wl0n, flux0n, 'k.', alpha=0.8, label='template normalized')
    ax2.set_xlabel('wavelength (A)')
    ax2.set_ylabel('flux')
    ax2.set_xlim(wl_min, wl_max)
    ax2.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(fig_save_dir, f'{obs_id:d}_spectrum.png'), dpi=300)
    plt.close()

    fig2 = plt.figure(figsize=(10, 3))
    ax = fig2.add_subplot(111)
    ax.plot(v_shifts, corrs, 'k.', alpha=0.8, label='correlation')
    ax.axvline(v_shift, color='r', alpha=0.8, label=f'v_shift = {v_shift} km/s')
    ax.set_xlabel('velocity shift (km/s)')
    ax.set_ylabel('correlation coefficient')
    ax.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(fig_save_dir, f'{obs_id:d}_correlation.png'), dpi=300)
    plt.close()


