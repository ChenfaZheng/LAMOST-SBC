from downloader import (
    download_fits_from_dr7, 
    download_model_spectrum, 
)
from toolkits import (
    read_spectrum, 
    gen_model_spectrum_name, 
    read_model_spectrum,
    wave_cut,
    moving_average,
    normalization, 
    shift_with_velocity,
    read_catalog, 
)
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import yaml

import os


def load_spectrum(save_dir):
    files = os.listdir(save_dir)
    files = [f for f in files if f.endswith('.fits')]
    fpaths = [os.path.join(save_dir, f) for f in files]
    return fpaths


def download_spectrum():
    save_dir = './spectrums/'
    obs_ids = [
        2314120, 
        101008, 
        101009, 
        101030, 
    ]
    for obs_id in obs_ids:
        download_fits_from_dr7(obs_id, save_dir)


def spectrum_args_gridder(
        ins: float,
        metal: float,
        carbon: float,
        alpha: float,
        teff: float,
        logg: float,
    ):
    """
    Grid the spectrum arguments.

    See Table 1 in Bohlin et al. 2017, AJ, 153, 234 for detail.

    The arg `ins` is the spectral resolution R. The range selected from 
    https://archive.stsci.edu/hlsp/bosz/search.php
    """
    if ins < 200 or ins > 300000:
        raise ValueError(f'ins should be between 200 and 300000, got {ins}')
    if metal < -2.5 or metal > 0.5:
        raise ValueError(f'metal should be between -2.5 and 0.5, got {metal}')
    if carbon < -0.75 or carbon > 0.5:
        raise ValueError(f'carbon should be between -0.75 and 0.5, got {carbon}')
    if alpha < -0.25 or alpha > 0.5:
        raise ValueError(f'alpha should be between -0.25 and 0.5, got {alpha}')
    if teff < 3500 or teff > 30000:
        raise ValueError(f'teff should be between 3500 and 30000, got {teff}')
    if logg < 0 or logg > 5:
        raise ValueError(f'logg should be between 0 and 5, got {logg}')

    def find_nearest_value(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    ins_grid = np.array([
        200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 300000
    ], dtype=int)
    ins = find_nearest_value(ins_grid, ins)

    metal = find_nearest_value(
        np.arange(-2.5, 0.5 + 0.25, 0.25), metal
    )
    carbon = find_nearest_value(
        np.arange(-0.75, 0.5 + 0.25, 0.25), carbon
    )
    alpha = find_nearest_value(
        np.arange(-0.25, 0.5 + 0.25, 0.25), alpha
    )
    
    teff_list = [
        np.arange(3500, 6000 + 250, 250), 
        np.arange(6250, 8000 + 250, 250),
        np.arange(8250, 12000 + 250, 250),
        np.arange(12500, 20000 + 500, 500),
        np.arange(21000, 30000 + 1000, 1000),
    ]
    teff = find_nearest_value(
        np.concatenate(teff_list), teff
    )
    if teff >= 3500 and teff <= 6000:
        if logg < 0 or logg > 5:
            raise ValueError('logg must be between 0 and 5 when teff is between 3500 and 6000')
        logg = find_nearest_value(
            np.arange(0, 5 + 0.5, 0.5), logg
        )
    elif teff >= 6250 and teff <= 8000:
        if logg < 1 or logg > 5:
            raise ValueError('logg must be between 1 and 5 when teff is between 6250 and 8000')
        logg = find_nearest_value(
            np.arange(1, 5 + 0.5, 0.5), logg
        )
    elif teff >= 8250 and teff <= 12000:
        if logg < 2 or logg > 5:
            raise ValueError('logg must be between 2 and 5 when teff is between 8250 and 12000')
        logg = find_nearest_value(
            np.arange(2, 5 + 0.5, 0.5), logg
        )
    elif teff >= 12500 and teff <= 20000:
        if logg < 3 or logg > 5:
            raise ValueError('logg must be between 3 and 5 when teff is between 12500 and 20000')
        logg = find_nearest_value(
            np.arange(3, 5 + 0.5, 0.5), logg
        )
    elif teff >= 21000 and teff <= 30000:
        if logg < 4 or logg > 5:
            raise ValueError('logg must be between 4 and 5 when teff is between 21000 and 30000')
        logg = find_nearest_value(
            np.arange(4, 5 + 0.5, 0.5), logg
        )
    return ins, metal, carbon, alpha, teff, logg
    

def main():
    # --------------------------------------------------
    # Read the configuration file
    # --------------------------------------------------
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec_dir = config['spectrum_dir']
    model_dir = config['model_dir']
    cat_path = config['catalog_path']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fig_save_dir = config['figure_dir']
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    
    ins = float(config['instrumental_broadening'])  # Instrumental broadening
    carbon = float(config['carbon'])  # Carbon
    band = config['band']  # Band
    mv_window = int(config['moving_average_window'])  # moving average window

    cut_min, cut_max = float(config['cut_min']), float(config['cut_max'])
    wl_min, wl_max, wl_step = float(config['wl_min']), float(config['wl_max']), float(config['wl_step'])
    v_shift_min, v_shift_max, v_shift_step = float(config['v_shift_min']), float(config['v_shift_max']), float(config['v_shift_step'])

    plot_min, plot_max = float(config['plot_min']), float(config['plot_max'])

    # --------------------------------------------------
    # Load the catalog
    # --------------------------------------------------
    cata = read_catalog(cat_path)    # note the memory consumption is large (~2GB)

    # --------------------------------------------------
    # Load the spectra
    # --------------------------------------------------
    files = os.listdir(spec_dir)
    files = [f for f in files if f.endswith('.fits')]
    fpaths = [os.path.join(spec_dir, f) for f in files]
    nfiles = len(fpaths)
    for idx, fpath in enumerate(fpaths):
        print(f'Now {idx+1}/{nfiles} ', fpath)
        wl, flux, obs_id = read_spectrum(fpath, band=band)
        # get the model spectrum
        cataline = cata[cata['obsid'] == obs_id]
        if len(cataline) == 0:
            print(f'obsid {obs_id} not found in the catalog')
            continue
        metal = float(cataline['feh'][0])
        alpha = float(cataline['alpha_m'][0])
        teff = float(cataline['teff'][0])
        logg = float(cataline['logg'][0])
        carbon = float(carbon)  # pre-defined in the config file
        ins = float(ins)  # pre-defined in the config file
        try:
            ins, metal, carbon, alpha, teff, logg = spectrum_args_gridder(
                ins, metal, carbon, alpha, teff, logg
            )
        except ValueError as e:
            print(e)
            continue
        model_fname = gen_model_spectrum_name(
            ins, metal, carbon, alpha, teff, logg
        )
        model_fpath = os.path.join(model_dir, model_fname)
        if not os.path.exists(model_fpath):
            print(f'Model spectrum {model_fname} not found.')
            try:
                download_model_spectrum(
                    ins, metal, carbon, alpha, teff, logg, model_dir
                )
            except Exception as e:
                print(e)
                continue
        wlm, fluxm = read_model_spectrum(model_fpath)

        # --------------------------------------------------
        # Preprocessing the spectrum
        # --------------------------------------------------
        # target spectrum
        flux = moving_average(flux, n=mv_window)
        wl, flux = normalization(wl, flux)
        # model spectrum
        fluxm = moving_average(fluxm, n=mv_window)
        wlm, fluxm = normalization(wlm, fluxm)
        # shift the model spectrum on the wavelength by velocity
        v_shifts = np.arange(v_shift_min, v_shift_max + v_shift_step, v_shift_step)
        wlms = [shift_with_velocity(wlm, v) for v in v_shifts]
        fluxms = [fluxm] * len(v_shifts)
        # --------------------------------------------------
        # Interpolate the spectrum
        # --------------------------------------------------
        # cut the spectrums
        wl, flux = wave_cut(wl, flux, cut_min, cut_max)
        for i, v in enumerate(v_shifts):
            wlms[i], fluxms[i] = wave_cut(wlms[i], fluxms[i], cut_min, cut_max)
        # interpolate the spectrums
        f_target = interp1d(wl, flux)
        f_models = [interp1d(wlms[i], fluxms[i]) for i in range(len(v_shifts))]
        # set resample wavelength
        wlr = np.arange(wl_min, wl_max + wl_step, wl_step)
        # --------------------------------------------------
        # Check the correlation
        # --------------------------------------------------
        corrs = []
        for i, v in enumerate(v_shifts):
            corr = pearsonr(f_target(wlr), f_models[i](wlr))[0]
            corrs.append(corr)
        corr_max_arg = np.argmax(corrs)
        v_shift_best = v_shifts[corr_max_arg]
        print(f'v_shift_best = {v_shift_best}')
        wlm_best = shift_with_velocity(wlm, v_shift_best)
        fluxm_best = fluxm

        # --------------------------------------------------
        # Save the result
        # --------------------------------------------------
        fig1 = plt.figure(figsize=(10, 5))
        ax1 = fig1.add_subplot(211)
        ax1.plot(wl, flux, 'r.', alpha=0.8, label='target normalized')
        ax1.plot(wlm_best, fluxm_best, 'k-', alpha=0.5, label='model best')
        ax1.set_xlabel('wavelength (A)')
        ax1.set_ylabel('flux')
        ax1.set_xlim(plot_min, plot_max)
        ax1.legend()
        ax2 = fig1.add_subplot(212)
        ax2.plot(wlm, fluxm, 'k.', alpha=0.8, label='model normalized')
        ax2.set_xlabel('wavelength (A)')
        ax2.set_ylabel('flux')
        ax2.set_xlim(plot_min, plot_max)
        ax2.legend()
        plt.tight_layout()
        fig1.savefig(os.path.join(fig_save_dir, f'{obs_id:d}_spectrum.png'), dpi=300)
        plt.close()

        fig2 = plt.figure(figsize=(10, 3))
        ax = fig2.add_subplot(111)
        ax.plot(v_shifts, corrs, 'k.', alpha=0.8, label='correlation')
        ax.axvline(v_shift_best, color='r', alpha=0.8, label=f'v_shift = {v_shift_best} km/s')
        ax.set_xlabel('velocity shift (km/s)')
        ax.set_ylabel('correlation coefficient')
        ax.legend()
        plt.tight_layout()
        fig2.savefig(os.path.join(fig_save_dir, f'{obs_id:d}_correlation.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    main()