from downloader import (
    download_fits_from_dr7, 
    download_model_spectrum, 
)
from plotter import (
    plot_spectrum,
    plot_corrs, 
    plot_shifted_spectrum_segs, 
)
from toolkits import (
    read_results, 
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
import yaml

import os
import argparse


def download_spectrum():
    save_dir = './spectrums/'
    obs_ids = [
        2314120, 
        # 101008, 
        # 101017, 
        # 19210055, 
        # 27904160, 
        # 48013172, 
        # 55203160, 
    ]
    for obs_id in obs_ids:
        download_fits_from_dr7(obs_id, save_dir)


def clear_generated_figures(fig_dir):
    for f in os.listdir(fig_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(fig_dir, f))


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



def main(parser):
    # --------------------------------------------------
    # Read the configuration file
    # --------------------------------------------------
    # check if --config is given
    if parser.config is None:
        raise ValueError('--config is required')
    # read the configuration file
    with open(parser.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec_dir = config['spectrum_dir']
    model_dir = config['model_dir']
    cat_path = config['catalog_path']
    result_dir = config['result_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fig_save_dir = config['figure_dir']
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # clear_generated_figures(fig_save_dir)
    
    ins = float(config['instrumental_broadening'])  # Instrumental broadening
    carbon = float(config['carbon'])  # Carbon
    mv_window = int(config['moving_average_window'])  # moving average window
    norm_type = config['norm_type']   # normalization type
    opt_method = config['opt_method']  # optimization method

    cut_min, cut_max = float(config['cut_min']), float(config['cut_max'])
    wl_min, wl_max, wl_step = float(config['wl_min']), float(config['wl_max']), float(config['wl_step'])
    wl_seg = float(config['wl_seg'])
    v_shift_min, v_shift_max, v_shift_step = float(config['v_shift_min']), float(config['v_shift_max']), float(config['v_shift_step'])

    plot_min, plot_max = float(config['plot_min']), float(config['plot_max'])

    skip_analysed = config['skip_analysed']

    # get analysed obsid list
    obs_id_analysed = []
    if skip_analysed:
        res_file_path = os.path.join(result_dir, 'results.txt')
        if os.path.exists(res_file_path):
            res_analysed = read_results(res_file_path)
            for this_id in res_analysed['obsid']:
                obs_id_analysed.append(int(this_id))

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

    # --------------------------------------------------
    # Check result files
    # --------------------------------------------------
    result_path = os.path.join(result_dir, 'results.txt')
    result_v_path = os.path.join(result_dir, 'results_v.txt')
    result_corr_path = os.path.join(result_dir, 'results_corr.txt')

    header = 'obsid,v_shift_mean,v_shift_std,corr_mean,corr_std,v_lamost,spid,fiberid,\n'
    if (not os.path.exists(result_path)) or (not skip_analysed):
        file_result = open(result_path, 'w', encoding='utf-8')
        file_result.write(header)
    else:
        with open(result_path, 'r', encoding='utf-8') as file_result:
            file_header = file_result.readline()
            if file_header != header:
                raise ValueError(f'Invalid header format for results file! Check {result_path}')
        file_result = open(result_path, 'a', encoding='utf-8')

    n_seg = len(np.arange(wl_min, wl_max, wl_seg))

    header = 'obsid,' + ','.join([f'v_{i}' for i in range(n_seg)]) + ',\n'
    if not os.path.exists(result_v_path) or (not skip_analysed):
        file_result_v = open(result_v_path, 'w', encoding='utf-8')
        file_result_v.write(header)
    else:
        with open(result_v_path, 'r', encoding='utf-8') as file_result_v:
            file_header = file_result_v.readline()
            if file_header != header:
                raise ValueError(f'Invalid header format for results file! Check {result_v_path}')
        file_result_v = open(result_v_path, 'a', encoding='utf-8')

    header = 'obsid,' + ','.join([f'corr_{i}' for i in range(n_seg)]) + ',\n'
    if not os.path.exists(result_corr_path) or (not skip_analysed):
        file_result_corr = open(result_corr_path, 'w', encoding='utf-8')
        file_result_corr.write(header)
    else:
        with open(result_corr_path, 'r', encoding='utf-8') as file_result_corr:
            file_header = file_result_corr.readline()
            if file_header != header:
                raise ValueError(f'Invalid header format for results file! Check {result_corr_path}')
        file_result_corr = open(result_corr_path, 'a', encoding='utf-8')
        
    
    # --------------------------------------------------
    # Start the analysis for each spectrum
    # --------------------------------------------------
    
    for idx, fpath in enumerate(fpaths):
        print(f'\nNow {idx+1}/{nfiles} ', fpath)
        wl, flux, obs_id = read_spectrum(fpath)
        if obs_id in obs_id_analysed:
            print(f'{obs_id} has been analysed, skip it')
            continue
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
        rv = float(cataline['rv'][0])   # Heliocentric radial velocity obtained by the LASP
        spid = int(cataline['spid'][0])
        fiberid = int(cataline['fiberid'][0])
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
        flux_ma = moving_average(flux, n=mv_window)
        wl_n, flux_n = normalization(wl, flux_ma, norm_type=norm_type, flux_raw=flux)
        # model spectrum
        fluxm_ma = moving_average(fluxm, n=mv_window)
        wlm_n, fluxm_n = normalization(wlm, fluxm_ma, norm_type=norm_type, flux_raw=fluxm)
        # shift the model spectrum on the wavelength by velocity
        v_shifts = np.arange(v_shift_min, v_shift_max + v_shift_step, v_shift_step)
        wlm_ns = [shift_with_velocity(wlm_n, v) for v in v_shifts]
        fluxm_ns = [fluxm_n] * len(v_shifts)
        wlm_cs = [shift_with_velocity(wlm, v) for v in v_shifts]    # `c` for `cut` in the next step
        fluxm_cs = [fluxm] * len(v_shifts)
        # --------------------------------------------------
        # Interpolate the spectrum
        # --------------------------------------------------
        # cut the spectrums
        wl_n, flux_n = wave_cut(wl_n, flux_n, cut_min, cut_max)
        wl_c, flux_c = wave_cut(wl, flux, cut_min, cut_max)
        for i, v in enumerate(v_shifts):
            wlm_ns[i], fluxm_ns[i] = wave_cut(wlm_ns[i], fluxm_ns[i], cut_min, cut_max)
            wlm_cs[i], fluxm_cs[i] = wave_cut(wlm_cs[i], fluxm_cs[i], cut_min, cut_max)
        # interpolate the spectrums
        fn_target = interp1d(wl_n, flux_n)
        fn_models = [interp1d(wlm_ns[i], fluxm_ns[i]) for i in range(len(v_shifts))]
        fc_target = interp1d(wl_c, flux_c)
        fc_models = [interp1d(wlm_cs[i], fluxm_cs[i]) for i in range(len(v_shifts))]
        # set resample wavelengths
        wlrs = []
        for wlr_tag in np.arange(wl_min, wl_max, wl_seg):
            wlr_seg = np.arange(wlr_tag, wlr_tag + wl_seg, wl_step)
            wlrs.append(wlr_seg)
        # --------------------------------------------------
        # Check the correlation
        # --------------------------------------------------
        corrs = []
        v_peaks = []
        corr_peaks = []
        wl_seg_bests = []
        flux_seg_bests = []
        wl_seg_bests_raw = []
        flux_seg_bests_raw = []
        for i, wlr_seg in enumerate(wlrs):
            corrs_seg = []
            for j, v in enumerate(v_shifts):
                corr = pearsonr(fn_target(wlr_seg), fn_models[j](wlr_seg))[0]
                corrs_seg.append(corr)
            corrs_seg = np.array(corrs_seg)
            corrs.append(corrs_seg)
            # get the peak of the correlation
            corr_max_arg = np.argmax(corrs_seg)
            len_seg = len(corrs_seg)
            peak_width = int(len_seg * 0.05)
            opt_width = peak_width if peak_width > 6 else 6
            corr_idxs = np.arange(len_seg)
            corr_peak_mask = (corr_idxs >= corr_max_arg - opt_width) & (corr_idxs <= corr_max_arg + opt_width)
            corr_seg_peak = corrs_seg[corr_peak_mask]
            v_shift_peak = v_shifts[corr_peak_mask]
            # polynomial fit
            fit_failed_signal = False
            if opt_method in ['quadratic']:
                # quadratic (2nd order) polynomial fit
                peak_paras = np.polyfit(v_shift_peak, corr_seg_peak, 2)
                # y = a*x**2 + b*x + c
                a, b, c = peak_paras
                v_peak = -b / (2 * a)
                corr_peak = a * v_peak ** 2 + b * v_peak + c
                if a >= 0:
                    fit_failed_signal = True
            elif opt_method in ['cubic']:
                # cubic (3nd order) polynomial fit
                peak_paras = np.polyfit(v_shift_peak, corr_seg_peak, 3)
                # y = a*x**3 + b*x**2 + c*x + d
                a, b, c, d = peak_paras
                critical_term = 4*b**2 - 12*a*c
                if critical_term < 0:
                    fit_failed_signal = True
                else:
                    v_peak = (-2*b - np.sqrt(critical_term)) / (6*a)
                    corr_peak = a * v_peak ** 3 + b * v_peak ** 2 + c * v_peak + d
            else:
                raise ValueError(f'Unknown opt_method: {opt_method}')
            if fit_failed_signal:
                print(f'polynomial fit failed to find a peak, use the max value instead')
                v_peak = v_shifts[corr_max_arg]
                corr_peak = corrs_seg[corr_max_arg]
            elif v_peak < v_shift_min or v_peak > v_shift_max:
                print(f'v_peak {v_peak:.2f} is out of range [{v_shift_min}, {v_shift_max}], set to nearest bound')
                v_peak = v_shift_min if v_peak < v_shift_min else v_shift_max
                corr_peak = corrs_seg[np.argmin(np.abs(v_shifts - v_peak))]
            v_peaks.append(v_peak)
            corr_peaks.append(corr_peak)
            # shift target spectrum
            wl_seg_best = shift_with_velocity(wlr_seg, -v_peak)
            flux_seg_best = fn_target(wlr_seg)
            wl_seg_bests.append(wl_seg_best)
            flux_seg_bests.append(flux_seg_best)
            wl_seg_best_raw = shift_with_velocity(wlr_seg, -v_peak)
            flux_seg_best_raw = fc_target(wlr_seg)
            wl_seg_bests_raw.append(wl_seg_best_raw)
            flux_seg_bests_raw.append(flux_seg_best_raw)
        corrs = np.array(corrs)
        v_peaks = np.array(v_peaks)
        v_peaks_mean = np.mean(v_peaks)
        v_peaks_std = np.std(v_peaks)
        corr_peaks = np.array(corr_peaks)
        corr_peaks_mean = np.mean(corr_peaks)
        corr_peaks_std = np.std(corr_peaks)
        print(f'v_peaks_mean {v_peaks_mean:.2f} std {v_peaks_std:.2f}')

        # --------------------------------------------------
        # Save the result
        # --------------------------------------------------
        # plot target spectrum
        spec_min, spec_max = np.min(wl), np.max(wl)
        spec_title = f'{obs_id:d} spectrum'
        spec_path = os.path.join(fig_save_dir, f'{obs_id:d}_spectrum.png')
        plot_spectrum(wl, flux, [spec_min, spec_max], spec_title, spec_path)

        # plot model spectrum
        model_title = f'{obs_id:d} model spectrum {model_fname}'
        model_path = os.path.join(fig_save_dir, f'{obs_id:d}_model_spectrum.png')
        plot_spectrum(wlm, fluxm, [spec_min, spec_max], model_title, model_path)

        # plot correlations
        corr_path = os.path.join(fig_save_dir, f'{obs_id:d}_correlation.png')
        corr_labels = []
        for wlr in wlrs:
            corr_labels.append(f'{wlr[0]:.0f}-{wlr[-1]:.0f}')
        plot_corrs(v_shifts, corrs, v_peaks, corr_peaks, rv, corr_labels, corr_path)

        # plot shifted normed spectrums
        spec_shift_normed_title = f'{obs_id:d} spectrum normed shifted'
        spec_shift_normed_path = os.path.join(fig_save_dir, f'{obs_id:d}_spectrum_normed_shift.png')
        spec_shift_labels = []
        for v_peak in v_peaks:
            spec_shift_labels.append(f'{v_peak:.2f}')
        plot_shifted_spectrum_segs(
            wl_seg_bests, flux_seg_bests, 
            wlm_n, fluxm_n, 
            spec_shift_labels, [cut_min, cut_max], 
            spec_shift_normed_title, spec_shift_normed_path
        )

        # plot shifted spectrums
        spec_shift_title = f'{obs_id:d} spectrum shifted'
        spec_shift_path = os.path.join(fig_save_dir, f'{obs_id:d}_spectrum_shift.png')
        plot_shifted_spectrum_segs(
            wl_seg_bests_raw, flux_seg_bests_raw, 
            wlm, fluxm, 
            spec_shift_labels, [cut_min, cut_max],
            spec_shift_title, spec_shift_path
        )

        # save the result
        file_result.write(
            f'{obs_id:d},{v_peaks_mean:.3f},{v_peaks_std:.3f},{corr_peaks_mean:.3f},{corr_peaks_std:.3f},{rv:.3f},{spid:d},{fiberid:d},\n'
        )
        file_result.flush()
        file_result_v.write(f'{obs_id:d},')
        for v_peak in v_peaks:
            file_result_v.write(f'{v_peak:.3f},')
        file_result_v.write('\n')
        file_result_v.flush()
        file_result_corr.write(f'{obs_id:d},')
        for corr_peak in corr_peaks:
            file_result_corr.write(f'{corr_peak:.3f},')
        file_result_corr.write('\n')
        file_result_corr.flush()
        
        # --------------------------------------------------
        # All done for this spectrum ! ^_^
        # --------------------------------------------------
        print(f'{obs_id:d} done ^_^')
    
    file_result.close()
    file_result_v.close()
    file_result_corr.close()
    print('\nAll done ^_^\n')


if __name__ == '__main__':
    # get the command line arguments
    parser = argparse.ArgumentParser(
        description='Calculate the velocity shift of a spectrum.'
    )
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    main(parser.parse_args())
