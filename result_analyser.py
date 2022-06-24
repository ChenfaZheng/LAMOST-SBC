from signal import valid_signals
import numpy as np
import matplotlib.pyplot as plt
import os


def read_results(fpath: str):
    """
    Read the results from a file.
    """
    result = np.loadtxt(
        fpath, 
        delimiter=',', 
        skiprows=1, 
        usecols=[0, 1, 2, 3, 4, 5, 6, 7], 
        dtype={
            'names': ('obsid', 'v_shift_mean', 'v_shift_std', 'corr_mean', 'corr_std', 'v_lamost', 'spid', 'fiberid'),
            'formats': ('i', 'f', 'f', 'f', 'f', 'f', 'i', 'i')
        }
    )
    return result


def read_list_file(fpath: str):
    """
    Read the list file.
    """
    with open(fpath, 'r') as f:
        lines = f.readlines()
    res = {}
    for line in lines:
        if line.startswith('obsid'):
            continue
        if line.startswith('#'):
            continue
        if len(line.split()) == 0:
            continue
        line_list = line.split(',')
        obsid = int(line_list[0])
        v_str_list = [v for v in line_list[1:] if v not in ['', '\n']]
        v_list = [float(v) for v in v_str_list]
        v_arr = np.array(v_list)
        res[obsid] = v_arr
    return res


if __name__ == '__main__':
    work_dir = './results/'
    # read data
    result_all_path = os.path.join(work_dir, 'results.txt')
    res_arr = read_results(result_all_path)
    v_list_path = os.path.join(work_dir, 'results_v.txt')
    v_dict = read_list_file(v_list_path)
    corr_list_path = os.path.join(work_dir, 'results_corr.txt')
    corr_dict = read_list_file(corr_list_path)

    # plot
    n_bins = 20

    # histogram of v_mean
    v_mean_arr = res_arr[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_mean (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean')
    fig_save_path = os.path.join(work_dir, 'v_mean_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_mean - v_lamost
    v_lamost_arr = res_arr[:, 5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr - v_lamost_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_mean - v_lamost (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean - v_lamost')
    fig_save_path = os.path.join(work_dir, 'v_mean_v_lamost_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_std
    v_std_arr = res_arr[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_std_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_std (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_std')
    fig_save_path = os.path.join(work_dir, 'v_std_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_mean weighted by corr_mean
    corr_mean_arr = res_arr[:, 3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr, bins=n_bins, density=True, weights=corr_mean_arr)
    ax.set_xlabel('v_mean (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean weighted')
    fig_save_path = os.path.join(work_dir, 'v_mean_weighted_by_corr_mean_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments
    seg_labels = [
        '4000-4499', 
        '4500-4999',
        '5000-5499',
        '5500-5999',
        '6000-6499',
        '6500-6999',
        '7000-7499',
        '7500-7999',
        '8000-8499',
        '8500-8999',
    ]
    obsid_arr = res_arr[:, 0]
    nrows = len(v_dict[obsid_arr[0]])
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        dvs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            dvs.append(v_dict[obsid][i] - v_lamost_arr[j])
        dvs = np.array(dvs)
        ax.hist(dvs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v - v_lamost (km/s)')
    fig.suptitle('Histogram of v - v_lamost for all segments')
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments weighted by corr
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        dvs = []
        corrs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            dvs.append(v_dict[obsid][i] - v_lamost_arr[j])
            corrs.append(corr_dict[obsid][i])
        dvs = np.array(dvs)
        corrs = np.array(corrs)
        ax.hist(dvs, bins=n_bins, density=True, weights=corrs, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v - v_lamost (km/s)')
    fig.suptitle('Histogram of v - v_lamost for all segments weighted')
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_weighted_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()


