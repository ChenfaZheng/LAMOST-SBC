from signal import valid_signals
import numpy as np
import matplotlib.pyplot as plt
import os

from toolkits import read_results, read_list_file


if __name__ == '__main__':
    work_dir = './results/'
    # read data
    result_all_path = os.path.join(work_dir, 'results.txt')
    res_arr = read_results(result_all_path)
    v_list_path = os.path.join(work_dir, 'results_v.txt')
    v_dict = read_list_file(v_list_path)
    corr_list_path = os.path.join(work_dir, 'results_corr.txt')
    corr_dict = read_list_file(corr_list_path)

    # mask if the obsid not match among three files
    obsid_arr = res_arr['obsid']
    obsid_mask = []
    for obsid in obsid_arr:
        if obsid in v_dict.keys() and obsid in corr_dict.keys():
            obsid_mask.append(True)
        else:
            obsid_mask.append(False)
    obsid_mask = np.array(obsid_mask)
    res_arr = res_arr[obsid_mask]
    obsid_arr = obsid_arr[obsid_mask]

    # drop masked data in dict
    v_dict = {k: v for k, v in v_dict.items() if k in obsid_arr}
    corr_dict = {k: v for k, v in corr_dict.items() if k in obsid_arr}

    # report mask status
    total_count = len(obsid_mask)
    mask_count = len(obsid_mask) - np.sum(obsid_mask)
    print('Total count: {}, Masked count: {}'.format(total_count, mask_count))

    # plot
    n_bins = 100

    # histogram of v_mean
    v_mean_arr = res_arr['v_shift_mean']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_mean (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_mean_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_mean - v_lamost
    v_lamost_arr = res_arr['v_lamost']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr - v_lamost_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_mean - v_lamost (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean - v_lamost')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_mean_v_lamost_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_lamost
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_lamost_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_lamost (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_lamost')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_lamost_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_std
    v_std_arr = res_arr['v_shift_std']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_std_arr, bins=n_bins, density=True)
    ax.set_xlabel('v_std (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_std')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_std_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v_mean weighted by corr_mean
    corr_mean_arr = res_arr['corr_mean']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(v_mean_arr, bins=n_bins, density=True, weights=corr_mean_arr)
    ax.set_xlabel('v_mean (km/s)')
    ax.set_ylabel('density')
    ax.set_title('Histogram of v_mean weighted')
    plt.tight_layout()
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
    obsid_arr = res_arr['obsid']
    nrows = len(v_dict[obsid_arr[0]])
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
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
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
    fig.suptitle('Histogram of v - v_lamost for all segments')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments weighted by corr
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
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
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
    fig.suptitle('Histogram of v - v_lamost for all segments weighted')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_weighted_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments in [-200, 200]
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        dvs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            v_this = v_dict[obsid][i]
            if v_this < -200 or v_this > 200:
                continue
            dvs.append(v_dict[obsid][i] - v_lamost_arr[j])
        dvs = np.array(dvs)
        ax.hist(dvs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v - v_lamost (km/s)')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
        ax.set_xlim([-200, 200])
    fig.suptitle('Histogram of v - v_lamost for all segments [-200, 200]')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_hist_200.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments in [-100, 100]
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        dvs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            v_this = v_dict[obsid][i]
            if v_this < -100 or v_this > 100:
                continue
            dvs.append(v_dict[obsid][i] - v_lamost_arr[j])
        dvs = np.array(dvs)
        ax.hist(dvs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v - v_lamost (km/s)')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
        ax.set_xlim([-100, 100])
    fig.suptitle('Histogram of v - v_lamost for all segments [-100, 100]')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_hist_100.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v - v_lamost for all segments in [-50, 50]
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        dvs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            v_this = v_dict[obsid][i]
            if v_this < -50 or v_this > 50:
                continue
            dvs.append(v_dict[obsid][i] - v_lamost_arr[j])
        dvs = np.array(dvs)
        ax.hist(dvs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v - v_lamost (km/s)')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
        ax.set_xlim([-50, 50])
    fig.suptitle('Histogram of v - v_lamost for all segments [-50, 50]')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_v_lamost_seg_hist_50.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v for all segments
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        vs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            v_this = v_dict[obsid][i]
            vs.append(v_this)
        vs = np.array(vs)
        ax.hist(vs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v (km/s)')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
    fig.suptitle('Histogram of v for all segments')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_seg_hist.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # histogram of v for all segments in [-200, 200]
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, 1, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):    # i for each segment
        vs = []
        for j, obsid in enumerate(obsid_arr):   # j for each obsid
            v_this = v_dict[obsid][i]
            if v_this < -200 or v_this > 200:
                continue
            vs.append(v_this)
        vs = np.array(vs)
        ax.hist(vs, bins=n_bins, density=True, label=seg_labels[i])
        ax.legend(loc='upper right')
        if i == 0:
            ax.set_ylabel('density')
        if i == nrows - 1:
            ax.set_xlabel('v (km/s)')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.8)
        ax.set_xlim([-200, 200])
    fig.suptitle('Histogram of v for all segments [-200, 200]')
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_seg_hist_200.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()