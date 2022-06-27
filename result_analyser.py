from signal import valid_signals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

from toolkits import read_results, read_list_file


if __name__ == '__main__':
    work_dir = './results_20190401_HD103621N361936B01_spec3/'
    # read data
    result_all_path = os.path.join(work_dir, 'results.txt')
    res_arr = read_results(result_all_path)
    v_list_path = os.path.join(work_dir, 'results_v.txt')
    v_dict = read_list_file(v_list_path)
    corr_list_path = os.path.join(work_dir, 'results_corr.txt')
    corr_dict = read_list_file(corr_list_path)

    do_previous = False

    if do_previous:

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

    # check velocity segments v.s. fiber_id with color v - v_lamost
    # set v range to [-20, 20]
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    dvmin, dvmax = 0, 0
    for obs_id, v_segs in v_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        v_lamost = result_line['v_lamost'][0]
        dv_segs = v_segs - v_lamost
        dvmax0 = np.max(dv_segs)
        dvmin0 = np.min(dv_segs)
        dvmax = max(dvmax, dvmax0)
        dvmin = min(dvmin, dvmin0)
    # NOTE to avoid borderline issue, we add a little bit to dvmin and dvmax
    dvmin = max(dvmin, -20)
    dvmax = min(dvmax, 20)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        vs = v_dict[obsid_this]
        v_lamost = result_line['v_lamost'][0]
        dvs = vs - v_lamost
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=dvmin, vmax=dvmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(dvs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_vseg_20.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # check velocity segments v.s. fiber_id with color v - v_lamost
    # set v range to [-50, 50]
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    dvmin, dvmax = 0, 0
    for obs_id, v_segs in v_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        v_lamost = result_line['v_lamost'][0]
        dv_segs = v_segs - v_lamost
        dvmax0 = np.max(dv_segs)
        dvmin0 = np.min(dv_segs)
        dvmax = max(dvmax, dvmax0)
        dvmin = min(dvmin, dvmin0)
    # NOTE to avoid borderline issue, we add a little bit to dvmin and dvmax
    dvmin = max(dvmin, -50)
    dvmax = min(dvmax, 50)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        vs = v_dict[obsid_this]
        v_lamost = result_line['v_lamost'][0]
        dvs = vs - v_lamost
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=dvmin, vmax=dvmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(dvs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_vseg_50.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # check velocity segments v.s. fiber_id with color v - v_lamost
    # set v range to [-200, 200]
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    dvmin, dvmax = 0, 0
    for obs_id, v_segs in v_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        v_lamost = result_line['v_lamost'][0]
        dv_segs = v_segs - v_lamost
        dvmax0 = np.max(dv_segs)
        dvmin0 = np.min(dv_segs)
        dvmax = max(dvmax, dvmax0)
        dvmin = min(dvmin, dvmin0)
    # NOTE to avoid borderline issue, we add a little bit to dvmin and dvmax
    dvmin = max(dvmin, -200)
    dvmax = min(dvmax, 200)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        vs = v_dict[obsid_this]
        v_lamost = result_line['v_lamost'][0]
        dvs = vs - v_lamost
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=dvmin, vmax=dvmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(dvs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_vseg_200.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # check velocity segments v.s. fiber_id with color v - v_lamost
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    dvmin, dvmax = 0, 0
    for obs_id, v_segs in v_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        v_lamost = result_line['v_lamost'][0]
        dv_segs = v_segs - v_lamost
        dvmax0 = np.max(dv_segs)
        dvmin0 = np.min(dv_segs)
        dvmax = max(dvmax, dvmax0)
        dvmin = min(dvmin, dvmin0)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        vs = v_dict[obsid_this]
        v_lamost = result_line['v_lamost'][0]
        dvs = vs - v_lamost
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=dvmin, vmax=dvmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(dvs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_vseg.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()





    # check velocity segments v.s. fiber_id with color correlation
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    corrmin, corrmax = 0, 0
    for obs_id, c_segs in corr_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        corrmax0 = np.max(c_segs)
        corrmin0 = np.min(c_segs)
        corrmax = max(corrmax, corrmax0)
        corrmin = min(corrmin, corrmin0)
    # NOTE to avoid borderline issue, we add a little bit to corrmin and corrmax
    corrmin = max(corrmin, 0.9)
    corrmax = min(corrmax, 1.0)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        corrs = corr_dict[obsid_this]
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=corrmin, vmax=corrmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(corrs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_corr_09.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # check velocity segments v.s. fiber_id with color correlation
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    corrmin, corrmax = 0, 0
    for obs_id, c_segs in corr_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        corrmax0 = np.max(c_segs)
        corrmin0 = np.min(c_segs)
        corrmax = max(corrmax, corrmax0)
        corrmin = min(corrmin, corrmin0)
    # NOTE to avoid borderline issue, we add a little bit to corrmin and corrmax
    corrmin = max(corrmin, 0.5)
    corrmax = min(corrmax, 1.0)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        corrs = corr_dict[obsid_this]
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=corrmin, vmax=corrmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(corrs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_corr_05.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()

    # check velocity segments v.s. fiber_id with color correlation
    fiberid_arr = res_arr['fiberid']
    v_seg_arr = np.arange(4000, 9000, 500)
    n_seg = len(v_seg_arr)
    fiberid_all_arr = np.arange(250, dtype=int) + 1
    minfid, maxfid = np.min(fiberid_all_arr), np.max(fiberid_all_arr)
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([4000, 9000])
    ax.set_ylim([minfid-1, maxfid+1])
    corrmin, corrmax = 0, 0
    for obs_id, c_segs in corr_dict.items():
        result_line = res_arr[res_arr['obsid'] == obs_id]
        corrmax0 = np.max(c_segs)
        corrmin0 = np.min(c_segs)
        corrmax = max(corrmax, corrmax0)
        corrmin = min(corrmin, corrmin0)
    for fiberid in sorted(fiberid_arr):
        result_line = res_arr[res_arr['fiberid'] == fiberid]
        obsid_this = result_line['obsid'][0]
        corrs = corr_dict[obsid_this]
        # set colormap
        cmap = plt.get_cmap('jet')
        cmap_norm = Normalize(vmin=corrmin, vmax=corrmax)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap)
        for i in range(n_seg):
            line_st = [v_seg_arr[i], v_seg_arr[i] + 500]
            line_ed = [fiberid, fiberid]
            c_val = cmap_mapper.to_rgba(corrs[i])
            ax.plot(line_st, line_ed, color=c_val, linestyle='-', linewidth=2.0)
    ax.set_xlabel('wavelength (A)')
    ax.set_ylabel('fiber_id')
    # make colorbar thinner
    plt.colorbar(cmap_mapper, ax=ax, fraction=0.06, pad=0.0, aspect=50)
    plt.tight_layout()
    fig_save_path = os.path.join(work_dir, 'v_fiberid_corr.png')
    fig.savefig(fig_save_path, dpi=300)
    plt.close()