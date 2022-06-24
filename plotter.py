import matplotlib.pyplot as plt


def plot_spectrum(wl, flux, xlim=None, title=None, save_to=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wl, flux, 'k-', linewidth=0.5)
    ax.set_xlabel('Wavelength [$\AA$]')
    ax.set_ylabel('Flux')
    if xlim is not None:
        ax.set_xlim(xlim)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_shifted_spectrum_segs(wl_segs, flux_segs, wlm, fluxm, v_labels=None, xlim=None, title=None, save_to=None):
    fig, ax_model = plt.subplots(figsize=(10, 5))
    ax_model.plot(wlm, fluxm, '-', color='grey', alpha=0.5, label='Model', linewidth=0.9)
    if xlim is not None:
        ax_model.set_xlim(xlim)
    ymax_model = ax_model.get_ylim()[1]
    ax_model.set_ylim([0, ymax_model])
    ax_model.set_xlabel('Wavelength [$\AA$]')
    ax_model.set_ylabel('Model Flux')
    ax_model.legend(loc='upper left')

    ax_spec = ax_model.twinx()
    ctr = 0
    for wl, flux in zip(wl_segs, flux_segs):
        ctr += 1
        if ctr == 1:
            lineart = ax_spec.plot(wl, flux, alpha=0.8, label='Target', linewidth=0.9)
        else:
            lineart = ax_spec.plot(wl, flux, alpha=0.8, linewidth=0.8)
        # get the color of the line
        color = lineart[0].get_color()
        # plot shadow rectangle
        ax_spec.fill_between(wl, flux, color=color, alpha=0.1)
    ax_spec.set_ylabel('Target Flux')
    ymax_spec = ax_spec.get_ylim()[1]
    # set ymin to 0
    ax_spec.set_ylim([0, ymax_spec])
    if v_labels is not None:
        for i, v in enumerate(v_labels):
            xpos = (wl_segs[i][0] + wl_segs[i][-1]) / 2
            ax_spec.text(xpos, ymax_spec*0.01, v, ha='center', va='bottom', color='black', alpha=0.6)
    if title is not None:
        ax_spec.set_title(title)
    ax_spec.legend(loc='upper right')
    plt.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_corrs(v_shifts, corrs, v_shift_bests, corr_bests, v_lamost, labels, save_to=None):
    nrows = len(corrs)
    vmax = max(v_shifts)
    vmin = min(v_shifts)
    if nrows != len(v_shift_bests):
        raise ValueError('Number of v_shift_bests must match number of corrs')
    if nrows != len(labels):
        raise ValueError('Number of labels must match number of corrs')
    fig = plt.figure(figsize=(6, 10))
    gs = fig.add_gridspec(nrows, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):
        corr_seg = corrs[i]
        label_seg = labels[i]
        v_shift_best = v_shift_bests[i]
        corr_best = corr_bests[i]
        ax.plot(v_shifts, corr_seg, 'k-', linewidth=0.5, label=label_seg)
        ax.plot(v_shift_best, corr_best, 'rx', markersize=5, label=f'v={v_shift_best:.2f}')
        # plot vectral line
        ax.axvline(x=v_shift_best, color='r', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.axvline(x=v_lamost, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
        if i == 0:
            ax.set_ylabel('Correlation')
        if v_lamost < vmin + (vmax - vmin) * 0.618:
            ax.legend(loc='upper right')
        else:
            ax.legend(loc='upper left')
        if i == nrows - 1:
            ax.set_xlabel('v [km/s]')
        ax.label_outer()
    fig.suptitle(f'v_lamost = {v_lamost:.2f} km/s')
    plt.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close(fig)
