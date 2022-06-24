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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wlm, fluxm, '-', color='grey', alpha=0.5, label='Model')
    ctr = 0
    for wl, flux in zip(wl_segs, flux_segs):
        ctr += 1
        if ctr == 1:
            lineart = ax.plot(wl, flux, alpha=0.8, label='Target')
        else:
            lineart = ax.plot(wl, flux, alpha=0.8)
        # get the color of the line
        color = lineart[0].get_color()
        # plot shadow rectangle
        ax.fill_between(wl, flux, color=color, alpha=0.1)
    ax.set_xlabel('Wavelength [$\AA$]')
    ax.set_ylabel('Flux')
    if v_labels is not None:
        for i, v in enumerate(v_labels):
            xpos = (wl_segs[i][0] + wl_segs[i][-1]) / 2
            ax.text(xpos, 0.1, v, ha='center', va='top')
    if xlim is not None:
        ax.set_xlim(xlim)
    # get ymax of the axes
    ymax = ax.get_ylim()[1]
    # set ymin to 0
    ax.set_ylim([0, ymax])
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close(fig)


def plot_corrs(v_shifts, corrs, v_shift_bests, corr_bests, labels, save_to=None):
    nrows = len(corrs)
    if nrows != len(v_shift_bests):
        raise ValueError('Number of v_shift_bests must match number of corrs')
    if nrows != len(labels):
        raise ValueError('Number of labels must match number of corrs')
    fig = plt.figure(figsize=(5, 10))
    gs = fig.add_gridspec(nrows, hspace=0)
    axs = gs.subplots(sharex=True)
    for i, ax in enumerate(axs):
        corr_seg = corrs[i]
        label_seg = labels[i]
        v_shift_best = v_shift_bests[i]
        corr_best = corr_bests[i]
        ax.plot(v_shifts, corr_seg, 'k-', linewidth=0.5, label=label_seg)
        ax.plot(v_shift_best, corr_best, 'ro', markersize=5, label=f'v={v_shift_best:.2f}')
        # plot vectral line
        ax.axvline(x=v_shift_best, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.legend(loc='lower right')
        if i == nrows - 1:
            ax.set_xlabel('v [km/s]')
        ax.label_outer()
    plt.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=300)
    else:
        plt.show()
    plt.close(fig)
