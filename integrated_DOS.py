import numpy as np


def _convolute_grids_fft(grid1, grid2, dimension=3):
    assert grid1.shape[1:1 + dimension] == grid2.shape[1:1 + dimension]

    out_w = 2 * grid1.shape[0]
    axes = tuple(range(1, 1 + dimension))

    grid1_hat = np.fft.fftn(grid1, axes=axes)
    grid2_hat = np.fft.fftn(grid2, axes=axes)
    conv_w = np.fft.ifft(
        np.fft.fft(grid1_hat, n=out_w, axis=0)
        * np.fft.fft(grid2_hat, n=out_w, axis=0),
        n=out_w,
        axis=0,
    )
    conv_spatial = np.fft.ifftn(conv_w, axes=axes)
    return conv_spatial.real if np.isrealobj(grid1) and np.isrealobj(grid2) else conv_spatial


def one_magnon_dos_grid(energy_dependent_part, tau_index=0, mode_indices=None):
    """Sum the spectral weight over the physical magnon branches."""
    spectral_weight = np.asarray(energy_dependent_part)
    if spectral_weight.ndim == 6:
        spectral_weight = spectral_weight[tau_index]

    assert spectral_weight.ndim == 5, (
        "energy_dependent_part must have shape "
        "(tau, W, Kx, Ky, Kz, modes) or (W, Kx, Ky, Kz, modes)"
    )

    if mode_indices is None:
        amount_modes = spectral_weight.shape[-1]
        mode_indices = np.arange(amount_modes // 2, amount_modes)

    return np.sum(spectral_weight[..., mode_indices], axis=-1)


def integrate_DOS(energy_dependent_part, dimension=3, tau_index=0, mode_indices=None):
    dos_grid = one_magnon_dos_grid(
        energy_dependent_part,
        tau_index=tau_index,
        mode_indices=mode_indices,
    )
    return _convolute_grids_fft(dos_grid, dos_grid, dimension=dimension)


def integrated_dos_w_values(w_values, convolution_degree=2):
    """Calculate the w values for the integrated DOS, given the w values of the one-magnon DOS."""
    w_values = np.asarray(w_values)
    if w_values.size < 2:
        return np.arange(convolution_degree * w_values.size + 1)

    step = w_values[1] - w_values[0]
    start = convolution_degree * w_values[0]
    return start + step * np.arange(convolution_degree * w_values.size + 1)
