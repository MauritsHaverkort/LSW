import numpy as np
import xarray as xr
import hvplot.xarray  # noqa: F401
import panel as pn

pn.extension("bokeh")


def plot_nd_heatmap_with_sliders(data, dim_names=None):
    """
    Plot an N-D array as a heatmap over the first two dimensions,
    with sliders to select indices along all remaining dimensions.

    Parameters
    ----------
    data : np.ndarray
        N-dimensional array. Must have ndim >= 2.
    dim_names : sequence of str, optional
        Names for each dimension. If None, uses ['dim0', 'dim1', ...].

    Returns
    -------
    layout : panel.Row
        Panel layout with sliders and interactive heatmap.
    """
    data = np.asarray(np.real(data))
    if data.ndim < 2:
        raise ValueError("data must have at least 2 dimensions")

    ndim = data.ndim

    # Dimension names
    if dim_names is None:
        dim_names = [f"dim{i}" for i in range(ndim)]
    if len(dim_names) != ndim:
        raise ValueError("len(dim_names) must equal data.ndim")

    # First two dims are plotted
    xdim, ydim = dim_names[0], dim_names[1]
    extra_dims = dim_names[2:]

    # Build xarray DataArray with simple index coords
    coords = {name: np.arange(n) for name, n in zip(dim_names, data.shape)}
    da = xr.DataArray(data, dims=dim_names, coords=coords, name="I")

    # Create a slider for each "extra" dimension
    sliders = {}
    for dim in extra_dims:
        size = da.sizes[dim]
        sliders[dim] = pn.widgets.IntSlider(
            name=dim, start=0, end=size - 1, value=size // 2
        )

    # Callback: slice with current slider values and return heatmap
    def view(**indexers):
        sl = da.isel(indexers) if indexers else da
        return sl.hvplot.image(x=xdim, y=ydim, frame_width=700, frame_height=500)

    # Bind sliders to the view function
    bound_view = pn.bind(view, **{dim: slider for dim, slider in sliders.items()})

    controls = pn.WidgetBox(*sliders.values(), width=250)
    return pn.Row(controls, bound_view, sizing_mode="stretch_both")

def plot_3D_grid_slider(data_3d, dim_names=("x", "y", "z")):
    xdim, ydim, zdim = dim_names
    nx, ny, nz = data_3d.shape

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    da = xr.DataArray(data_3d, dims=[xdim, ydim, zdim],
                      coords={xdim: x, ydim: y, zdim: z})

    # x_sel = pn.widgets.IntSlider(name=xdim, start=0, end=nx-1, value=nx//2)
    # y_sel = pn.widgets.IntSlider(name=ydim, start=0, end=ny-1, value=ny//2)
    z_sel = pn.widgets.IntSlider(name=zdim, start=0, end=nz-1, value=nz//2)

    @pn.depends(z=z_sel)
    def view(z):
        return da.isel({zdim: z}).hvplot.image()

    return pn.Row(pn.Column(z_sel), view)

def plot_rixs_spectrum(spectrum, plot_type = 1):
    # Suppose: data shape R p v w a b c
    spectrum = np.real(spectrum).astype(np.float32)
    R, p, v, w, a, b, c = spectrum.shape

    da = xr.DataArray(
        spectrum,
        dims=["R", "p", "v", "w", "a", "b", "c"],
        coords={
            "R": np.arange(R),
            "p": np.arange(p),
            "v": np.arange(v),
            "w": np.arange(w),
            "a": np.arange(a),
            "b": np.arange(b),
            "c": np.arange(c),
        },
        name="I",
    )

    # sliders
    R_sel = pn.widgets.IntSlider(name="R index", start=0, end=R - 1, value=0)
    p_sel = pn.widgets.IntSlider(name="p index", start=0, end=p - 1, value=0)
    a_sel = pn.widgets.IntSlider(name="a", start=0, end=a - 1, value=0)
    b_sel = pn.widgets.IntSlider(name="b", start=0, end=b - 1, value=0)
    c_sel = pn.widgets.IntSlider(name="c", start=0, end=c - 1, value=0)

    # checkboxes: sum over R / p
    sum_R = pn.widgets.Checkbox(name="Sum over R", value=False)
    sum_p = pn.widgets.Checkbox(name="Sum over p incoming sigma", value=False)

    @pn.depends(
        R=R_sel, p=p_sel,
        a=a_sel, b=b_sel, c=c_sel,
        sum_R=sum_R, sum_p=sum_p,
    )
    def view_2d(R, p, a, b, c, sum_R, sum_p):
        sl = da

        # handle R dimension
        if sum_R:
            sl = sl.sum(dim="R")
        else:
            sl = sl.sel(R=R)

        # handle p dimension
        if sum_p:
            sl = sl.sel(p=[2, 3]).sum(dim="p")
        else:
            sl = sl.sel(p=p)

        # fix remaining indices
        sl = sl.sel(a=a, b=b, c=c)
        match plot_type:
            case 0:
                title = f"R = {'sum' if sum_R else R}, p = {'sum' if sum_p else p}, a = {a}, b = {b}, c = {c}"
                return sl.hvplot.image(
                    x="w", y="v",
                    aspect="equal",
                    frame_width=600,
                    frame_height=600,
                    title=title,
                )
            case 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                offset_plot_from_slice(sl, ax=ax)
                return fig

    controls = pn.WidgetBox(
        "# RIXS spectrum controls",
        sum_R, R_sel,
        sum_p, p_sel,
        a_sel, b_sel, c_sel,
        width=250,
    )

    return pn.Row(controls, view_2d, sizing_mode="stretch_both")

import matplotlib.pyplot as plt

def offset_plot_from_slice(sl, ax=None, scale=None, cmap='viridis'):
    """
    sl : xarray.DataArray with dims ('v','w')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    v_vals = sl.coords['v'].values
    w_vals = sl.coords['w'].values
    data   = sl.values          # shape (Nv, Nw)

    Nv, Nw = data.shape

    # choose vertical offset
    if scale is None:
        scale = 1.1 * np.nanmax(np.abs(data))

    # colormap for curves
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, Nv))

    for i, v in enumerate(v_vals):
        y = data[i] + i * scale          # vertical offset
        ax.plot(w_vals, y, color=colors[i], lw=0.8)

    # nice y-ticks at each offset, labelled by v
    ax.set_yticks(scale * np.arange(Nv))
    ax.set_yticklabels([f"{v:g}" for v in v_vals])

    ax.set_xlabel("w")
    ax.set_ylabel("v (offset)")
    ax.margins(x=0)

    return ax
