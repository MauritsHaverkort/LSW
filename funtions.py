import io
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from typing import List
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib import gridspec
# Removed the top-level import of Config to avoid circular import
import json 
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from Configurations import Config
from scipy.signal import fftconvolve
import scipy as scipy
from scipy.ndimage import gaussian_filter1d
from funtions import np
import sympy as sp


#plot fonts control
import matplotlib as mpl
fontsize = 18 #18 for presentations, 10 for thesis full size, 18 (?) for thesis half size
mpl.rcParams.update({
    "font.size": fontsize,          # base font
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "figure.titlesize": fontsize,
})

def turn_comp_info_into_str(comp_info):
    result=""
    for key, value in comp_info.items():
        result += f"{key}{value}_"
    return result
def array_to_str(array):
    result=""
    for i in range(len(array)):
        result+=str(array[i])
        if i!=len(array)-1:
            result+=""
    return result

import numpy as np

import numpy as np

########################################################################################################################
########################################################################################################################
################################                          DEBUGGING                               ######################
########################################################################################################################
########################################################################################################################

def debug_check_rixs_coeff_array(
    RIXSMatrix,
    rixs_coeff_arr_numerical_4,
    S_op_list,
    prefactor_list,
    config,
    tol=1e-12,
    max_print=20,
):
    """
    Verifies that for each (i,j): sum_{a,b,sigma} coeff[i,j,a,b,sigma] * prefactor[sigma] * Sa*Sb
    reproduces the symbolic RIXSMatrix[i,j] (after rewriting squares).
    Also prints the largest mismatches.
    """
    
    def expand_squares_to_products(expr):
        return expr.replace(
            lambda x: isinstance(x, sp.Pow) and x.exp == 2,
            lambda x: x.base * x.base
        )
    
    mismatches = []
    for i in range(3):
        for j in range(3):
            Rij = RIXSMatrix[i, j]
            Rij_cmp = expand_squares_to_products(sp.expand(Rij))

            # reconstruct
            Rij_reconstruct = 0
            for a in range(config.amount_spatial_S_operator):
                for b in range(config.amount_spatial_S_operator):
                    # same identity convention as your extraction
                    if config.amount_spatial_S_operator == 4 and (a != 3 and b == 3):
                        continue

                    SaSb = S_op_list[a] * S_op_list[b]
                    for sigma, p in enumerate(prefactor_list):
                        c = rixs_coeff_arr_numerical_4[i, j, a, b, sigma]
                        if c != 0:
                            Rij_reconstruct += sp.nsimplify(c) * p * SaSb

            Rij_reconstruct = sp.expand(Rij_reconstruct)

            diff = sp.expand(Rij_cmp - Rij_reconstruct)

            if diff != 0:
                mismatches.append((i, j, diff))
                
            print(f"\nRIXSMatrix[{i},{j}] reconstructed:")
            sp.pprint(Rij_reconstruct)
    
    if not mismatches:
        print("OK: coefficient array reconstructs all RIXSMatrix elements (under square->product rewrite).")
        return True

    print(f"FAIL: {len(mismatches)} / 9 matrix elements mismatch.")
    for k, (i, j, diff) in enumerate(mismatches[:max_print]):
        print(f"\nMismatch (i={i}, j={j}):")
        # print only short form
        print(diff)

    return False


def compare_tensors_new(a, b, rtol=1e-6, atol=1e-12, zero_tol=1e-12, threshold=1e-8):
    a = np.asarray(a); b = np.asarray(b)
    a[np.abs(a) < threshold] = 0
    b[np.abs(b) < threshold] = 0
    
    assert a.shape == b.shape, "Shapes must match"

    # Valid entries: finite and at least one not ~0
    finite = np.isfinite(a) & np.isfinite(b)
    nontrivial = (np.abs(a) > zero_tol) | (np.abs(b) > zero_tol)
    M = finite & nontrivial
    if not np.any(M):
        print("No valid elements to compare."); return

    aM, bM = a[M], b[M]

    # --- 1) Absolute / relative errors ---
    abs_err = np.abs(aM - bM)
    rel_err_asym = abs_err / np.maximum(np.abs(bM), zero_tol)        # asymmetric
    rel_err_sym  = 2*abs_err / (np.abs(aM) + np.abs(bM) + 1e-300)    # symmetric, safe

    L_inf = abs_err.max()
    fro_rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)
    print(f"L_inf max |a-b|: {L_inf:.3e}")
    print(f"Frobenius relative error ||a-b||/max(||a||,||b||): {fro_rel:.3e}")
    print(f"Median symmetric relative error 2|a-b|/(|a|+|b|): {np.median(rel_err_sym):.3e}")
    print(f"95th pct symmetric relative error: {np.percentile(rel_err_sym,95):.3e}")

    # --- 2) Ratio statistics (elementwise, where b ≠ 0) ---
    nz = np.abs(bM) > zero_tol
    if np.any(nz):
        r = aM[nz] / bM[nz]
        print(f"Ratio |a/b|: median={np.median(np.abs(r)):.3e}, IQR=({np.percentile(np.abs(r),25):.3e},{np.percentile(np.abs(r),75):.3e})")
        # phase difference of a vs b (in radians)
        dphi = np.angle(r)
        # wrap to [-pi,pi]
        print(f"Phase diff angle(a)−angle(b): median={np.median(dphi):.3e} rad, MAD={np.median(np.abs(dphi-np.median(dphi))):.3e}")

    # --- 3) Best complex scalar factor α (least squares) ---
    # α* = argmin_α ||a - α b||_F → α = <b,a>/<b,b>, using conjugate inner product
    denom = np.vdot(bM, bM)
    if np.abs(denom) > 0:
        alpha = np.vdot(bM, aM) / denom
        scaled_err = np.linalg.norm(a - alpha*b) / max(np.linalg.norm(a), 1e-300)
        print(f"Best scale α: |α|={np.abs(alpha):.6g}, arg(α)={np.angle(alpha):.3e} rad")
        print(f"Relative error after scaling ||a-αb||/||a||: {scaled_err:.3e}")

    # --- 4) Magnitude-only and correlation ---
    mag_rmse = np.sqrt(np.mean((np.abs(aM)-np.abs(bM))**2))
    # Complex correlation / coherence in [0,1]
    coh = np.abs(np.vdot(aM, bM)) / (np.linalg.norm(aM)*np.linalg.norm(bM) + 1e-300)
    print(f"Magnitude RMSE: {mag_rmse:.3e}")
    print(f"Complex coherence |<a,b>|/(||a|| ||b||): {coh:.6f}")

    # --- 5) Quick pass/fail with tolerances (predefined behavior) ---
    print(f"np.allclose(a,b, rtol={rtol}, atol={atol}) → {np.allclose(a, b, rtol=rtol, atol=atol)}")

def compare_tensors(a,b):
    # Mask out values where both a and b are close to zero
    mask = (np.abs(a) > 1e-10) | (np.abs(b) > 1e-10)
    a_masked = a[mask]
    b_masked = b[mask]

    relative = np.abs(a / (b + 1e-10))
    print(f"max, mean, median, std relative np.abs(a/(b+1e-10)): {np.max(relative)}, {np.mean(relative)}, {np.median(relative)}, {np.std(relative)}")
    difference = np.abs(a - b)
    print(f"max, mean, median, std distance np.abs(a-b): {np.max(difference)}, {np.mean(difference)}, {np.median(difference)}, {np.std(difference)}")

    # Masked comparison
    relative_masked = np.abs(a_masked / (b_masked + 1e-10))
    difference_masked = np.abs(a_masked - b_masked)
    print(f"[Masked] max, mean, median, std relative np.abs(a/(b+1e-10)): {np.max(relative_masked)}, {np.mean(relative_masked)}, {np.median(relative_masked)}, {np.std(relative_masked)}")
    print(f"[Masked] max, mean, median, std distance np.abs(a-b): {np.max(difference_masked)}, {np.mean(difference_masked)}, {np.median(difference_masked)}, {np.std(difference_masked)}")

    a_rounded = np.round(a, decimals=5)
    b_rounded = np.round(b, decimals=5)
    mask = np.abs(a_rounded) > 1e-12  # Exclude values where a is (close to) zero
    relative_difference_masked = np.abs((a_rounded[mask] - b_rounded[mask])) / np.abs(a_rounded[mask])
    if not np.any(mask):
        print("No valid elements to compare after rounding.")
        return
    print(f"[MASKED] max, mean, median, std distance np.abs((a-b))/abs(a)): {np.max(relative_difference_masked)}, {np.mean(relative_difference_masked)}, {np.median(relative_difference_masked)}, {np.std(relative_difference_masked)}")


def is_hermitian(A, tol=1e-10):
        return np.allclose(A, A.conj().T, atol=tol)

def is_positive_definite(H, tol=1e-12, verbose=True):
    """
    Test whether a Hermitian matrix H is positive definite.
    
    Parameters
    ----------
    H : (N,N) array_like
        Complex or real Hermitian matrix.
    tol : float
        Numerical tolerance for positive eigenvalues.
    verbose : bool
        Print diagnostic messages.
        
    Returns
    -------
    bool : True if H > 0 (all eigenvalues > tol)
    """
    # Check Hermiticity first
    if not np.allclose(H, H.conj().T, atol=tol):
        if verbose:
            print("Matrix is not Hermitian (H ≠ H†)")
        return False

    # Try Cholesky factorization (fastest)
    try:
        np.linalg.cholesky(H)
        if verbose:
            print("Matrix is positive definite (Cholesky succeeded).")
        return True
    except np.linalg.LinAlgError:
        # If Cholesky fails, check eigenvalues for diagnostic info
        eigvals = np.linalg.eigvalsh(H)
        if verbose:
            print("Eigenvalues:", eigvals)
        return np.all(eigvals > tol)

def get_total_size(array: np.ndarray) -> int:
    return array.size * array.itemsize

def inspect_npz_file_size(filename):
    import zipfile
    
    print(f"File: {filename}")
    print(f"Total compressed file size: {os.path.getsize(filename)/1024**2:.3f} MB\n")

    # Load npz normally
    data = np.load(filename, allow_pickle=True)

    # Read zip archive for compressed sizes
    with zipfile.ZipFile(filename, "r") as z:
        zipinfo = {info.filename: info for info in z.infolist()}

    print(f"{'Key':30} {'Uncompressed (MB)':>18} {'Compressed (MB)':>18} {'Shape / Type'}")
    print("-" * 90)

    total_uncompressed = 0
    total_compressed = 0

    for key in data.files:
        arr = data[key]

        uncompressed = arr.nbytes
        compressed = zipinfo[key + ".npy"].compress_size  # npz stores entries as individual .npy files

        total_uncompressed += uncompressed
        total_compressed += compressed

        print(f"{key:30} {uncompressed/1024**2:18.3f} {compressed/1024**2:18.3f}  {arr.shape} {arr.dtype}")

    print("-" * 90)
    print(f"{'TOTAL':30} {total_uncompressed/1024**2:18.3f} {total_compressed/1024**2:18.3f}\n")

    return total_uncompressed, total_compressed


def analyze_matrix(matrix):
    # Convert to NumPy array for convenience
    arr = np.array(matrix, dtype=object)  # object dtype allows for mixed types (e.g., None)

    print("=== Matrix Analysis ===")

    # Check for None
    has_none = np.any(arr == None)  # noqa: E711
    print(f"Contains None values: {has_none}")

    # Remove None for numeric analysis
    arr_no_none = arr[arr != None]  # noqa: E711

    # Convert to complex for safety
    arr_complex = np.array(arr_no_none, dtype=np.complex128)

    # Check for imaginary components
    imag_part = np.imag(arr_complex)
    has_imag = np.any(imag_part != 0)
    max_imag = np.max(np.abs(imag_part)) if has_imag else 0
    print(f"Contains imaginary values: {has_imag}")
    if has_imag:
        print(f"Largest imaginary component: {max_imag}")

    # Check for NaNs
    has_nan = np.any(np.isnan(arr_complex))
    print(f"Contains NaN values: {has_nan}")

    # Check for negative real values
    real_part = np.real(arr_complex)
    has_negative = np.any(real_part < 0)
    print(f"Contains negative values: {has_negative}")

    # Filter out NaNs for stats
    numeric_valid = real_part[~np.isnan(real_part)]

    if numeric_valid.size > 0:
        min_val = np.min(numeric_valid)
        max_val = np.max(numeric_valid)
        mean_val = np.mean(numeric_valid)
        print(f"Min value: {min_val}")
        print(f"Max value: {max_val}")
        print(f"Average (mean): {mean_val}")
    else:
        print("No valid numeric values for min/max/mean.")

    print("========================\n")


def return_grid_samples_of_tensors(tensor, number_axis_to_exclude, reduction):
    # result = np.zeros((tensor.shape[:number_axis_to_exclude] + (tuple(np.ceil(np.array(tensor.shape[number_axis_to_exclude:]) / reduction).astype(int),))), dtype=np.complex128)
    """returns a sampled version of the tensor where every 'reduction' value is taken --> all axis that are included are collapsed into one single axis
    should be used to compare large tensors for symmetries
    """
    old_shape = tensor.shape[number_axis_to_exclude:]
    axis_indices = []
    for i,s in enumerate(old_shape):
        axis_indices.append(np.arange(0,s,reduction))
    mesh = np.meshgrid(*axis_indices, indexing='ij')
    combos = np.stack(mesh, axis=-1).reshape(-1, len(axis_indices))

    result = np.zeros(tensor.shape[:number_axis_to_exclude] + (combos.shape[0],), dtype=tensor.dtype)
    for spatial_index in np.ndindex(tensor.shape[:number_axis_to_exclude]):
        for new_index,old_idx in enumerate(combos):
            result[spatial_index + (new_index,)] = tensor[spatial_index + tuple(old_idx)]
    return result

def plot_tensor_at_certain_axis(tensor, axis_to_plot_individ, logscale=False):
    plt.figure("Debug tensor plotting")
    plt.title(f"shape of tensor {tensor.shape}, plotting of first {axis_to_plot_individ} axis")
    for idx in np.ndindex(tensor.shape[:axis_to_plot_individ]):
        
        plt.plot(tensor[idx].flatten(), label=f"idx {idx}")
    if logscale:
        plt.yscale("log")
    plt.legend()
    plt.show()

def give_symmetries_of_tensors(tensor, number_axis_to_include, threshold=None, 
                               use_median_relative_difference=True, 
                               use_allclose = True,
                               threshold_multiplier = 1e-2, 
                               rounding_decimals = 2, 
                               return_conjugated_symmetries=False, 
                               give_mean_abs_value=False, 
                               compare_only_a_sample = True):
    """given a tensor this returns a list of lists of indices that are the same
    
    when using median relative difference: only the median difference is looked at to be stable against single numerical outliners dankgerous if one wants to check for exyct same
    """
    assert number_axis_to_include != None, "number_axis_to_include must be provided"
    if np.any(tensor.shape[number_axis_to_include:]) > 15:
        reduction = 5
    else:
        reduction = 2
    
    
    if threshold is None:
        threshold = np.mean(np.abs(tensor))*threshold_multiplier
    if compare_only_a_sample:
        rounding_decimals = 10 #when only comparing a sample the smaller differences matter more

    tensor = return_grid_samples_of_tensors(tensor, number_axis_to_include, reduction=reduction) if compare_only_a_sample else tensor

    def is_similar_max_diff(a, b, threshold):
        return np.max(np.abs(a - b)) < threshold
    def conjugated_is_similar(a, b, threshold):
        return is_similar_max_diff(a, np.conj(b), threshold)
    def is_similar_mean_relative_diff(a, b, threshold):
        a_rounded = np.round(a, decimals=rounding_decimals)
        b_rounded = np.round(b, decimals=rounding_decimals)
        mask = np.abs(a_rounded) > 1e-12  # Exclude values where a is (close to) zero
        if not np.any(mask):
            return is_similar_max_diff(a,b,threshold)  # No valid elements to compare
        return np.mean(np.abs((a_rounded[mask] - b_rounded[mask]) / a_rounded[mask])) < threshold
    def is_similar_median_relative_diff(a, b, threshold):
        a_rounded = np.round(a, decimals=rounding_decimals)
        b_rounded = np.round(b, decimals=rounding_decimals)
        mask = np.abs(a_rounded) > 1e-12  # Exclude values where a is (close to) zero
        
       
        if not np.any(mask):
            return is_similar_max_diff(a,b,threshold)  # No valid elements to compare
        return np.abs(np.median((a_rounded[mask] - b_rounded[mask]) / a_rounded[mask])) < threshold

    symmetries = []
    conj_symmetries = []
    for indices in np.ndindex(tensor.shape[:number_axis_to_include]):
        new_symmetry = True
        #adding the first symmetry batch
        if len(symmetries) == 0:
            symmetries.append([indices])
            conj_symmetries.append([])
            continue
        #check if the current indices belong to an already known symmetry batch
        for sym_idx,sym in enumerate(symmetries):
            if use_allclose:
                if np.allclose(tensor[indices], tensor[sym[0]], ):
                    new_symmetry = False
                    sym.append(indices)
                    break
            else:
                is_similar = is_similar_max_diff(tensor[indices], tensor[sym[0]], threshold)
                if is_similar:
                    new_symmetry = False
                    sym.append(indices)
                    break
                elif conjugated_is_similar(tensor[indices], tensor[sym[0]], threshold) and return_conjugated_symmetries:
                    new_symmetry = False
                    conj_symmetries[sym_idx].append(indices)
                    break
                elif is_similar_median_relative_diff(tensor[indices], tensor[sym[0]], threshold) and use_median_relative_difference:
                    new_symmetry = False
                    sym.append(indices)
                    break
        #if not create a new symmetry batch
        if new_symmetry:
            symmetries.append([indices])
            conj_symmetries.append([])
    

    #construct a tensor of the same shape as the axis to include but just fill it with the corresponding symmetry class index --> for further testing of symmetries
    tensor_of_symm_class_index = np.zeros(tensor.shape[:number_axis_to_include], dtype=int)
    for sym_class_index,sym_class in enumerate(symmetries+conj_symmetries):
        for item in sym_class:
            if isinstance(item, tuple):
                tensor_of_symm_class_index[*item]=sym_class_index
    if give_mean_abs_value: 
        for sym in symmetries:
            sym.insert(0, f"mean abs value: {np.mean(np.abs(tensor[sym[0]]))}")
        for sym_idx,sym in enumerate(conj_symmetries):
            if len(sym)>0:
                sym.insert(0, (f"conjugated indices",sym_idx))
    result = symmetries + conj_symmetries if return_conjugated_symmetries else symmetries
    return result, tensor_of_symm_class_index

def turn_equivalence_list_into_array(symmetry_list, shape_indices):
    symmetry_array = np.zeros((len(symmetry_list), *shape_indices), dtype=np.complex128)
    for i, sym in enumerate(symmetry_list):
        for indices in sym:
            symmetry_array[i, *indices] = 1
    return symmetry_array

def give_symmetries_of_tensors_fast(
    tensor,
    number_axis_to_include,
    *,
    mode="maxdiff",                # "maxdiff" | "median_rel"
    threshold=1e-2,
    rounding_decimals=2,           # used in "median_rel"
    return_conjugated_symmetries=True,
    chunk_rows=None                # set an int to process in chunks if memory is tight
):
    """
    Vectorized grouping of subtensors taken from the first `number_axis_to_include` axes.

    We flatten each subtensor (the remaining axes) to a 1D row, build a canonical
    byte key per row (and its conjugate), and group by identical keys via np.unique.
    """
    assert False, "Not in use (not faster than previous implementation)"
    assert number_axis_to_include is not None
    lead_shape = tensor.shape[:number_axis_to_include]
    tail_shape = tensor.shape[number_axis_to_include:]
    n_rows = int(np.prod(lead_shape))
    n_cols = int(np.prod(tail_shape))

    # View as a 2D array: rows = indices over the included axes; columns = flattened subtensor.
    X = tensor.reshape(n_rows, n_cols)

    # --- Build canonical keys -------------------------------------------------
    # Two fast modes:
    # (A) "maxdiff": quantize values by bin size = threshold to approximate max-abs-diff < threshold.
    # (B) "median_rel": normalize each row by its median(abs(.)) to remove scale, then round.
    #     This approximates your "median relative diff" check but is vectorized.

    def _quantize_maxdiff_block(block):
        # Quantize to integer bins so rows within threshold map to same bins.
        # Using np.round(block/threshold) keeps any two entries within < threshold in the same or adjacent bins,
        # which is good for grouping; it’s conservative (slightly stricter) but very fast.
        return np.round(block / threshold).astype(np.int64)

    def _normalize_median_rel_block(block):
        # Row-wise scale by median(abs), ignoring zeros; then round to decimals.
        med = np.median(np.abs(block), axis=1, keepdims=True)
        med[med < 1e-12] = 1.0  # avoid division-by-zero; for all-zero rows we leave as-is
        Y = block / med
        return np.round(Y, decimals=rounding_decimals)

    def _block_to_keys(block):
        """Return (keys, conj_keys) as bytes for this block."""
        if mode == "maxdiff":
            Q = _quantize_maxdiff_block(block)
            QC = _quantize_maxdiff_block(np.conj(block))
        elif mode == "median_rel":
            Q = _normalize_median_rel_block(block)
            QC = _normalize_median_rel_block(np.conj(block))
        else:
            raise ValueError("mode must be 'maxdiff' or 'median_rel'")

        # Ensure deterministic dtype for bytes keys
        if np.iscomplexobj(Q):
            # split complex into two real parts for stable bytes
            Qr = np.asarray(Q.real, dtype=np.float64)
            Qi = np.asarray(Q.imag, dtype=np.float64)
            QC_r = np.asarray(QC.real, dtype=np.float64)
            QC_i = np.asarray(QC.imag, dtype=np.float64)
            key_bytes  = np.hstack([Qr, Qi]).view(np.uint8).reshape(len(Q), -1).copy()
            ckey_bytes = np.hstack([QC_r, QC_i]).view(np.uint8).reshape(len(QC), -1).copy()
        else:
            key_bytes  = np.asarray(Q,  dtype=np.float64).view(np.uint8).reshape(len(Q),  -1).copy()
            ckey_bytes = np.asarray(QC, dtype=np.float64).view(np.uint8).reshape(len(QC), -1).copy()

        # Join each row’s bytes into a single Python bytes object (works well with np.unique on object dtype)
        keys  = key_bytes.view(np.uint8).reshape(len(Q), -1)
        ckeys = ckey_bytes.view(np.uint8).reshape(len(QC), -1)
        keys  = np.array([memoryview(row).tobytes() for row in keys], dtype=object)
        ckeys = np.array([memoryview(row).tobytes() for row in ckeys], dtype=object)
        return keys, ckeys

    # Process entire matrix or in chunks to save memory
    labels = np.empty(n_rows, dtype=np.int64)
    conj_flags = np.zeros(n_rows, dtype=np.bool_)  # True if matched via conjugate

    # We’ll assign class ids with a dict incrementally to avoid storing giant object arrays at once.
    table = {}  # key -> class_id
    next_class = 0

    def _assign(keys, ckeys, row_offset):
        nonlocal next_class
        for i, (k, ck) in enumerate(zip(keys, ckeys)):
            # Canonicalize: pick min(k, ck) so conjugates share classes when desired
            if return_conjugated_symmetries:
                if ck < k:
                    canon = ck
                    is_conj = True
                else:
                    canon = k
                    is_conj = False
            else:
                canon = k
                is_conj = False

            cls = table.get(canon)
            if cls is None:
                cls = next_class
                table[canon] = cls
                next_class += 1

            labels[row_offset + i] = cls
            conj_flags[row_offset + i] = is_conj

    if chunk_rows is None or chunk_rows >= n_rows:
        k, ck = _block_to_keys(X)
        _assign(k, ck, 0)
    else:
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            k, ck = _block_to_keys(X[start:stop])
            _assign(k, ck, start)

    # Build output group lists (indices over the first axes)
    symmetries = [[] for _ in range(int(labels.max()) + 1)]
    conj_symmetries = [[] for _ in range(int(labels.max()) + 1)] if return_conjugated_symmetries else None

    # Map row index -> multi-index
    multi_idx = np.array(np.unravel_index(np.arange(n_rows), lead_shape)).T  # shape (n_rows, ndim)

    for row, cls in enumerate(labels):
        tup = tuple(multi_idx[row])
        if return_conjugated_symmetries and conj_flags[row]:
            conj_symmetries[cls].append(tup)
        else:
            symmetries[cls].append(tup)

    # tensor_of_symm_class_index (same as before)
    tensor_of_symm_class_index = labels.reshape(lead_shape)

    # Optionally add headers like you did (kept minimal to avoid extra passes)
    if return_conjugated_symmetries:
        result = []
        for cls, items in enumerate(symmetries):
            if items:
                mean_abs = float(np.mean(np.abs(tensor[items[0]])))
                result.append([f"mean abs value: {mean_abs}", *items])
        for cls, items in enumerate(conj_symmetries):
            if items:
                result.append([("conjugated indices", cls), *items])
    else:
        result = []
        for cls, items in enumerate(symmetries):
            if items:
                mean_abs = float(np.mean(np.abs(tensor[items[0]])))
                result.append([f"mean abs value: {mean_abs}", *items])

    return result, tensor_of_symm_class_index

def test_symmetries_sqw_4(symmetries, tensor_of_symm_class_index, material):
    #testing if the order does or doesn not play a role for same amount of a and ad operators
    permutation = [0,2,1,3] 
    temp=[]
    for idx in material.return_ad_a_spectra_indices(order_of_operator=4):
        idx_o_2_altered = idx.copy()
        idx_o_2_altered[3] = permutation[idx[3]] 
        idx_o_1_altered = idx.copy()
        idx_o_1_altered[1] = permutation[idx[1]] 
        if tensor_of_symm_class_index[*idx]!=tensor_of_symm_class_index[*idx_o_2_altered]:
            temp.append((idx, idx_o_2_altered))
        if tensor_of_symm_class_index[*idx]!=tensor_of_symm_class_index[*idx_o_1_altered]:
            temp.append((idx, idx_o_1_altered))
        
        #testing if swapping i oi and j oj leads to same symmetry class
        idx_ij_swapped = idx.copy()
        # idx_ij_swapped[0], idx_ij_swapped[2] = idx_ij_swapped[2], idx_ij_swapped[0]
        idx_ij_swapped = [idx[2], idx[3], idx[0], idx[1]]
        if tensor_of_symm_class_index[*idx]!=tensor_of_symm_class_index[*idx_ij_swapped]:
            temp.append((idx, idx_ij_swapped))
        
        
    return temp

def give_highest_indices_of_matrix(matrix, number_of_indices=5):
    n = number_of_indices  # number of indices to return
    flat_indices = np.argsort(-np.abs(matrix.flatten()))
    top_n_indices = np.unravel_index(flat_indices[:n], matrix.shape)
    print(f"Indices of top {n} differences from identity:")
    res = []
    for idx in zip(*top_n_indices):
        res.append(f"Index: {tuple(int(i) for i in idx)}, Value: {matrix[idx]}")
        print(res[-1])
    return res

def get_difference_of(M1,M2):
    result=0
    for i in range(len(M1)):
        for j in range(len(M1[0])):
            result+=abs(M1[i][j]-M2[i][j])
    return result

########################################################################################################################
########################################################################################################################
################################                          KREIN TRANSFORMATIONS                   ######################
########################################################################################################################
########################################################################################################################


def first_order_perturbation_degenerate(dH, E, U, Gamma, deg_tol=1e-10):
    """
    First-order (static) perturbation theory with degeneracies.

    Parameters
    ----------
    H   : (N,N) Hermitian array
    dH  : (N,N) Hermitian array, derivative dH/dλ at the same point
    deg_tol : float
        Two eigenvalues |E_i - E_j| <= deg_tol are treated as degenerate.

    Returns
    -------
    E    : (N,) eigenvalues of H (sorted ascending)
    U    : (N,N) eigenvectors of H, possibly rotated within degenerate blocks
           to diagonalize P dH P (a smooth gauge).
    dE   : (N,) first-order derivatives of eigenvalues
    dU   : (N,N) first-order derivatives of eigenvectors in the chosen gauge
           (columns are d|n>/dλ). Inside each degenerate block we enforce
           parallel-transport: the component within the block is zero.
    """
    dH = np.asarray(dH)
    # Diagonalize H
    #E, U = la.eig(H)  # H U = U diag(E)
    M = dH.shape[0]
    #sorting again the eigenvalues and eigenvectors since now they were multiplied with Gamma
    # idx_sorted_eigvals = np.argsort(-E.real)
    # E = E[idx_sorted_eigvals]
    # U = U[:, idx_sorted_eigvals]
    # dH = dH[np.ix_(idx_sorted_eigvals, idx_sorted_eigvals)]
    
    # Helper: build degenerate blocks from sorted E
    blocks = [[0]]
    
    for i in range(1,M):# first index already in block
        new_block = True
        for b in blocks:
            if abs(E[i] - E[b].mean()) < deg_tol:
                b.append(i)
                new_block = False
                break
        if new_block:
            blocks.append([i])
    blocks=[np.array(b) for b in blocks]
    # Rotate eigenvectors within each block to diagonalize P dH P
    U_rot = U.copy()
    dE = np.empty(M, dtype=np.complex128)
    for blk in blocks:
        U_blk = U_rot[:, blk]                       # (N, g)
        W = U_blk.conj().T @ dH @ U_blk            # (g, g), Hermitian
        wvals, V = la.eig(W) #np.diag(Gamma[blk,blk]) @  diagonalize inside block
        V_herm_conj = np.conj(V.T)
        U_rot[:, blk] = U_blk @ V          # rotate basis in block
        dE[blk] = wvals.real                       # eigenvalue derivatives

    # Precompute projected dH in the rotated basis
    Hp = U_rot.conj().T @ dH @ U_rot               # <m|dH|n> in rotated basis

    # Eigenvector derivatives: only components outside each block
    dU = np.zeros_like(U_rot, dtype=U_rot.dtype)   # columns are d|n>/dλ
    Ecol = E[:, None]
    for blk in blocks:
        # indices outside this degenerate subspace
        out = np.setdiff1d(np.arange(M), blk, assume_unique=True)
        if out.size == 0:
            continue
        # Hp_out_blk: (nout, g) with entries <m|dH|n>, m ∉ blk, n ∈ blk
        Hp_out_blk = Hp[np.ix_(out, blk)]
        # Denominators (E_n - E_m): shape (nout, g) -> we want m rows, n cols
        denom = (E[blk][None, :] - E[out][:, None])
        # Safe: gaps to states outside block should be > deg_tol; if not, it's a larger block
        if np.any(np.abs(denom) <= deg_tol):
            raise ValueError("Detected overlapping/insufficiently separated blocks; "
                             "increase deg_tol or ensure isolated degenerate manifold.")
        K_out_blk = Hp_out_blk / denom             # (nout, g)
        # Contribution to d|n>/dλ from outside states:
        dU[:, blk] += U_rot[:, out] @ K_out_blk    # (N, g)
        # Inside-block component is set to zero (parallel-transport gauge)

    return E, U_rot, dE, dU

def first_order_perturbation_degenerate_paraunitary(dH, E, U, Gamma, deg_tol=1e-10):
    """
    First-order perturbation theory with degeneracies for a paraunitary (pseudo-/para-Hermitian) problem.

    Context
    -------
    We assume the *unperturbed* eigenproblem is for the bosonic/BdG Liouvillian-like operator
        L0 = Gamma @ H0,
    with right eigenpairs (E_n, v_n) already known:
        L0 @ v_n = E_n v_n,
        U = [v_0, v_1, ...]  (columns are right eigenvectors),
    and metric (indefinite) Gamma (Hermitian, involutory in bosonic BdG: Gamma^2 = I).

    A small perturbation of H0 -> H0 + λ dH induces a perturbation of L0:
        L(λ) = Gamma @ H(λ) = L0 + λ * (Gamma @ dH).

    The *left* eigenvectors are the Γ-duals of the right eigenvectors:
        W^dagger = S^{-1} U^dagger Gamma,   where S = U^dagger Gamma U (Gram in Γ-metric).
    With the common bosonic normalization we target S = diag(s_n), s_n = ±1.

    Inputs
    ------
    dH    : (N,N) Hermitian array (the perturbation of H, not multiplied by Gamma).
    E     : (M,) eigenvalues of L0 (real; sorted as given). M is the number of columns in U.
    U     : (N,M) right eigenvectors of L0 (columns).
    Gamma : (N,N) Hermitian indefinite metric (e.g., diag(+1,...,+1,-1,...,-1)).
    deg_tol : float, threshold to group |E_i - E_j| <= deg_tol as degenerate.

    Returns
    -------
    E         : (M,) unchanged eigenvalues (as given in input; we do not resort globally here).
    U_rot     : (N,M) eigenvectors after *intrablock* rotation to diagonalize the projected perturbation
                in the Γ-metric (a smooth gauge inside each degenerate block).
    dE        : (M,) first-order energy corrections.
    dU        : (N,M) first-order eigenvector derivatives in the chosen gauge (d|n>/dλ).
                Inside each degenerate block we impose parallel-transport (no intrablock component).

    Notes
    -----
    • First-order shifts use the left–right biorthogonal pairing in the Γ-metric:
          dE_n = <w_n | (Gamma dH) | v_n> / <w_n|v_n>.
      With the Γ-dual choice above and Gamma^2 = I, this equals
          dE_n = (v_n^dagger dH v_n) / s_n, where s_n = v_n^dagger Gamma v_n = ±1.

    • Degenerate blocks (indices B) are handled by the generalized eigenproblem
          W c = ΔE G c,
      where G = U_B^dagger Gamma U_B  (Gram in Γ-metric)
            W = U_B^dagger dH   U_B   (note: Γ dH sandwiched with Γ-duals collapses to dH).
      We then rotate U within the block by C whose columns solve the problem. We also
      re-G-normalize C so that (C^dagger G C) = G_target (diagonal sign matrix of the block).

    • Off-block eigenvector derivatives use
          |δv_n> = sum_{m∉block} |v_m>  [ <w_m| (Gamma dH) |v_n> / (E_n - E_m) ].
      In matrix form, with S = diag(s), this becomes
          K_{m n} = ( (U^dagger dH U)_{m n} / s_m ) / (E_n - E_m).
      That is: divide the numerator by the sign s_m of the *left* state.

    • We assume the input U is already reasonably Γ-orthogonal (no Jordan blocks).
      We gently re-normalize columns to have v_n^dagger Γ v_n = ±1 (and record s_n).
    """

    dH = np.asarray(dH)
    Gamma = np.asarray(Gamma)
    U = np.asarray(U)
    E = np.asarray(E)

    N, M = U.shape
    assert dH.shape == (N, N)
    assert Gamma.shape == (N, N)
    assert E.shape == (M,)

    # --- 0) Γ-gram and (re)normalization to s_n = ±1 --------------------------
    S = U.conj().T @ Gamma @ U  # (M,M) Gram in Γ-metric; should be ~ diagonal (±1 on diag) if U is paraunitary
    # Extract signs and re-normalize columns so that diag becomes strictly ±1.
    s_raw = np.real(np.diag(S))
    # # Protect against tiny norms (shouldn't happen for proper modes):
    # s_raw[np.isclose(s_raw, 0.0, atol=1e-14)] = 1.0
    scal = 1.0 / np.sqrt(np.abs(s_raw))
    U = U @ np.diag(scal)               # rescale columns
    S = U.conj().T @ Gamma @ U          # refresh Gram
    s = np.sign(np.real(np.diag(S)))    # s_n = ±1
    # Enforce exact diagonal ±1 by a small diagonal correction:
    U = U @ np.diag(1.0 / np.sqrt(np.abs(np.real(np.diag(S)))))  # second pass for stability
    S = U.conj().T @ Gamma @ U
    # At this point: S ≈ diag(s). We take s = sign(diag(S)) as the mode signature.

    # --- 1) Build degenerate blocks from sorted E ------------------------------
    blocks = [[0]]
    for i in range(1, M):
        placed = False
        for b in blocks:
            if abs(E[i] - np.mean(E[b])) <= deg_tol:
                b.append(i)
                placed = True
                break
        if not placed:
            blocks.append([i])
    blocks = [np.array(b, dtype=int) for b in blocks]

    # --- 2) Rotate inside each degenerate block (Γ-generalized diagonalization) -
    U_rot = U.copy()
    dE = np.empty(M, dtype=np.complex128)

    for blk in blocks:
        U_blk = U_rot[:, blk]                 # (N,g)
        # Gram and projected "W" in Γ-metric:
        G = U_blk.conj().T @ Gamma @ U_blk    # (g,g) ~ diag(±1)
        W = U_blk.conj().T @ dH @ U_blk       # (g,g)
        # Solve the generalized eigenproblem: W c = (ΔE) G c
        # (SciPy eig handles generalized problems.)
        wvals, C = la.eig(W, G)

        # G-orthonormalize C so that C^dagger G C = G_target = diag(signs of the original block)
        # First build G_C = C^dagger G C, then whiten it to the block's sign-diagonal.
        G_C = C.conj().T @ G @ C
        # Target signature (take the signs from the original G diagonal):
        s_blk = np.sign(np.real(np.diag(G)))
        G_target = np.diag(s_blk)
        # Map G_C -> G_target via right-side transform R so that C' = C R and (C')^dagger G C' = G_target.
        # One way: compute H = G^{-1/2} G_C G^{-1/2} in the metric induced by G_target; but here G is close to diag(s).
        # A robust, simple fix is: R = (G_C)^{-1/2} followed by a rephase to match signs -> yields C'^dagger G C' ≈ I.
        # Then we left-multiply by a diagonal Sqrt(G_target) if you want ±1 explicitly. Since G_target^2=I, it's just signs.
        # We'll do: R = invsqrt(G_C), then "sign-correct" columns to enforce the desired s_blk.
        # (la.sqrtm is fine at this small g; for speed, eigh on Hermitian would be nicer.)
        R = la.inv(la.sqrtm(G_C))
        C_fix = C @ R
        # After this, C_fix^dagger G C_fix ≈ I. To imprint ±1, we just rescale columns by sqrt(|s_blk|)=1 and flip phases if needed.
        # (s_blk are ±1 already; if you want exactly G_target, you could multiply columns with (-1) where s_blk=-1; not necessary for correctness.)

        # Rotate basis in the block:
        U_rot[:, blk] = U_blk @ C_fix
        # First-order energy corrections are the generalized eigenvalues:
        dE[blk] = wvals

    # --- 3) Precompute projected matrices in the rotated basis -----------------
    # NOTE: for all matrix elements <w_m| (Gamma dH) |v_n> we use:
    #        <w_m| = (1/s_m) <v_m| Gamma
    #  =>    <w_m| (Gamma dH) |v_n> = (1/s_m) (v_m^dagger dH v_n)
    # Hence we only need U_rot^dagger dH U_rot and the sign vector s.
    Hp_dH = U_rot.conj().T @ dH @ U_rot         # (M,M)
    S_rot = np.real(np.diag(U_rot.conj().T @ Gamma @ U_rot))  # should still be ±1 per column
    s_rot = np.sign(S_rot)
    srot_inv = 1.0 / s_rot

    # --- 4) Eigenvector derivatives (parallel transport inside block) ----------
    dU = np.zeros_like(U_rot, dtype=U_rot.dtype)
    for blk in blocks:
        out = np.setdiff1d(np.arange(M), blk, assume_unique=True)
        if out.size == 0:
            continue
        # Numerators:  <w_m| (Gamma dH) |v_n> = (Hp_dH[m,n]) / s_m
        numer = (Hp_dH[np.ix_(out, blk)].T * srot_inv[out]).T  # (nout,g)
        # Denominators: (E_n - E_m)
        denom = (E[blk][None, :] - E[out][:, None])
        if np.any(np.abs(denom) <= deg_tol):
            raise ValueError("Detected overlapping blocks (tiny energy gap to an 'outside' state). "
                             "Increase deg_tol or include those states in the degenerate block.")
        K_out_blk = numer / denom
        # Contribution to d|n>/dλ from outside states only:
        dU[:, blk] += U_rot[:, out] @ K_out_blk

    # For completeness, ensure the inside-block (parallel-transport) component is zero:
    # (Already true by construction — we never added intrablock pieces.)

    # Energies E unchanged (these are the unperturbed values). dE are first-order shifts.
    return E, U_rot, dE.real, dU

def path_k_space(k_points, total_steps, a, steps=None):
    xticks_pos=[]
    xticks_labels=[]
    for i in range(len(k_points)):
        k_points[i]=k_points[i]*np.pi/a
    
    k_vectors=[k_points[0]]
    k_scalars=[0]
    #if steps is not given, it is calculated from the total_steps and the number of k_points
    if steps==None:
        steps=[]
        for i in range(len(k_points)-1):
            steps.append(int(total_steps/(len(k_points)-1)))
    previous_k_point=k_points[0]
    previous_k_scalar=0

    xticks_pos.append(0)
    xticks_labels.append(np.round(k_points[0],2))
    for i in range(1,len(k_points)):
        
        k_dir= k_points[i]-previous_k_point
        
        for k_ in np.linspace(0,1,steps[i-1]+1)[1:]:  # start from 1 to avoid the first point being repeated (if I would take 0 this would give the same ponint as from the previous direction)
            k_vectors.append(k_dir*k_+previous_k_point)
            k_scalars.append(la.norm(k_dir)*k_+previous_k_scalar)
        
        xticks_pos.append(k_scalars[-1])
        xticks_labels.append(np.round(k_points[i],2))
        previous_k_point = k_points[i]
        previous_k_scalar = k_scalars[-1]
    
    return k_scalars,k_vectors, xticks_labels, xticks_pos

########################################################################################################################
########################################################################################################################
################################                          General matrix / array operations       ######################
########################################################################################################################
########################################################################################################################


def extend_array_equal_spacing(arr, target_len: int):
    """
    Extend a 1D array to a target shape by adding equal spacing.
    """
    if arr.ndim != 1:
        raise ValueError("Input array must be 1D")

    arr = np.array(arr)
    # Extend w_values with equal spacing until it matches grid.shape[1]
    if len(arr) < target_len:
        step = arr[1] - arr[0] if len(arr) > 1 else 1
        num_to_add = target_len - len(arr)
        last_value = arr[-1]
        extension = [last_value + step * (i + 1) for i in range(num_to_add)]
        arr = np.concatenate([arr, extension])

    return arr

def orthonormalise(A):
    """
    Orthonormalises the column vectors of matrix A (like MATLAB's orth).
    """
    # Q, R = np.linalg.qr(A, mode='reduced')   # QR decomposition
    # # # remove near-zero columns (if rank deficient)
    # # tol = 1e-12
    # # keep = np.abs(np.diag(R)) > tol
    
    # if Q.shape[1] == A.shape[1]:
    #     return Q
    # else:   # matrix is rank deficient --> no orthonormalisation possible
    #     return A    
    # MATLAB-like orth() using SVD
    Q = scipy.linalg.orth(A)
    return Q

def cut_array_by_one(arr, dimensions_to_exclude=[], cut_also_first_element=False):
    """
    Cuts the last element of each dimension of the array, except if the dimension is already 1. also one can give dimensions to exclude as a list"""
    slices = []
    startindex = 1 if cut_also_first_element else 0
    
    for i,s in enumerate(arr.shape):
        if i in dimensions_to_exclude:
            slices.append(slice(0,s))
        elif s>1:
            slices.append(slice(startindex,s-1))
        else:
            slices.append(slice(0,s))
    return arr[tuple(slices)]

def subtract_grids_with_padding(big, small):
    # Ensure `big` is the larger-by-shape; swap if not
    if any(s > b for s, b in zip(small.shape, big.shape)):
        big, small = small, big  # result will be small->padded to big
    
    out = np.zeros_like(big, dtype=np.result_type(big, small))
    # slice the overlapping region
    sl = tuple(slice(0, s) for s in small.shape)
    out[sl] = small
    return big - out

def make_grid_periodic(grid):
    """Takes a grid of shape (..., X, Y, Z) and makes it periodic by adding the first slice to the end of each dimension."""
    slices = [slice(None)] * grid.ndim
    slices[-3] = slice(0, 1) #slice with length 1 [0:1] --> in the x-dimension only the one element
    grid_x = np.concatenate((grid, grid[tuple(slices)]), axis=-3)
    slices = [slice(None)] * grid_x.ndim
    slices[-2] = slice(0, 1)
    grid_xy =  np.concatenate((grid_x, grid_x[tuple(slices)]), axis=-2)
    slices = [slice(None)] * grid_xy.ndim
    slices[-1] = slice(0, 1)
    grid_xyz =  np.concatenate((grid_xy, grid_xy[tuple(slices)]), axis=-1)
    return grid_xyz

def remove_small_parts(H, threshold=1e-6):
    """
    Remove small parts of the matrix below a certain threshold.
    
    Parameters:
        H (numpy.ndarray): The input matrix.
        threshold (float): The threshold value.
        
    Returns:
        numpy.ndarray: The modified matrix with small parts removed.
    """
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.abs(H[i, j]) < threshold:
                H[i, j] = 0
    return H



def flatten_axes(arr: np.ndarray, axes_to_flatten):
    """
    Flatten the specified axes of a NumPy array into a single axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axes_to_flatten : tuple of int
        Axes to merge into one. Can be in any order and may be negative.

    Returns
    -------
    out : np.ndarray
        Array where selected axes are replaced by a single axis.
        Shape (..., flattened_size, ...) depending on axis positions.
    """

    # Normalize axes (handle negative axes)
    axes = tuple(sorted(ax % arr.ndim for ax in axes_to_flatten))

    # Compute the flattened size
    flat_size = np.prod([arr.shape[ax] for ax in axes])

    # Build list of new axes order: 
    #   1. all axes *not* being flattened (in original order)
    #   2. the axes to flatten (in original order)
    remaining_axes = tuple(ax for ax in range(arr.ndim) if ax not in axes)
    new_order = remaining_axes + axes

    # Move flattened axes to the end
    arr_reordered = arr.transpose(new_order)

    # New shape = remaining dims + (flat_size,)
    new_shape = arr_reordered.shape[:len(remaining_axes)] + (flat_size,)

    return arr_reordered.reshape(new_shape)

    
    def return_angle_between_vectors(v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'"""
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return angle
########################################################################################################################
########################################################################################################################
################################                          PLOTTING                                ######################
########################################################################################################################
########################################################################################################################


def plot_3d_grid_slider(grid, 
                        title="3D Grid Viewer", 
                        xlabel="X", ylabel="Y", intensity="Intensity", 
                        axis_labels=None,  # list of 3 strings
                        useLogNorm=False, range_from_max=1000,
                        xticks_pos=[], xticks_labels=[],
                        k_values_scalar=None, w_values=None):
    """
    Plots a 3D grid using a 2D heatmap and a slider for the third dimension.
    
    Parameters:
        grid: np.ndarray of shape (A, B, C)
        axis_labels: optional list of 3 labels for the three axes
        k_values_scalar: values for X-axis
        w_values: values for Y-axis
    """
    from matplotlib.widgets import Slider
    assert grid.ndim == 3, "Grid must be 3D (X, Y, Z)"
    dim = grid.shape

    if np.iscomplexobj(grid):
        print("Warning: Grid contains complex values. Discarding imaginary part.")
        grid = np.real(grid)
    # Setup figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, 'Slice', 0, dim[0] - 1, valinit=0, valstep=1)

    z_idx = 0  # initial slice
    

    if k_values_scalar is None:
        k_values_scalar = np.arange(dim[2] + 1)
    if w_values is None:
        w_values = np.arange(dim[1] + 1)

    # if useLogNorm:
    #     norm = LogNorm(vmin=np.max(grid) / range_from_max, vmax=np.max(grid))
    # else:
    #     norm = None
    vmin = np.min(np.real(grid))
    vmax = np.max(np.real(grid))
    
    norm = LogNorm(vmin, vmax) if useLogNorm else Normalize(vmin, vmax)
    
    mesh = ax.pcolormesh(k_values_scalar, w_values, grid[:,:,z_idx], 
                     cmap='viridis', shading='auto',
                     norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, label=intensity)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks_pos and xticks_labels:
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_labels)

    # Slider update logic
    def update(val):
        z = int(slider.val)
        mesh.set_array(grid[z].ravel())
        fig.canvas.draw_idle()

    slider.on_changed(update)
    ax.set_title(f"{title} – Slice 0")
    
    fig.show()
    plt.pause(0.001) 
    
    return fig



def plot_2_1_d_grid(grid, title="Grid", xlabel="X-axis", ylabel="Y-axis", intensity_label="Intensity (a.u.)", save_path=None, useLogNorm=False, range_from_max=1000, useStandardAxis=False, config:Config=None,
              xticks_pos=[], xticks_labels=[], k_values_scalar=None, w_values=[], dos_grid=None, use_dos=True, use_intensity_k = True, prevent_reusing_fig=False, plot_also_imaginary_part=False,
              gaussian_filter_along_w = True, sigma_w = 2):
    """Plots a 2D grid with optional DOS plot on the side. returs the figure object. if it is complex, it will plot the real part and optionally the imaginary part as well."""
    
    grid = np.array(grid)
    
    #check if the grid is complex so in the case of "plot_also_imaginary_part" imag part is plotted (if not negligible) and if not, a warning is given and the imag part is discarded
    imag_fig = None
    if np.iscomplexobj(grid):
        if plot_also_imaginary_part and np.max(np.abs(np.imag(grid))) > 1e-3*np.max(np.abs(np.real(grid))): #plot only imag values when they are non zero
            #recall the function to plot the imaginary part
            imag_fig = plot_2_1_d_grid(np.imag(grid), title=title + " (Imaginary Part)", xlabel=xlabel, ylabel=ylabel, intensity_label=intensity_label, save_path=save_path, useLogNorm=useLogNorm, range_from_max=range_from_max, useStandardAxis=useStandardAxis, config=config,
                      xticks_pos=xticks_pos, xticks_labels=xticks_labels, k_values_scalar=k_values_scalar, w_values=w_values, dos_grid=dos_grid, use_dos=use_dos, prevent_reusing_fig=prevent_reusing_fig, plot_also_imaginary_part=False)
            if np.max(np.abs(np.imag(grid))) > 1e+3*np.max(np.abs(np.real(grid))): # if the imaginary part is much larger than the real part, we return only the imaginary part
                return imag_fig
        else:
            print("Warning: Grid contains complex values. Discarding imaginary part.")
            if config is not None:
                config.warnings.add("Some plotted grid contained non negligible complex values that are not plotted.")
        grid = np.real(grid)
    title = title #+"\n" # make more space between title and plot
    if prevent_reusing_fig:
        fig = plt.figure( figsize=(10, 6))
        fig.canvas.manager.set_window_title(title)
        
    else:
        fig = plt.figure(title, figsize=(10, 6))
    
    
    if useStandardAxis:
        xlabel = "k values in 1/a"
        ylabel = "energy units of J"
        if config is not None:
            k_values_scalar=config.plot_info["x_values_scalar"]
            xticks_label = config.plot_info["x_ticks_label"]
            xticks_pos = config.plot_info["x_ticks_pos"]
            w_values = config.plot_info["y_ticks_label"]

    # if not useLogNorm:
    #     image=plt.pcolormesh(k_values_scalar, w_values, grid, cmap='viridis', shading='auto' )
    # else:
    #     image=plt.pcolormesh(k_values_scalar, w_values,  grid, cmap='viridis', shading='auto', norm=LogNorm(np.max(grid)/range_from_max,np.max(grid))) 
    # if len(xticks_pos)!=0 and len(xticks_labels) != 0:
    #     ax = plt.gca()
    #     ax.set_xticks(xticks_pos,xticks_labels)
    # # if len(w_values)!=0:
    # #     ax = plt.gca()
    # #     ax.set_yticks(ticks=np.linspace(0, grid.shape[0],len(w_values)),labels=np.round(w_values,2))
        

    # fig.colorbar(image, label=intensity)  
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)


    # gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4,1], wspace=0.05)
    gs = gridspec.GridSpec(
    2, 4, figure=fig,
    width_ratios=[4.0, 0.12, 0.8, 1.2],  # <-- col2 is the gap
    height_ratios=[4.0, 1.0],
    wspace=0.15,  # keep general spacing small
    hspace=0.6
)

    ax0 = fig.add_subplot(gs[0, 0])              # heatmap
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax0)  # bottom line plot (same width as ax0)
    cax = fig.add_subplot(gs[0, 1])              # colorbar axis (close to heatmap)
    # gs[0,2] is intentionally empty spacer
    
    ax1 = fig.add_subplot(gs[0, 3], sharey=ax0) if use_dos else None  # right DOS plot
    ax0.set_title(title)
    # if the grid is 2D, plot it as a 2D heatmap
    if len(grid.shape)==2:
        if len(w_values) != 0:     
            if len(w_values) != grid.shape[0]+1:
                w_values = np.array(w_values)
                # Extend w_values with equal spacing until it matches grid.shape[1]
                if len(w_values) < grid.shape[0]+1:
                    step = w_values[1] - w_values[0] if len(w_values) > 1 else 1
                    num_to_add = grid.shape[0]+1 - len(w_values)
                    last_value = w_values[-1]
                    extension = [last_value + step * (i + 1) for i in range(num_to_add)]
                    w_values = np.concatenate([w_values, extension])
        else:
            w_values = np.arange(grid.shape[0]+1)
        if use_dos:
            dos_whole_k_space = False
            if dos_grid is not None:
                dos = sum_k_dependency(dos_grid)
                dos_whole_k_space = True
            else:   
                dos = sum_k_dependency(grid)
                    
            # Side DOS plot
            
            ax1.plot(dos, w_values[:-1])  # exclude last edge of w_values for bin alignment
            if dos_whole_k_space:
                ax1.set_xlabel("DOS (whole k-space or other)")
            else:
                ax1.set_xlabel(intensity_label)
            ax1.yaxis.set_tick_params(labelleft=False)  # Hide y-axis labels to avoid duplication
            ax1.grid(True)
            

        # grid is giving in the form (Y,X)!!!
        if k_values_scalar is None:
            #making the spacing correct
            k_values_scalar = [0]
            xticks_pos_new=[]
            if isinstance(xticks_labels, np.ndarray) and xticks_pos: #In case the labels are given as array points in k space the scaling can be adjusted so that the distances between points match the actual distances in k space (instead of just being equally spaced)
                for i in range(len(xticks_labels)):
                    #converting string back to array
                    x = xticks_labels[i]-xticks_labels[i-1] if i > 0 else xticks_labels[i]
                    steps = (xticks_pos[i]-xticks_pos[i-1]) if i > 0 else xticks_pos[i]
                    for j in range(steps):
                        k_values_scalar.append(k_values_scalar[-1] + np.linalg.norm(x)/steps)
                    xticks_pos_new.append(k_values_scalar[-1])
                #x_ticks pos was just the scalar value (index of calculated point) now it is the float value of the actual position
                xticks_pos = xticks_pos_new
                #k_values_scalar has to be one larger than the grid.shape[1]
                k_values_scalar.append(k_values_scalar[-1] + np.round(np.linalg.norm(x)/steps))
                
                k_values_scalar = np.array(k_values_scalar)
            else:
                k_values_scalar = np.arange(grid.shape[1]+1)
        elif k_values_scalar.shape[0] == grid.shape[1]: #for the plotting routine the k-values have to be one larger than the grid.shape[1] (just append last value again)
            k_values_scalar = np.append(k_values_scalar, k_values_scalar[-1]) 
        
        if use_intensity_k:
            # beneath intensity plot
            
            intensity_k = np.sum(grid, axis=0)
           
            ax2.plot(k_values_scalar[:-1], intensity_k)
            # ax2.set_ylabel(intensity_label)
            
        
        if len(w_values) == 0:
            w_values = np.arange(grid.shape[0]+1)
        if config.shading_colormesh_plots == 'gouraud':
            X, Y = np.meshgrid(k_values_scalar[:-1], w_values[:-1])
        else:
            X, Y = np.meshgrid(k_values_scalar, w_values) 
        if gaussian_filter_along_w:
            grid = gaussian_filter1d(grid, sigma=sigma_w, axis=0)
        if not useLogNorm:
            image = ax0.pcolormesh(X, Y, grid, cmap=config.cmap_plots, shading=config.shading_colormesh_plots)
        else:
            try:
                image = ax0.pcolormesh(
                        X,
                        Y,
                        grid,
                        cmap=config.cmap_plots,
                        shading=config.shading_colormesh_plots,
                        norm=SymLogNorm(
                            linthresh=1e-10*np.max(grid),          # linear range around zero
                            linscale=1.0,            # scale of the linear range
                            vmin=np.max(grid) / range_from_max,
                            vmax=np.max(grid),
                            base=10
                        )
                    )
            except Exception as e:
                print("Log Norm failed in plot_grid")
                image = ax0.pcolormesh(X, Y, grid, cmap=config.cmap_plots, shading=config.shading_colormesh_plots)
        try:
            cb = fig.colorbar(image, cax=cax)
            if not useLogNorm:
                from matplotlib.ticker import ScalarFormatter
                formatter = ScalarFormatter(useMathText=False)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
                cb.ax.yaxis.set_major_formatter(formatter)
                cb.update_ticks()
            cb.set_label(intensity_label)
        except Exception as e:
            print("Colorbar failed in plot_grid")
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel)

    # if it is 1D, plot it as a line plot
    elif len(grid.shape)==1:
        ax0.plot(grid)
    else: 
        raise ValueError("Grid must be 1D or 2D for plotting. Current shape: {}".format(grid.shape))
    if xticks_pos and xticks_labels:
        ax0.set_xticks(xticks_pos)
        ax0.set_xticklabels(xticks_labels)




    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if imag_fig is None:
        return fig
    else:
        return fig, imag_fig


def plot_iso_surface(values, surface_count=4,isomax_percentage_of_max=1, title="Isosurface Plot", xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis", save_path=None):
    
    import plotly.graph_objects as go

    values = np.real(values)

    # Get shape
    nx, ny, nz = values.shape
    x, y, z = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=np.min(values),
        isomax=np.max(values)*isomax_percentage_of_max,
        surface_count=surface_count,
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        )
    )

    if save_path:
        fig.write_html(save_path)
        
        

    return fig

def plot_points_3D(points, title="3D Points Plot", xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis", save_path=None):
    # Assuming points is an array of shape (N, 3)
    assert len(points.shape) == 2, "points must be of form (N, 3) or (3, N)"
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T  # Transpose if needed
    points = np.array(points)
    assert points.shape[1] == 3, "Points array must have shape (N, 3)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_3D_scatter_color_coded(grid):
    assert len(grid.shape) == 3, "Input grid must be 3D"
    # Create a 3D offset plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.indices(grid.shape)
    plot=ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=grid.flatten())
    fig.colorbar(plot)  
    plt.show()

def plot_3D_scatter(Z, title="3D Scatter Plot", xlabel="X-axis", ylabel="Y-axis", zlabel="Z-axis", save_path=None):
    # Assuming eigvals_grid is already defined as a 4D array
    if not isinstance(Z, list):
        Z = [Z]
    for z in Z:
        z = np.real(z)  # Ensure Z is real for plotting

        # Create coordinate grid
        nx, ny = z.shape
        X, Y = np.meshgrid(np.arange(ny), np.arange(nx))

        # Flatten for scatter plot
        x = X.ravel()
        y = Y.ravel()
        z = z.ravel()

        # Create 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Scatter Plot')
        
        return fig
    
    # plt.show()

def plot_max_x_projection(values, title="Max X Projection", xlabel="W", ylabel="kx", zlabel="ky", save_path=None, log_scale=True):
    import plotly.graph_objects as go

    values = np.real(values)
    nx, ny, nz = values.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)

    # For each (y, z), find x index of max value along x
    x_max_indices = np.argmax(values, axis=0)

    if log_scale:
        # Use log scale for the maximum values
        x_max_values = np.log(np.max(values, axis=0) + 1e-10)
    else:
        # Use linear scale for the maximum values
        x_max_values = np.max(values, axis=0)

    # Convert indices to real x coordinates
    x_coords = x[x_max_indices]

    # Create coordinate arrays
    Y, Z = np.meshgrid(y, z, indexing='ij')
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    X_flat = x_coords.flatten()
    V_flat = x_max_values.flatten()

    # Create scatter plot
    scatter = go.Scatter3d(
        x=X_flat,
        y=Y_flat,
        z=Z_flat,
        mode='markers',
        marker=dict(
            size=4,
            color=V_flat,
            colorscale='Viridis',
            colorbar=dict(title='Max Value') if not log_scale else dict(title='Log(Max Value)'),
            opacity=0.8
        )
    )

    # Create wireframe box
    wireframe = go.Mesh3d(
        x=[0, 0, 0, 0, 1, 1, 1, 1],
        y=[0, 0, 1, 1, 0, 0, 1, 1],
        z=[0, 1, 0, 1, 0, 1, 0, 1],
        i=[0, 0, 0, 1, 1, 2, 4, 5, 6, 3, 3, 7],
        j=[1, 2, 4, 3, 5, 3, 5, 4, 7, 2, 7, 6],
        k=[2, 4, 5, 2, 6, 7, 7, 6, 5, 6, 6, 4],
        color='gray',
        opacity=0.1,
        showscale=False
    )

    fig = go.Figure(data=[scatter, wireframe])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        )
    )

    if save_path:
        fig.write_html(save_path)

    return fig

def plot_sqw_ad_a_intensities(material, config:Config, sqw_intensities, q_values_grid, VALUES_TO_SAVE, title = ""):
    try:
        #fig = plt.figure(f"intensities k dependent tau:{t}")
        M = 2*material.N
        fig_grid, axs_grid = plt.subplots(M, M, figsize=(12, 10))
        fig_grid2, axs_grid2 = plt.subplots(M, M, figsize=(12, 10))
        fig_grid3, axs_grid3 = plt.subplots(M, M, figsize=(12, 10))
        if title != "":
            fig_grid.suptitle(title)
        else:
            fig_grid.suptitle(f"intensities sqw ad_a(NxN grid)")
        for a in range(M):
            for b in range(M):
                ax = axs_grid[a, b] if M!=1 else axs_grid
                ax2 = axs_grid2[a, b] if M!=1 else axs_grid2
                ax3 = axs_grid3[a, b] if M!=1 else axs_grid3
                # TODO: this is a old function that might make problems at certain grid sizes (thus try)
                try:
                    intensities_k_along_path, xticks_pos, xticks_labels = path_through_k_grid(
                        config,
                        material=material,
                        relative_grid_points=material.get_extended_relative_grid_points_path(),
                        dimension=material.dimension,
                        grid=sqw_intensities[0,a, b, :, :, :],
                        q_grid=q_values_grid,
                        no_w_values=True,
                    )
                except Exception as e:
                    config.warnings.add(f"Error in path_through_k_grid: {e}")
                    continue
                #lw = functions.give_line_width([i, b, a], [2 * material.N, 3, 3])
                ax2.pcolormesh(np.real(sqw_intensities[0,a, b, :, :, sqw_intensities.shape[-1]//3]),vmin = 0, vmax=5, cmap='viridis', shading='auto')
                ax3.pcolormesh(np.imag(sqw_intensities[0,a, b, :, :, sqw_intensities.shape[-1]//3]),vmin = 0, vmax=5, cmap='viridis', shading='auto')
                ax.plot(np.real(intensities_k_along_path), label=f"I_i{a}_j{b}", linewidth=2)
                ax.set_ylim(-1, 5)
                ax.set_title(f"i={a}, j={b}")
                ax.set_xticks(xticks_pos)
                ax.set_xticklabels(xticks_labels)
                ax.legend(fontsize=7)
        fig_grid.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_grid2.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    except NameError:
        config.warnings.add("try to plot sqw but not defined")

    return fig_grid, fig_grid2

def plot_path_through_grid(path, q_grid=None, material=None):
    """
    Plots a 3D scatter plot of the path through the grid.
    Assumes grid is an array-like of shape (N, 3) or (3, N).
    Axes are set to have equal spacing.
    If material is provided, the lattice vectors are also drawn to visualize the grid structure.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    
    fig = plt.figure("Path Through Grid (3D)")
    ax = fig.add_subplot(111, projection='3d')
    if material is not None:
        lv = material.lattice_vectors if material is not None else None
        rec_lv = give_reciprocal_lv(lv=lv)
        for r in rec_lv:
            # Draw the reciprocal lattice vectors
            ax.quiver(0, 0, 0, r[0], r[1], r[2], color='blue', arrow_length_ratio=0.1)
        for r in lv:
            
            # Draw the reciprocal lattice vectors
            ax.quiver(0, 0, 0, r[0], r[1], r[2], color='red', arrow_length_ratio=0.1)

    path = np.array(path)
    if path.shape[0] == 3 and path.shape[1] != 3:
        # Shape (3, N) -> transpose to (N, 3)
        path = path.T
    points = q_grid[path[:, 0], path[:, 1], path[:, 2]] if q_grid is not None else path
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio for all axes
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_plane_from_normal(ax, point, normal, size=1.0, alpha=0.4, color='lightblue', resolution=20):
    """Plot a plane defined by a point and normal vector."""
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # pick vector not parallel to normal
    tmp = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])

    # basis vectors of the plane
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    s = np.linspace(-size, size, resolution)
    t = np.linspace(-size, size, resolution)
    S, T = np.meshgrid(s, t)

    X = point[0] + S*u[0] + T*v[0]
    Y = point[1] + S*u[1] + T*v[1]
    Z = point[2] + S*u[2] + T*v[2]  # corrected bug: use u and v consistently

    ax.plot_surface(X, Y, Z, alpha=alpha, color=color)

def plot_BZ_from_lv(ax, reciprocal_vectors, color='lightgreen', starting_depth=3, include_negative=True):
    """Plot the Brillouin zone from reciprocal lattice vectors."""
    # Create a meshgrid for the Brillouin zone
    origin = np.zeros(3)
    def plot_from_origin(origin, depth):
        if depth == 0:
            return
        for vec in reciprocal_vectors:
            if np.any(vec != origin):
                linestyle = 'dashed' if depth < starting_depth else 'solid'
                alpha = 1 if depth < starting_depth else 0.5
                ax.quiver(*origin, *vec, color=color, linestyle=linestyle, alpha=alpha)
                if include_negative:
                    ax.quiver(*origin, *-vec, color=color, linestyle=linestyle, alpha=alpha)
            plot_from_origin(origin + vec, depth - 1)
            if include_negative:
                plot_from_origin(origin - vec, depth - 1)
    plot_from_origin(origin, depth=starting_depth)
def plot_vectors_3d(vectors, colors=None, labels=None, title="3D Vectors", save_path=None, fig=None, ax=None, axis_labels=("kx", "ky", "kz"), view_elev=20, view_azim=45, view_roll=0, origins=None):
            """
            Plots any number of vectors in a 3D space.

            Parameters:
                vectors (List[np.ndarray]): List of 3D vectors to plot.
                colors (List[str]): List of colors for each vector (optional).
                labels (List[str]): List of labels for each vector (optional).
                title (str): Title of the plot.
                save_path (str): Path to save the plot (optional).
            """
            vectors = [np.array(v).astype(np.complex128).flatten() for v in vectors]  # Ensure all vectors are numpy arrays
            fig = fig or plt.figure()
            ax = ax or fig.add_subplot(111, projection='3d')
            ax.set_title(title)

            origins = [np.zeros(3)]*len(vectors) if origins is None else origins  # Origin point for all vectors
            colors = colors or ['b'] * len(vectors)
            labels = labels or [f"Vector {i+1}" for i in range(len(vectors))]

            for origin_point, vector, color, label in zip(origins, vectors, colors, labels):
                ax.quiver(*origin_point, *vector, color=color, label=label)

            max_len = max(np.linalg.norm(v) for v in vectors)
            ax.set_xlim([-max_len, max_len])
            ax.set_ylim([-max_len, max_len])
            ax.set_zlim([-max_len, max_len])

            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_zlabel(axis_labels[2])
            ax.legend()
            
            ax.view_init(elev=view_elev, azim=view_azim, roll=view_roll)

            if save_path:
                plt.savefig(save_path)
            #plt.show()
            
            return fig, ax


########################################################################################################################
########################################################################################################################
################################                          PATHS THROUGH GRIDS           ######################
########################################################################################################################
########################################################################################################################

def path_through_grid_from_relative_points(relative_points, q_grid:np.ndarray):
    """
    Takes a list of relative points and returns startpoint, directions and steps to go through the grid.
    """
    relative_points = np.array(relative_points)
    
    absoulte_points = np.einsum("ij,j->ij",relative_points,q_grid.shape[:-1])
    # Check if absolute points are close to whole numbers
    if not np.allclose(absoulte_points, np.round(absoulte_points), atol=1e-6):
        assert False, "realtive points dont match actual grid points. Please check the relative points or the amount of k-values (should be divisible by the realtive positions)." 
    else:
        absoulte_points = np.round(absoulte_points).astype(int)
    startpoint = absoulte_points[0]
    directions = []
    steps = []
    for i in range(1, len(absoulte_points)):
        direction = absoulte_points[i] - absoulte_points[i-1]
        # Find the greatest common divisor (GCD) of all three components
        # Use abs to avoid negative GCDs
        def gcd3(a, b, c):
            return np.gcd(np.gcd(abs(a), abs(b)), abs(c)) 
        divisor = gcd3(direction[0], direction[1], direction[2]) if not np.all(direction == 0) else 1
        if divisor == 0:
            divisor = 1

        directions.append(direction // divisor)
        steps.append(divisor)
    return startpoint, directions, steps

def interpolated_path_from_relative_points(relative_grid_points:np.ndarray, steps, nx=None, ny=None, nz=None):
    """
    Takes relative_grid_points of shape (N, 3) and steps and returns interpolated float indices of the target grid with dimension (nx, ny, nz) of shape (N*steps, 3).
    if nx, ny, nz are not given, they are assumed to be 1 --> returns relative position in the target grid."""
    
    # ---- Build the interpolated path ----
    steps_lin = np.linspace(0, 1, steps)
    if steps == 1:
        all_rel = relative_grid_points
    else:
        all_rel = np.concatenate([
            np.einsum("i,j->ij", steps_lin, relative_grid_points[i]-relative_grid_points[i-1]) + relative_grid_points[i-1]
        for i in range(1, len(relative_grid_points))
    ])
    if nx is None or ny is None or nz is None:
        return all_rel
    else:
        path_coords = np.column_stack([
            all_rel[:, 0] * (nx - 1),
            all_rel[:, 1] * (ny - 1),
            all_rel[:, 2] * (nz - 1),
        ])
    return path_coords

def from_q_points_to_angles(q_points):
    assert False, "doesnt work since the three angles can only move around the unit sphere... so not all w-values are accessible"
    phi = []
    alpha = []
    theta = []
    p,t,a = sp.symbols("p t a")
    for q in q_points:
        sol = sp.solve([sp.Eq(q[0], sp.sin(p) * ((sp.cos(a) - sp.cos(2*t - a)))),
                  sp.Eq(q[1], sp.cos(p) * ((sp.cos(a) - sp.cos(2*t - a)))),
                  sp.Eq(q[2], -sp.sin(a) - sp.sin(2*t - a))], (p, t, a))
        phi.append(sol[0][0])        
        alpha.append(sol[0][2])
        theta.append(sol[0][1])
    return np.array(phi), np.array(alpha), np.array(theta)
    

def path_through_grid_interpolated(grid, relative_grid_points:np.ndarray, steps, make_periodic=True, config:Config=None):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import map_coordinates
    """takes a grid of shape (..., X, Y, Z) and relative_grid_points with respect to the whole grid of shape (N, 3) and steps and returns an interpolated path through the grids (...,N).
    Example : grid of shape (..., 3, 4, 5) --> relative_grid_points[0] = [0.5,0.5,0,5] returns the grid interpolated at (...,1,1.5,2) for each index ..."""

    if make_periodic:
        grid = make_grid_periodic(grid)

    # grid.shape = (A, B, ..., X, Y, Z)
    grids_shape = grid.shape[:-3]
    nx, ny, nz = grid.shape[-3:]

    # # Axes
    # x = np.arange(nx)
    # y = np.arange(ny)
    # z = np.arange(nz)
    
    #these are the indices but floating numbers for interpolation
    path_coords = interpolated_path_from_relative_points(relative_grid_points, steps, nx, ny, nz)
    
    #for debugging
    # lv = np.array([grid[:,1,0,0], grid[:,0,1,0], grid[:,0,0,1]])
    # path_coords@lv
    # ---- Prepare grid for vectorized interpolation ----
    # grid_flat: (batch, nx, ny, nz)
    batch = np.prod(grids_shape)
    grid_flat = grid.reshape(batch, nx, ny, nz)
    
    # Move batch dim to last: (nx, ny, nz, batch)
    # grid_for_interp = np.moveaxis(grid_flat, 0, -1)

    # Create vectorized interpolator
    # interp = RegularGridInterpolator((x, y, z), grid_for_interp, method="linear", )
    # config.debug_timer.print_difference("before allocating vals for interpolation")
    vals = np.empty((batch, path_coords.shape[0]), dtype=np.complex128) 
    # config.debug_timer.print_difference("after allocating vals for interpolation")
    
    for b in range(batch):
        #use map_coordinates to have wrapping around the grid
        
        #need to do real and imaginary part separately since map_coordinates does not support complex numbers
        real_interp = map_coordinates(np.real(grid_flat[b]), path_coords.T,
                              order=1, mode='wrap')
        imag_interp = map_coordinates(np.imag(grid_flat[b]), path_coords.T,
                                    order=1, mode='wrap')
        vals[b] = real_interp + 1j * imag_interp
        # vals[b] = map_coordinates(grid_flat[b], path_coords.T, order=1, mode='wrap')
    config.debug_timer.print_difference("after interpolation loop")
    
    # Interpolate -> (num_steps_total, batch)
    #vals = interp(path_coords)

    # Reshape back
    return vals.reshape(grids_shape + (path_coords.shape[0],))

def path_through_k_grid(config:Config, grid, max_value_k=None, q_grid=None,relative_grid_points = None, startpoint=None, directions=None, steps=None, dimension=None, plot_path=False, material=None, no_w_values=False):
    """
    Takes a startpoint, a list of directions and the number of steps in each direction and returns a grid with the values of the original grid at each point and w-value (y-axis)
    If relative_grid_points are given, it calculates the startpoint, directions and steps from them.
    If no startpoint, directions and steps or relative_grid_points are given, it uses the standard paths through the k-grid.
    If no_w_values is True, it goes only throgh the k-values and adds a dummy dimension to the grid.
    If max_value_k is given, it is used to calculate the xticks labels. 
    plot_path is used to plot the path through the grid, if material is given, the lattice vectors are also plotted.
    
    return: result, xticks_pos, xticks_labels
    result: the values of the grid at each point in the path
    """

    if relative_grid_points is not None:
        #if relative_grid_points is given, it is used to calculate the startpoint, directions and steps
        startpoint, directions, steps = path_through_grid_from_relative_points(relative_grid_points, q_grid)
    
    if no_w_values:
        #adding a dummy dimension if no_w_values is True
        grid = grid[np.newaxis, ...]
    assert grid.ndim == 4, "Grid must be 4D (w, kx, ky, kz) or 3D (kx, ky, kz) if no_w_values is True. Current shape: {}".format(grid.shape)
    
    if startpoint is None or directions is None or steps is None:
        if dimension is None:
            raise ValueError("Either startpoint, directions and steps must be given or dimension or relative_grid_points must be given.")
        else:
            #these are the standard paths through the k-grid
            if dimension == 2:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([-1, -1, 0]),
                ]
            elif dimension == 3:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([-1, -1, -1]),
                ]
            elif dimension == 1:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                ]

            steps = [int(grid.shape[1] / 2)]*len(directions)

    path=[startpoint]
    result=[]
    xticks_pos=[]
    xticks_labels= np.round(np.einsum("ia, aj -> ij", relative_grid_points,np.array(material.get_reciprocal_lattice_vectors())),2).tolist() if (material is not None and relative_grid_points is not None) else []
    xticks_labels = [np.array(label) for label in xticks_labels]
    def add_xticks_pos_labels():
        if q_grid is not None:
            xticks_pos.append(len(path)-1)
            # mask_overshoot_grid_size = q_grid.shape[:-1]<=path[-1]
            # path[-1][mask_overshoot_grid_size] = np.array(q_grid.shape[:-1])[mask_overshoot_grid_size]-1
            # if np.any(mask_overshoot_grid_size):
            #     config.warnings.add("some grid points exceeded the grid size and were set to the maximum value of the grid.")
            if (material is None or relative_grid_points is None):
                xticks_labels.append(np.round(q_grid[*path[-1]],2))
            # xticks_labels.append(np.round(q_grid[*path[-1]],2))
            # if np.all(xticks_labels[-1] == 0) and relative_grid_points is not None and np.any(relative_grid_points[len(xticks_labels)-1] != np.array([0,0,0])):
            #     xticks_labels[-1] = np.round(2*q_grid[*path[-2]]-q_grid[*path[-3]],2) # extrapolate linearly if the label is zero since it is already wrapped around
        elif max_value_k is not None:
            xticks_pos.append(len(path)-1) #-1 since path has already one element
            xticks_labels.append(np.round((path[-1])/len(grid[1])*max_value_k,2)) #TODO: this doesnt fit completely since the path doesnt start at 1 but at 0
        else:
            raise ValueError("Either q_grid or max_value_k must be given to calculate the xticks labels.")
    
    #taken out since I want to "overstep" so that 1/2 as relative point ends up in the middle of k-space
    #steps[0] = steps[0] - 1 # # the first step is already taken with the startpoint, so we take one less step
    for i in range(len(directions)):
        #each time the direction is changed, the value of the k-value is added to the xticks
        add_xticks_pos_labels()
        for j in range(steps[i]):
            # if np.any(path[-1]+directions[i] < 0):
                
            #     break #if the next point is out of bounds, we stop the path
            path.append(path[-1]+directions[i])
            path[-1] = path[-1]%q_grid.shape[:-1]  #wrap around the grid if the point is out of bounds
    #the last point is added
    add_xticks_pos_labels()
    for p in path:
        try:
            result.append(grid[:,p[0],p[1],p[2]])
        except IndexError as e:
            print(f"IndexError: {e} at point {p}. This point is out of bounds for the grid with shape {grid.shape}.")
            return None, None, None
    if plot_path:
        plot_path_through_grid(path, q_grid=q_grid, material=material) #taking only the k-part of the grid for plotting

    if not no_w_values:
        #if no_w_values is True, the grid has only one dimension --> there is no need to transpose
        result = np.array(result).T
    
    return result, xticks_pos, xticks_labels

def path_through_k_grid_old(config:Config, grid, max_value_k=None, q_grid=None,relative_grid_points = None, startpoint=None, directions=None, steps=None, dimension=None, plot_path=False, material=None, no_w_values=False):
    """
    Takes a startpoint, a list of directions and the number of steps in each direction and returns a grid with the values of the original grid at each point and w-value (y-axis)
    If relative_grid_points are given, it calculates the startpoint, directions and steps from them.
    If no startpoint, directions and steps or relative_grid_points are given, it uses the standard paths through the k-grid.
    If no_w_values is True, it goes only throgh the k-values and adds a dummy dimension to the grid.
    If max_value_k is given, it is used to calculate the xticks labels. 
    plot_path is used to plot the path through the grid, if material is given, the lattice vectors are also plotted.
    
    return: result, xticks_pos, xticks_labels
    result: the values of the grid at each point in the path
    """

    if relative_grid_points is not None:
        #if relative_grid_points is given, it is used to calculate the startpoint, directions and steps
        startpoint, directions, steps = path_through_grid_from_relative_points(relative_grid_points, q_grid)
    
    if no_w_values:
        #adding a dummy dimension if no_w_values is True
        grid = grid[np.newaxis, ...]
    assert grid.ndim == 4, "Grid must be 4D (w, kx, ky, kz) or 3D (kx, ky, kz) if no_w_values is True. Current shape: {}".format(grid.shape)
    
    if startpoint is None or directions is None or steps is None:
        if dimension is None:
            raise ValueError("Either startpoint, directions and steps must be given or dimension or relative_grid_points must be given.")
        else:
            #these are the standard paths through the k-grid
            if dimension == 2:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([-1, -1, 0]),
                ]
            elif dimension == 3:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                    np.array([-1, -1, -1]),
                ]
            elif dimension == 1:
                startpoint=np.array([0, 0, 0])
                directions=[
                    np.array([1, 0, 0]),
                    np.array([-1, 0, 0]),
                ]

            steps = [int(grid.shape[1] / 2)]*len(directions)

    path=[startpoint]
    result=[]
    xticks_pos=[]
    xticks_labels=[]
    def add_xticks_pos_labels():
        if q_grid is not None:
            xticks_pos.append(len(path)-1)
            mask_overshoot_grid_size = q_grid.shape[:-1]<=path[-1]
            path[-1][mask_overshoot_grid_size] = np.array(q_grid.shape[:-1])[mask_overshoot_grid_size]-1
            if np.any(mask_overshoot_grid_size):
                config.warnings.add("some grid points exceeded the grid size and were set to the maximum value of the grid.")
            xticks_labels.append(np.round(q_grid[path[-1][0],path[-1][1],path[-1][2]],2))
        elif max_value_k is not None:
            xticks_pos.append(len(path)-1) #-1 since path has already one element
            xticks_labels.append(np.round((path[-1])/len(grid[1])*max_value_k,2)) #TODO: this doesnt fit completely since the path doesnt start at 1 but at 0
        else:
            raise ValueError("Either q_grid or max_value_k must be given to calculate the xticks labels.")
    
    steps[0] = steps[0] - 1 # # the first step is already taken with the startpoint, so we take one less step
    for i in range(len(directions)):
        #each time the direction is changed, the value of the k-value is added to the xticks
        add_xticks_pos_labels()
        for j in range(steps[i]):
            if np.any(path[-1]+directions[i] < 0):
                
                break #if the next point is out of bounds, we stop the path
            path.append(path[-1]+directions[i])
    #the last point is added
    add_xticks_pos_labels()
    for p in path:
        try:
            result.append(grid[:,p[0],p[1],p[2]])
        except IndexError as e:
            print(f"IndexError: {e} at point {p}. This point is out of bounds for the grid with shape {grid.shape}.")
            return None, None, None
    if plot_path:
        plot_path_through_grid(path, q_grid=q_grid, material=material) #taking only the k-part of the grid for plotting

    if not no_w_values:
        #if no_w_values is True, the grid has only one dimension --> there is no need to transpose
        result = np.array(result).T
    
    return result, xticks_pos, xticks_labels

########################################################################################################################
########################################################################################################################
################################                          CONVOLUTIONS           ######################
########################################################################################################################
########################################################################################################################


def convolute_grids(grid1, grid2, dimension=3, double_convolution=False, grid3=None):
    """takes two or three grids of shape (w,k*dim) and convolves them in the k-dimensions."""

    assert False, "This function is deprecated. Please use convolute_grids_fft instead."

    conv_degree=2
    if double_convolution:
        conv_degree=3
    amount_w_values = grid1.shape[0]
    grid_fineness = grid1.shape[1]
    conv_grids = np.zeros((conv_degree * amount_w_values,) + (grid_fineness,) * dimension + (1,)*(3-dimension), dtype=complex)

    tiling_shape=(2,)*dimension+(1,)*(3-dimension)
    #print(tiling_shape)
    for w_index in range(amount_w_values):
        #print(w_index)
        for w_index_2 in range(w_index, amount_w_values):
            #print(np.shape(np.tile(spectral_weight_grid[w_index_2],tiling_shape)))
            #print(np.shape(spectral_weight_grid[w_index_2]))
            temp_result=fftconvolve( np.tile(grid1[w_index_2],tiling_shape),grid2[w_index], mode='valid')
            #this is done since the the covolution of the first and the last "element" would otherwise give the same result
            temp_result=cut_array_by_one(temp_result)
            if double_convolution: 
                for w_index_3 in range(w_index_2, amount_w_values):
                    temp_result_2=fftconvolve( np.tile(temp_result,tiling_shape),grid3[w_index_3], mode='valid')
                    temp_result_2=cut_array_by_one(temp_result_2)
                    conv_grids[w_index+w_index_2+w_index_3]+= temp_result_2
            else:
                #TODO: check if correct to take the real part here
                conv_grids[w_index+w_index_2]+=temp_result
    return conv_grids
   # print("shape conv_intens "+str(np.shape(conv_intensities)))
    
#optimised version
def convolute_grids_fft(grid1, grid2, dimension=3, double_convolution=False, grid3=None, use_real=True):
    """
    Fast version: vectorized via FFT.
    grid* shape: (W, K1, K2, K3?) with only the first `dimension` spatial dims used.
    Returns shape ((deg)*W, K1, K2, K3?) where deg=2 or 3.
    """
    from scipy.fft import fftn, ifftn, fft, ifft, rfftn, irfftn

    assert grid1.shape[1:1+dimension] == grid2.shape[1:1+dimension]
    if double_convolution:
        assert grid3 is not None and grid3.shape[1:1+dimension] == grid1.shape[1:1+dimension]

    W = grid1.shape[0]
    spatial_shape = grid1.shape[1:1+dimension]
    # Preserve trailing singleton dims up to 3
    trailing = grid1.shape[1+dimension:]
    out_W = (3 if double_convolution else 2) * W

    # Choose real FFTs if inputs are real to save time/memory
    rfft = use_real and np.isrealobj(grid1) and np.isrealobj(grid2) and (not double_convolution or np.isrealobj(grid3))
    fftn_fn, ifftn_fn = (rfftn, irfftn) if rfft else (fftn, ifftn)

    # 1) FFT over spatial axes for each w-slice
    axes = tuple(range(1, 1+dimension))
    A_hat = fftn_fn(grid1, s=None, axes=axes)
    B_hat = fftn_fn(grid2, s=None, axes=axes)
    if double_convolution:
        C_hat = fftn_fn(grid3, s=None, axes=axes)

    # 2) Convolution along w using FFT (treat each spatial-frequency bin independently)
    #    conv over w: for each spatial bin, (A_hat *_w B_hat)[t] = sum_i A_hat[i]*B_hat[t-i]
    #    Compute via FFT along axis=0.
    #    Zero-pad length to out_W for 'full' convolution length = W+W-1 (here you sum to index i+j),
    #    but your code uses i from j..W-1 and accumulates into w_index+w_index_2 (length 2W),
    #    so we pad to 2W (or 3W for triple).
    W = grid1.shape[0]
    
    pad_W = out_W
    def wfft(x):
        return fft(x, n=pad_W, axis=0) #n is zero padding! if pad_W > x.shape[0]
    def iwfft(x):
        return ifft(x, n=pad_W, axis=0)

    AB_hat_w = iwfft(wfft(A_hat) * wfft(B_hat))  # shape: (out_W, *spatial_freq_shape, *trailing)

    if double_convolution:
        ABC_hat_w = iwfft(wfft(AB_hat_w) * wfft(C_hat))  # convolve once more along w
        H_w = ABC_hat_w
    else:
        H_w = AB_hat_w

    # 3) Inverse FFT over spatial axes to get back k-space convolution for every w-sum index
    #    Take real part if inputs were real (tiny imag from numerics).
    conv_spatial = ifftn_fn(H_w, s=None, axes=axes)
    if rfft:
        conv_spatial = conv_spatial.real
    # Output shape ((deg)*W, K1, K2, K3?, trailing...)
    return conv_spatial

def convolute_grids_fft_phases(grid1, grid2, d_in_first_unit_cell, dimension=3, double_convolution=False, grid3=None, use_real=True):
    """
    Fast version: vectorized via FFT.
    grid* shape: (W, K1, K2, K3?) with only the first `dimension` spatial dims used.
    Returns shape ((deg)*W, K1, K2, K3?) where deg=2 or 3.
    """
    from scipy.fft import fftn, ifftn, fft, ifft, rfftn, irfftn

    assert grid1.shape[1:1+dimension] == grid2.shape[1:1+dimension]
    if double_convolution:
        assert grid3 is not None and grid3.shape[1:1+dimension] == grid1.shape[1:1+dimension]

    def shift_grid_by_vector(grid, d_vector):
        """Shifts the grid in k-space by a phase factor corresponding to d_vector in real space."""
        # shifted_grid = np.zeros_like(grid, dtype=complex)
        shift_indices = np.round(np.array(grid.shape[1:])*d_vector).astype(int)
        shifted_grid = np.roll(grid, shift=shift_indices, axis=tuple(range(1, 4)))
        return shifted_grid
    temp = shift_grid_by_vector(grid1, d_vector=d_in_first_unit_cell)
    W = grid1.shape[0]
    spatial_shape = grid1.shape[1:1+dimension]
    # Preserve trailing singleton dims up to 3
    trailing = grid1.shape[1+dimension:]
    out_W = (3 if double_convolution else 2) * W

    # Choose real FFTs if inputs are real to save time/memory
    rfft = use_real and np.isrealobj(grid1) and np.isrealobj(grid2) and (not double_convolution or np.isrealobj(grid3))
    fftn_fn, ifftn_fn = (rfftn, irfftn) if rfft else (fftn, ifftn)

    # 1) FFT over spatial axes for each w-slice
    axes = tuple(range(1, 1+dimension))
    A_hat = fftn_fn(grid1, s=None, axes=axes)
    B_hat = fftn_fn(grid2, s=None, axes=axes)
    if double_convolution:
        C_hat = fftn_fn(grid3, s=None, axes=axes)

    B_hat = shift_grid_by_vector(B_hat, d_vector=d_in_first_unit_cell)
    # 2) Convolution along w using FFT (treat each spatial-frequency bin independently)
    #    conv over w: for each spatial bin, (A_hat *_w B_hat)[t] = sum_i A_hat[i]*B_hat[t-i]
    #    Compute via FFT along axis=0.
    #    Zero-pad length to out_W for 'full' convolution length = W+W-1 (here you sum to index i+j),
    #    but your code uses i from j..W-1 and accumulates into w_index+w_index_2 (length 2W),
    #    so we pad to 2W (or 3W for triple).
    W = grid1.shape[0]
    
    pad_W = out_W
    def wfft(x):
        return fft(x, n=pad_W, axis=0) #n is zero padding! if pad_W > x.shape[0]
    def iwfft(x):
        return ifft(x, n=pad_W, axis=0)

    AB_hat_w = iwfft(wfft(A_hat) * wfft(B_hat))  # shape: (out_W, *spatial_freq_shape, *trailing)

    if double_convolution:
        ABC_hat_w = iwfft(wfft(AB_hat_w) * wfft(C_hat))  # convolve once more along w
        H_w = ABC_hat_w
    else:
        H_w = AB_hat_w

    # 3) Inverse FFT over spatial axes to get back k-space convolution for every w-sum index
    #    Take real part if inputs were real (tiny imag from numerics).
    conv_spatial = ifftn_fn(H_w, s=None, axes=axes)
    if rfft:
        conv_spatial = conv_spatial.real
    # Output shape ((deg)*W, K1, K2, K3?, trailing...)
    return conv_spatial

########################################################################################################################
########################################################################################################################
################################                          OUTPUT / SAVINGS / RENDERING           ######################
########################################################################################################################
########################################################################################################################
def save_grid_to_file(grid, *coords, filename="grid.npz"):
    """
    Save a d-dimensional grid and its coordinate arrays to a .npz file.

    Parameters:
        grid (np.ndarray): The grid data.
        *coords: Any number of coordinate arrays (1D arrays for each axis).
        filename (str): Output filename (should end with .npz).
    """
    # Prepare a dictionary with coordinate names
    save_dict = {"grid": grid}
    for i, coord in enumerate(coords):
        save_dict[f"coord_{i}"] = coord
    np.savez(filename, **save_dict)

def load_grid_from_file(filename="grid.npz"):
    """
    Load a d-dimensional grid and its coordinate arrays from a .npz file.

    Returns:
        grid (np.ndarray): The grid data.
        coords (list of np.ndarray): List of coordinate arrays.
        
    """
    data = np.load(filename)
    grid = data["grid"]
    # Collect all coordinate arrays in order
    coords = []
    i = 0
    while f"coord_{i}" in data:
        coords.append(data[f"coord_{i}"])
        i += 1
    return grid, coords



def save_figures_to_png(figures, out_put_folder="outputs/figs", dpi=200, notitle=True):
    import os

    if not os.path.exists(out_put_folder):
        os.makedirs(out_put_folder)

    for i, fig in enumerate(figures):
        try:
            title = fig.canvas.manager.get_window_title()
        except Exception:
            title = f"Figure_{i}"
        
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
        
        if notitle:
            try:
                for ax in fig.axes:
                    ax.set_title("")
                fig.suptitle("")
            except Exception:
                pass

        try:
            # fig.set_size_inches(16,9, forward=True) #bring it into "fullscreen" size
            
            fig.tight_layout() #make it look nicer and avoid cutting off labels, this is done after setting the size to avoid it being too small
            fig.savefig(
                f"{out_put_folder}/{safe_title}.png",
                dpi=dpi,
                pad_inches=0.05,
                bbox_inches='tight'
            )
        except Exception as e:
            print(f"Failed to save figure {title} as PNG: {e}")
            
def save_figures_as_images_to_pdf(
    config:Config,
    figures,
    filename: str = "output.pdf",
    figures_per_page: int = 4,
    dpi: int = 200,
    page_title_prefix: str = "Page",
    material= None
):
    from matplotlib.backends.backend_pdf import PdfPages
    from Material import Material
    
    material:Material = material  # type hint
    
    material_description = f"{material.name}" if material is not None else "Material Description Not Provided"
    for fig in figures:
        try:
            fig.axes[0].set_title(f"{fig.canvas.manager.get_window_title()}")
        except Exception as e:
            continue
    with PdfPages(filename) as pdf:
        for i in range(0, len(figures), figures_per_page):
            figs_subset = figures[i:i + figures_per_page]
            n = len(figs_subset)
            ncols = 2
            fig, axs = plt.subplots(nrows=math.ceil(n/ncols), ncols=ncols, figsize=(8.5, 11), dpi=dpi)
            if len(axs)> len(figs_subset):
                axs = axs[:len(figs_subset)]
            axs = axs.flatten() #if n > 1 else [axs]  # Flatten if more than one figure, else keep as single Axes
           
            fig.suptitle(f"{material_description} {i // figures_per_page + 1}", fontsize=16)
            if len(config.warnings) != 0 and i==0: 
                warning_str = ""
                warning_str = f"""time: {config.timestamp}\n config:{config.name} // W: {config.amount_w_values} K: {config.amount_k_values}  \n tau:{material.tau} normal: {material.normal_vector_to_rotation_plane} spin direction: {material.atoms[0][0].spin_dir}\n
                polarisations: {config.rixs_config.polarisations_to_sum} """
                for w in config.warnings:
                    warning_str = warning_str + f"\n{w}"
                fig.text(0.5, 0.95, f"{warning_str}", ha='center', va='top', fontsize=10, color='gray')

            for ax, fig_src in zip(axs, figs_subset):
                try:
                    buf = io.BytesIO()
                    fig_src.savefig(buf, format='png', dpi=dpi)
                    buf.seek(0)

                    img = plt.imread(buf)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes around image
                    buf.close()
                except Exception as e:
                    config.warnings.add(f"Failed to add figure to PDF: {e}")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        info_str = "additonal information:\n"
            
        if config.use_explicit_calc_of_sqw and config.calc_two_magnon_spectra:
            info_str += "sqw 4 symmetries (a ad) (less then 10 entries)\n"
            for sym in material.sqw_4_symmetries:
                if len(sym) < 10: #only add symmetries with less than 10 entries to avoid cluttering the pdf
                    info_str += f"{sym}\n"
            
        if material.sym_classes_to_xyz_4 is not None:
            info_str += "sqw 4 symmetries (xyz) (less then 10 entries)\n"
            temp = give_symmetries_of_tensors(material.sym_classes_to_xyz_4,4, give_mean_abs_value=True, compare_only_a_sample = False)
            for sym in temp[0]:
                if len(sym) < 10: #only add symmetries with less than 10 entries to avoid cluttering the pdf
                    info_str += f"{sym}\n"
        
        if material.sym_classes_to_xyz_2 is not None:
            info_str += "sqw 2 symmetries (xyz)(less then 10 entries)\n"
            temp = give_symmetries_of_tensors(material.sym_classes_to_xyz_2,4, give_mean_abs_value=True, compare_only_a_sample = False)
            for sym in temp[0]:
                if len(sym) < 10: #only add symmetries with less than 10 entries to avoid cluttering the pdf
                    info_str += f"{sym}\n"
        fig_text = plt.figure(figsize=(8.5, 11), dpi=dpi)
        fig_text.clf()
        fig_text.text(0.5, 0.5, info_str, ha="center", va="center", wrap=True, fontsize=12)
        pdf.savefig(fig_text)
        plt.close(fig_text)
            

def render_matrix_as_image(matrix, filename="matrix.png"):
    """
    Renders a given matrix as a LaTeX-styled image and saves it.
    """
    # Generate the LaTeX code for the matrix
    tex_code = "\\begin{bmatrix}\n"
    for row in matrix:
        tex_code += "  " + " & ".join(map(str, row)) + " \\\\\n"
    tex_code += "\\end{bmatrix}"

    # Configure matplotlib for LaTeX rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Create a figure and render the LaTeX matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"${tex_code}$", fontsize=20, ha="center", va="center")
    ax.axis("off")  # Turn off axes for clean output

    # Save the image
    #plt.savefig(filename, bbox_inches="tight", pad_inches=0.5)
    plt.show()


def save_latex_matrices(array, filename="latex_matrices.txt", env="bmatrix", precision=4, split_files=False, output_dir="matrices"):
    """
    Save 2D or 3D NumPy array as LaTeX matrices.

    Parameters:
        array (np.ndarray): 2D or 3D array to export.
        filename (str): Output file if saving all matrices in one.
        env (str): LaTeX environment: "bmatrix", "pmatrix", etc.
        precision (int): Number of decimal places.
        split_files (bool): If True, saves each matrix to a separate file.
        output_dir (str): Directory to save separate files.
    """
    def format_matrix(mat):
        lines = [f"\\begin{{{env}}}"]
        for row in mat:
            row_str = " & ".join(
                f"{x.real:.{precision}f}" + (f" + {x.imag:.{precision}f}i" if np.iscomplexobj(x) and x.imag != 0 else "")
                for x in row
            )
            lines.append(f"{row_str} \\\\")
        lines.append(f"\\end{{{env}}}")
        return "\n".join(lines)

    if array.ndim == 2:
        # Single matrix
        latex = format_matrix(array)
        with open(filename, "w") as f:
            f.write(latex)
        print(f"Saved 2D matrix to {filename}")

    elif array.ndim == 3:
        if split_files:
            os.makedirs(output_dir, exist_ok=True)
            for i, mat in enumerate(array):
                file_path = os.path.join(output_dir, f"matrix_{i:03d}.txt")
                with open(file_path, "w") as f:
                    f.write(format_matrix(mat))
            print(f"Saved {len(array)} matrices to '{output_dir}' directory.")
        else:
            with open(filename, "w") as f:
                for i, mat in enumerate(array):
                    f.write(f"% Matrix {i}\n")
                    f.write(format_matrix(mat))
                    f.write("\n\n")
            print(f"Saved {len(array)} matrices to {filename}")
    else:
        raise ValueError("Input array must be 2D or 3D.")


def save_config_to_json(config, filename: str = None, max_array_size: int = 1000):
    """
    Save dataclass config to JSON, skipping or summarizing large/unsupported entries.
    """
    from LSW import Configurations
    from dataclasses import asdict, is_dataclass

    config: Configurations = config  # type hint

    if filename is None:
        timestamp = getattr(config, "timestamp", "no_time")
        filename = f"config/config_{timestamp}.json"

    # Convert dataclass to dict if applicable
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config.__dict__

    def safe_convert(obj):
        """Convert non-serializable objects into JSON-safe representations."""
        # --- NumPy types ---
        if isinstance(obj, np.generic):
            return obj.item()  # scalar NumPy type
        if isinstance(obj, np.ndarray):
            if obj.size > max_array_size:
                return f"<ndarray shape={obj.shape}, dtype={obj.dtype}, size too large>"
            return obj.tolist()
        # --- Built-ins ---
        if isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        # --- Dictionaries and sequences ---
        if isinstance(obj, dict):
            return f"no dicts are safed"
        if isinstance(obj, (list, tuple)):
            return [safe_convert(v) for v in obj]
        # --- Fallback ---
        return f"<Unsupported type {type(obj).__name__}>"

    config_safe = {k: safe_convert(v) for k, v in config_dict.items()}

    try:
        with open(filename, "w") as f:
            json.dump(config_safe, f, indent=4)
        print(f"Config saved to {filename}")
    except Exception as e:
        print(f"Error saving configuration to {filename}: {e}")

def find_files_with_keywords(directory:str, keywords: List[str]) -> List[str]:
    """
    Finds all files in a given directory whose filenames contain certain keywords.

    Parameters:
        directory (str): The directory to search in.
        keywords (List[str]): List of keywords to search for.

    Returns:
        List[str]: List of file paths whose filenames contain the keywords.
    """
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(keyword in file for keyword in keywords):
                matching_files.append(os.path.join(root, file))
    return matching_files




########################################################################################################################
########################################################################################################################
################################                  NUMERICS, Grid creation / integration           ######################
########################################################################################################################
########################################################################################################################
#TODO: normalise "better"
def normalise_grid(grid):
    total_sum = np.sum(grid)
    if total_sum != 0:
        grid = grid / total_sum
    return grid

def create_Gamma(half_size):
    half_size = int(half_size)
    Gamma = np.zeros((2*half_size, 2*half_size), dtype=complex)
    for i in range(half_size):
        Gamma[i, i] = 1
        Gamma[half_size + i, half_size + i] = -1
    return Gamma

def sum_k_dependency(grid):
    """Sums k-dependency of a grid over all k-values, i.e. sums over all but the first axis"""
    if grid.ndim < 2:
        raise ValueError("Grid must have at least 2 dimensions.")
    
    summed_grid = np.sum(grid, axis=tuple(range(1, grid.ndim)))
    return summed_grid



def integrate_nd(f, grid_points, spacing,dimension):
    """
    Piecewise-linear (trapezoidal) integration over 1–3 axes.

    Parameters
    ----------
    f : ndarray
        Values on a grid. Shape must start with (Nx[, Ny[, Nz]]).
    coords : sequence of 1D arrays
        Coordinate arrays [x], [x,y], or [x,y,z] (non-uniform allowed).
    dimension : int, optional
        Number of axes to integrate (1, 2, or 3). Defaults to len(coords).

    Returns
    -------
    integral : ndarray or float
        Result after integrating over the first `dimension` axes.
        (If f has extra trailing dims, they are preserved.)
    """
    # if dimension is None:
    #     dimension = len(grid_points.shape[:-1])
    if not (1 <= dimension <= 3):
        raise ValueError("dimension must be 1, 2, or 3")

    # grid_points = [np.asarray(c) for c in grid_points[:dimension]]
    # basic shape checks
    if f.shape[:dimension] != grid_points.shape[:-1][:dimension]:
        raise ValueError("f shape and grid_points shape do not match.")
    res = f[(slice(None),)*dimension + (0,)*(3-dimension)]  # only include the axis that I want to integrate over
    # integrate from last axis down so axis indices stay valid
    for ax in reversed(range(dimension)):
        # grid_slice = (slice(None),) * (ax+1) + (0,)*(2-ax) + (ax,)
        slice_only_ax = (0,)*(ax) + (slice(None),) + (0,)*(2-ax) + (ax,)
        # slice_only_ax_vec = (0,)*(ax) + (slice(None),) + (0,)*(2-ax) 
        # grid_vec_along_ax = grid_points[slice_only_ax_vec]
        # delta_g_ax = np.array([0]+[la.norm(np.array(grid_vec_along_ax[i+1] - grid_vec_along_ax[i])) for i in range(len(grid_vec_along_ax)-1)])
        
        sign_correction = 1 if grid_points[slice_only_ax][-1] > grid_points[slice_only_ax][0] else -1
        res = sign_correction * np.trapezoid(res, spacing) # the grid points might be decreasing instead increasing in some axis --> the sign correction corrects that by explicit checking
    return res



def give_refined_grid_around_point_old(grid, point_idx, dimension, refinement_factor=100, fraction_to_neighbour_to_include=1/2, decrease_ratio=1/2):
    point = grid[*point_idx]
    neigbour_index = point_idx+np.array((-1,)*(dimension)+(0,)*(3-dimension))
                        # (1,)+(-1,)*(dimension-1)+(0,)(3-dimension),
                        # (-1,)+(1,)+(-1,)*(dimension-2)+(0,)(3-dimension),
                        # (-1,)*2+(1,)]
    #prevent being outside of the grid
    if np.any(np.array(neigbour_index)<0):
        neigbour_point = grid[*(-neigbour_index)]
    else:
        neigbour_point = grid[*neigbour_index]
        
    d= point+(neigbour_point-point)*fraction_to_neighbour_to_include #vector pointing from the point to the corner neighbour
    box_corner_point = point+d
    # spacings_per_d = [np.linspace(box_corner_point[axis], box_corner_point[axis]-2*d[axis], refinement_factor) for axis in range(dimension)]
    spacings_per_d = [decreasing_step_spacing(start=np.array([box_corner_point[axis]]), end=np.array([point[axis]]), num_points=refinement_factor, decrease_ratio=decrease_ratio) for axis in range(dimension)]
    spacings_per_d += [np.array([0]) for _ in range(3-dimension)]
    X, Y, Z = np.meshgrid(*spacings_per_d, indexing='ij')
    grid_points = np.stack([X, Y, Z], axis=-1)
    
    grid_volume = 1
    for axs in range(dimension):
        grid_volume *= (2 * abs(d[axs]))
    return grid_points, grid_volume

def give_refined_grid_around_point(material, grid, point_idx, dimension, refinement_factor=100, fraction_to_neighbour_to_include=1/2, decrease_ratio=1/2):
    from Material import Material
    material:Material = material  # type hint
    point = grid[*point_idx]
    
    # spacings_per_d = [np.linspace(box_corner_point[axis], box_corner_point[axis]-2*d[axis], refinement_factor) for axis in range(dimension)]
    spacing = decreasing_step_spacing_float(start=fraction_to_neighbour_to_include, end=0, num_points=refinement_factor, decrease_ratio=decrease_ratio) 
    
    
    spacings_per_d = [np.array(spacing) for _ in range(dimension)]
    spacings_per_d += [np.array([0]) for _ in range(3-dimension)]
    
    X, Y, Z = np.meshgrid(*spacings_per_d, indexing='ij')
    grid_float_idcs = np.stack([X, Y, Z], axis=-1)
    grid_float_idcs = grid_float_idcs[:,:,:,:dimension] #only take the dimensions that are defined in the mateial
    lv_recip_scaled = np.array([material.get_reciprocal_lattice_vectors()[axs]/grid.shape[:3][axs] for axs in range(dimension)])
    
    grid_refined = np.einsum('...v,vl->...l', grid_float_idcs, lv_recip_scaled)+ point 
    
    scale = fraction_to_neighbour_to_include*2
    
    #I dont think I need this the volume element can be taken out, (I might not have the correct statistics since for non euclidean vectors certain directions might be overrepresented but the scaling should remain correct)
    if False: 
        match dimension:
            case 1:
                euclidean_volume = np.abs(la.norm(lv_recip_scaled[0]))  
                spacing_volume = np.abs(la.norm(lv_recip_scaled[0])/euclidean_volume *scale) #this takes into account that the volume element for non euclidean lattice vectors will not be the same as for euclidean ones, 
            case 2:
                euclidean_volume = np.abs(la.norm(lv_recip_scaled[0]) * la.norm(lv_recip_scaled[1])) 
                spacing_volume = np.abs(la.norm(la.cross(lv_recip_scaled[0], lv_recip_scaled[1]))/euclidean_volume * scale**2) #takes only "positive quarter" of the box thus times 4 
            case 3:
                euclidean_volume = np.abs(la.norm(lv_recip_scaled[0]) * la.norm(lv_recip_scaled[1]) * la.norm(lv_recip_scaled[2]))
                spacing_volume = np.abs(la.dot(la.cross(lv_recip_scaled[0], lv_recip_scaled[1]), lv_recip_scaled[2])/euclidean_volume * scale**3) 
                
    return grid_refined, spacing

def decreasing_step_spacing_float(start: float, end: float, num_points: int, decrease_ratio: float, use_approach_from_both_sides: bool = True) -> np.ndarray:
    if num_points < 2:
        raise ValueError("num_points must be at least 2 to create a spacing.")
    if decrease_ratio <= 0 or decrease_ratio > 1:
        raise ValueError("decrease_ratio must be positive and less than or equal to 1.")
    if decrease_ratio == 1:
        return np.linspace(start, end, num_points)


    r = decrease_ratio

    # Generate the geometric series
    series_factors = np.array([1-r ** i for i in range(num_points)])
    result = (end-start) * series_factors + start
    new_start = start + 2* (end - start)
    result_opposite = (end-new_start) * (series_factors) + new_start
    result = np.concatenate((result, np.flip(result_opposite)), axis=0)
    
    return result

def decreasing_step_spacing(start: np.ndarray, end: np.ndarray, num_points: int, decrease_ratio: float, use_approach_from_both_sides: bool = True) -> np.ndarray:
    """Generates a sequence of points between start and end (never reaches end) with spacing that is always reduced by geometric_ratio for each step."""
    
    if num_points < 2:
        raise ValueError("num_points must be at least 2 to create a spacing.")
    if decrease_ratio <= 0 or decrease_ratio > 1:
        raise ValueError("decrease_ratio must be positive and less than or equal to 1.")
    if decrease_ratio == 1:
        return np.linspace(start, end, num_points)


    r = decrease_ratio

    # Generate the geometric series
    series_factors = np.array([1-r ** i for i in range(num_points)])

    # series * (end - start) + start (working for vectors as well)
    series = np.einsum('..., N->N...', (end - start), series_factors) + np.broadcast_to(start[None,...], (series_factors.shape[0],start.shape[0]))
    if use_approach_from_both_sides:
        new_start = start + 2* (end - start)
        series_other_side = np.einsum('..., N->N...', (end - new_start), series_factors) + np.broadcast_to(new_start[None,...], (series_factors.shape[0],new_start.shape[0]))
        series = np.concatenate((series, np.flip(series_other_side)), axis=0)
    return series

def create_q_grid(vectors:np.ndarray, steps:List[int], amount_BZ=1, include_endpoint=False):
    """ gives a up to 3 dimensional grid spanned by the three vectors given, each direction with the corresponding number of steps."""
    if amount_BZ != 1:
        vectors = [v * amount_BZ for v in vectors]
    if isinstance(steps, int):
        steps = [steps] * len(vectors)
    else:
        if len(steps) != len(vectors):
            raise ValueError("Number of steps must match number of vectors.")
    steps_list = steps
    while len(vectors) < 3:
        vectors.append(np.array([0, 0, 0]))
        steps_list.append(1)
    result = np.zeros((steps_list[0], steps_list[1], steps_list[2], len(vectors[0])))
    # vectors_left=len(vectors)
    spacing = []
    for s in steps_list:
        
        spacing.append(np.linspace(0, 1, s, endpoint= include_endpoint)) #the last point should not be the same as the first if end_periodic is False
        # spacing.append(np.linspace(0, 1-1/s, s)) 
        # spacing.append(np.linspace(0, 1, s))
    #spacing = np.array(spacing)
    # Use broadcasting and einsum to generate the grid efficiently
    grid_shape = (steps_list[0], steps_list[1], steps_list[2], len(vectors[0]))
    # Create meshgrid for the spacing values
    v1, v2, v3 = np.meshgrid(spacing[0], spacing[1], spacing[2], indexing='ij')
    # Stack to shape (..., 3)
    coeffs = np.stack([v1, v2, v3], axis=-1)  # shape: (s0, s1, s2, 3)
    vectors_arr = np.stack(vectors, axis=0)   # shape: (3, d)
    # Use einsum to multiply and sum: coeffs[...,i] * vectors_arr[i,:] -> (..., d)
    result = np.einsum('...i,ij->...j', coeffs, vectors_arr)
    
    return result

def give_reciprocal_lv(lv):
    V = np.dot(lv[0], np.cross(lv[1], lv[2]))
    result = [
        np.array(2 * np.pi / V * np.cross(lv[1], lv[2])),
        np.array(2 * np.pi / V * np.cross(lv[2], lv[0])),
        np.array(2 * np.pi / V * np.cross(lv[0], lv[1])),
    ]
    return result



if __name__ == "__main__":
    from LSW import call_main_with_material_config

    call_main_with_material_config()


def is_approximately_diagonal(matrix, tolerance=1e-3):
    """
    Check whether a matrix is approximately diagonal.

    Parameters:
        matrix (numpy.ndarray): The input matrix.
        tolerance (float): The threshold for considering an element approximately zero.

    Returns:
        bool: True if the matrix is approximately diagonal, False otherwise.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Ensure the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square.")

    # Create a mask for off-diagonal elements
    off_diagonal_mask = ~np.eye(rows, dtype=bool)

    # Extract off-diagonal elements
    off_diagonal_elements = matrix[off_diagonal_mask]

    # Check if all off-diagonal elements are within the tolerance
    return np.all(np.abs(off_diagonal_elements) < tolerance)