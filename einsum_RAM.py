from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sys import path
from typing import Any, Optional, Sequence, Tuple, Union
import uuid
from matplotlib import path
from Configurations import Config as Config

import numpy as np
import opt_einsum as oe

try:
    import psutil
except ImportError:
    psutil = None

memmap_basename = "oe_out"

@dataclass
class ContractResult:
    array: np.ndarray  # either ndarray or np.memmap
    used_memmap: bool
    memmap_path: Optional[str]
    expr: str
    output_shape: Tuple[int, ...]
    dtype: np.dtype


def _parse_output_subscripts(expr: str) -> str:
    # expr format: "ab,bc->ac" or "ab,bc"
    if "->" in expr:
        return expr.split("->", 1)[1].strip()
    # implicit output: Einstein summation convention = sorted unique indices that appear once
    inputs = expr.split("->", 1)[0]
    in_terms = [t.strip() for t in inputs.split(",")]
    counts = {}
    for term in in_terms:
        for ch in term:
            if ch == " ":
                continue
            counts[ch] = counts.get(ch, 0) + 1
    out = "".join([k for k in sorted(counts.keys()) if counts[k] == 1])
    return out


def _infer_output_shape(expr: str, operands: Sequence[np.ndarray]) -> Tuple[int, ...]:
    """this is used to estimate output size without performing the contraction (from the size of the input arrays and the subscripts)"""
    in_part = expr.split("->", 1)[0]
    in_terms = [t.strip() for t in in_part.split(",")]
    if len(in_terms) != len(operands):
        raise ValueError(f"Expression has {len(in_terms)} operands but got {len(operands)} arrays.")

    out_sub = _parse_output_subscripts(expr)
    dim_map: dict[str, int] = {}

    for term, arr in zip(in_terms, operands):
        term = term.replace(" ", "")
        if arr.ndim != len(term):
            raise ValueError(f"Operand with shape {arr.shape} does not match subscripts '{term}'.")
        for ch, dim in zip(term, arr.shape):
            if ch in dim_map and dim_map[ch] != dim:
                raise ValueError(f"Index '{ch}' has inconsistent dims: {dim_map[ch]} vs {dim}.")
            dim_map[ch] = dim

    return tuple(dim_map[ch] for ch in out_sub)

# experimental to capture ... 
def _infer_output_shape_ellipsis(expr: str, operands: Sequence[np.ndarray]) -> Tuple[int, ...]:
    """
    Estimate einsum output shape without performing the contraction.
    Supports ellipsis ("...") in input and output subscripts.

    Notes
    -----
    - Handles explicit output form: "ab...,bc->ac..."
    - Handles implicit output form (no '->'): follows NumPy einsum rule:
      output indices are the ellipsis (if present) plus indices that appear exactly once.
    - Assumes valid einsum expression; will raise on inconsistent dimensions.
    """
    expr = expr.replace(" ", "")
    if "->" in expr:
        in_part, out_sub = expr.split("->", 1)
    else:
        in_part, out_sub = expr, None

    in_terms = in_part.split(",")
    if len(in_terms) != len(operands):
        raise ValueError(
            f"Expression has {len(in_terms)} operands but got {len(operands)} arrays."
        )

    # ---- Determine how many axes the ellipsis stands for in each operand ----
    # term_len_no_ellipsis counts explicit labels (single letters)
    ell_lens = []
    for term, arr in zip(in_terms, operands):
        if "..." in term:
            term_wo = term.replace("...", "")
            n_ell = arr.ndim - len(term_wo)
            if n_ell < 0:
                raise ValueError(
                    f"Operand with shape {arr.shape} does not match subscripts '{term}'."
                )
        else:
            if arr.ndim != len(term):
                raise ValueError(
                    f"Operand with shape {arr.shape} does not match subscripts '{term}'."
                )
            n_ell = 0
        ell_lens.append(n_ell)

    max_ell = max(ell_lens) if ell_lens else 0

    # ---- Build a canonical ellipsis label sequence: E0, E1, ... ----
    ell_labels = [f"§E{m}§" for m in range(max_ell)]  # sentinel labels unlikely to collide

    # ---- Expand each term by replacing "..." with the rightmost portion of ell_labels ----
    # Numpy aligns ellipsis dimensions to the left; effectively, shorter ellipses are padded on the left.
    expanded_terms = []
    for term, n_ell in zip(in_terms, ell_lens):
        if "..." in term:
            # pad on the left so that the *rightmost* ellipsis axes line up
            pad = max_ell - n_ell
            replacement = "".join(ell_labels[pad:])
            expanded = term.replace("...", replacement)
        else:
            expanded = term
        expanded_terms.append(expanded)

    # ---- Infer output subscripts if implicit ----
    if out_sub is None:
        # Count each label occurrence (including ellipsis labels)
        counts: dict[str, int] = {}
        for term in expanded_terms:
            for ch in term:
                counts[ch] = counts.get(ch, 0) + 1

        # If any operand had ellipsis, numpy includes ellipsis in output by default
        has_ellipsis = max_ell > 0
        out_labels = []
        if has_ellipsis:
            out_labels.extend(ell_labels)

        # Then labels that appear exactly once, in sorted order for determinism
        singles = sorted([k for k, v in counts.items() if v == 1 and k not in ell_labels])
        out_labels.extend(singles)
    else:
        # Expand output ellipsis if present
        if "..." in out_sub:
            replacement = "".join(ell_labels)
            out_expanded = out_sub.replace("...", replacement)
        else:
            out_expanded = out_sub
        out_labels = list(out_expanded)

    # ---- Map labels to dimensions and validate consistency ----
    dim_map: dict[str, int] = {}

    for term, arr, n_ell in zip(expanded_terms, operands, ell_lens):
        # term now has length == arr.ndim
        if len(term) != arr.ndim:
            raise ValueError(
                f"Internal error: expanded subscripts '{term}' do not match ndim={arr.ndim}."
            )
        for ch, dim in zip(term, arr.shape):
            if ch in dim_map and dim_map[ch] != dim:
                raise ValueError(
                    f"Index '{ch}' has inconsistent dims: {dim_map[ch]} vs {dim}."
                )
            dim_map[ch] = dim

    # ---- Construct output shape ----
    try:
        return tuple(dim_map[ch] for ch in out_labels)
    except KeyError as e:
        raise ValueError(f"Output index '{e.args[0]}' not present in any input.") from None

def _available_ram_bytes() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None

def clean_memmap_files(config: Config) -> None:
    """Delete the memmap file at the given path, if it exists."""
    memmap_dir = Path("intermediate_arrays")
    if memmap_dir.exists() and memmap_dir.is_dir():
        for file in memmap_dir.glob(f"{memmap_basename}_{config.timestamp}_*.dat"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Could not delete memmap file {file}: {e}")

def contract_autostore(
    expr: str,
    *operands: np.ndarray,
    optimize: Union[str, bool, Any] = "optimal",
    config:Config,
    dtype: Optional[np.dtype] = None,
    memmap_dir: Union[str, Path] = "intermediate_arrays",
    
    memmap_mode: str = "w+",
    ram_safety_factor: float = 0.50,
    force_memmap: bool = False,
    return_info: bool = False,
) -> Union[np.ndarray, ContractResult]:
    """
    Like opt_einsum.contract(expr, *operands, optimize=...), but chooses a memmap output
    if the estimated output size is too large compared to available RAM.

    Parameters
    ----------
    expr : einsum expression, e.g. "ab,bc->ac"
    operands : numpy arrays
    optimize : opt_einsum optimize argument ("optimal", "greedy", path, ...)
    dtype : if provided, output dtype (else numpy result-type of operands)
    memmap_dir : directory for memmap file
    memmap_basename : base filename (suffix .dat is appended)
    memmap_mode : memmap file mode ("w+", "r+", ...)
    ram_safety_factor : use memmap if output_bytes > ram_safety_factor * available_ram
    force_memmap : always use memmap output
    return_info : if True, return ContractResult with metadata

    Returns
    -------
    out : np.ndarray or np.memmap (both are array-like and support further numpy ops)
    """
    ops = [np.asarray(op) for op in operands]
    out_dtype = np.dtype(dtype) if dtype is not None else np.result_type(*ops)

    out_shape = _infer_output_shape(expr, ops)
    out_bytes = int(np.prod(out_shape, dtype=np.int64)) * out_dtype.itemsize

    avail = _available_ram_bytes()
    use_memmap = force_memmap
    if not use_memmap and avail is not None:
        use_memmap = out_bytes > int(ram_safety_factor * avail)

    memmap_path = None
    if use_memmap:
        memmap_dir = Path(memmap_dir)
        memmap_dir.mkdir(parents=True, exist_ok=True)
        memmap_path = str(memmap_dir / f"{memmap_basename}_{config.timestamp}_{uuid.uuid4().hex}") + ".dat"
        out = np.memmap(memmap_path, dtype=out_dtype, mode=memmap_mode, shape=out_shape)
        # Important: fill with zeros only if you depend on it (w+ gives uninitialized file sometimes)
        # out[:] = 0
        out_arr: np.ndarray = out
        res = oe.contract(expr, *ops, optimize=optimize, out=out_arr)
        # oe.contract returns the same object when out is provided
        out_arr.flush() if isinstance(out_arr, np.memmap) else None
    else:
        res = oe.contract(expr, *ops, optimize=optimize)
        if res.dtype != out_dtype:
            res = res.astype(out_dtype, copy=False)

    if return_info:
        return ContractResult(
            array=res,
            used_memmap=use_memmap,
            memmap_path=memmap_path,
            expr=expr,
            output_shape=out_shape,
            dtype=out_dtype,
        )
    return res
