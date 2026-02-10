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
    """this is used to estimate output size without performing the contraction"""
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
