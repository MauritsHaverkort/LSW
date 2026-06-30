import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from Material import Material


SPIN_LABELS = ("Sx", "Sy", "Sz", "1")


def newest_material_file():
    """Use the newest saved material as a convenient default for interactive analysis."""
    material_dir = Path("outputs/material")
    files = sorted(material_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            "No material file found in outputs/material. Pass one with --material."
        )
    return files[-1]


def load_material(path: Path) -> "Material":
    """Load the pickled Material object stored by the main LSW workflow."""
    with np.load(path, allow_pickle=True) as data:
        if "material" in data:
            return data["material"].item()
    raise KeyError(f"{path} does not contain a saved 'material' object")


def load_config_dict(path):
    """Accept either a full input JSON file or a bare config dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("config", data)


def prefactor_names(symmetry_class):
    """Return the fundamental XAS prefactor columns expected for a material symmetry."""
    if symmetry_class == "Oh":
        return ["a1g", "eg", "t1u", "t2g"]
    if symmetry_class == "D4h":
        return [
            "a1gB0",
            "a1gA0",
            "a2u1",
            "eu1",
            "b1g2",
            "a1gB2",
            "a1gA2",
            "b2g2",
            "eg2",
        ]
    raise ValueError(f"Unsupported symmetry class for RIXS prefactors: {symmetry_class}")


def load_prefactors(prefactor_file, incident_energies, names):
    """Load complex prefactors at the sampled incident energies used in the RIXS run."""
    data = pd.read_csv(prefactor_file, sep="\t")
    source_energies = np.asarray(data["Energy"])
    source_indices = [int(np.argmin(np.abs(source_energies - e))) for e in incident_energies]
    loaded_energies = source_energies[source_indices]

    prefactors = {}
    for name in names:
        real_key = f"Re {name}"
        imag_key = f"Im {name}"
        if real_key not in data or imag_key not in data:
            prefactors[name] = np.zeros(len(loaded_energies), dtype=np.complex128)
            continue
        prefactors[name] = (
            np.asarray(data[real_key])[source_indices]
            + 1j * np.asarray(data[imag_key])[source_indices]
        )
    return loaded_energies, prefactors


def sigma_prefactors(prefactors):
    """Build sigma_prefactors[s,r,v] = conj(prefactor_s[v]) * prefactor_r[v]."""
    values = list(prefactors.values())
    amount = len(values)
    amount_energies = len(values[0])
    sigma = np.zeros((amount, amount, amount_energies), dtype=np.complex128)
    for s, pref_s in enumerate(values):
        for r, pref_r in enumerate(values):
            sigma[s, r] = np.conj(pref_s) * pref_r
    return sigma


def channel_label(channel):
    """Format a raw O,U,M,N spin-channel index as <O U G M N>."""
    return (
        "<"
        + SPIN_LABELS[channel[0]]
        + SPIN_LABELS[channel[1]]
        + "G"
        + SPIN_LABELS[channel[2]]
        + SPIN_LABELS[channel[3]]
        + ">"
    )


def build_symmetry_channel_map(material: "Material"):
    """Group raw O,U,M,N channels into the same symmetry classes used by the plots."""
    import funtions as functions

    sym_classes_to_xyz_2, _ = material.calc_sym_classes_to_xyz()
    sym_class_2 = functions.give_symmetries_of_tensors(
        sym_classes_to_xyz_2,
        4,
        give_mean_abs_value=False,
        compare_only_a_sample=False,
    )
    class_index = np.asarray(sym_class_2[1], dtype=int)
    labels = {}
    for class_id, symmetry_batch in enumerate(sym_class_2[0]):
        representative = symmetry_batch[1] if len(symmetry_batch) > 1 else symmetry_batch[0]
        labels[class_id] = f"class {class_id}: {material.index_convention_to_string_xyz(representative)}"
    return class_index, labels


def parse_channel(text):
    """Parse CLI channel labels such as 'Sx,Sy,Sz,1' into integer O,U,M,N indices."""
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(
            f"Channel '{text}' must contain four labels/indices, e.g. Sx,Sy,Sz,1."
        )

    mapping = {label.lower(): i for i, label in enumerate(SPIN_LABELS)}
    parsed = []
    for part in parts:
        if part.isdigit():
            parsed.append(int(part))
        else:
            parsed.append(mapping[part.lower()])
    return tuple(parsed)


def axis_values(values, index):
    """Return a readable angle value when available, otherwise fall back to the index."""
    if values is None:
        return index
    values = np.asarray(values)
    if values.ndim == 0 or index >= values.size:
        return index
    return values.reshape(-1)[index]


def config_value(config, key, default=None):
    """Read a config value from either a dataclass/object or a dictionary."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_analysis_config(material, config_path):
    """Prefer the config saved on the material, with JSON as a fallback."""
    material_config = getattr(material, "config", None)
    if material_config is not None:
        return material_config
    return load_config_dict(config_path)


def incident_energies_from_rixs_config(rixs_config):
    """Return incident energies from a dataclass or JSON-style RIXS config."""
    incident_energies = config_value(rixs_config, "incident_energies")
    if incident_energies is not None:
        return np.asarray(incident_energies, dtype=float)

    start = config_value(rixs_config, "incident_energies_start")
    end = config_value(rixs_config, "incident_energies_end")
    amount = config_value(rixs_config, "incident_energies_amount")
    if start is None or end is None or amount is None:
        raise ValueError(
            "RIXS config must provide incident_energies or "
            "incident_energies_start/end/amount."
        )
    return np.linspace(start, end, int(amount))


def analyze_dominant_channels(
    coeff_xyz_rixs,
    sigma,
    incident_energies,
    polarisation_indices=None,
    target_channels=None,
    target_symmetry_classes=None,
    top=20,
    average_angles=False,
    classify="channel",
    best_per_class=False,
    symmetry_channel_map=None,
    symmetry_channel_labels=None,
    alpha=None,
    two_theta=None,
    phi=None,
):
    """Rank where raw spin channels or grouped symmetry channels dominate the RIXS tensor.

    coeff_xyz_rixs axes are R,O,U,M,N,s,r,p,a,b,c.
    After contraction with sigma_prefactors[s,r,v], the analyzed tensor has
    axes v,O,U,M,N,p,a,b,c.
    """
    if coeff_xyz_rixs.ndim != 11:
        raise ValueError(
            "Expected coeff_xyz_rixs with shape R,O,U,M,N,s,r,p,a,b,c; "
            f"got {coeff_xyz_rixs.shape}"
        )

    contracted = np.einsum(
        "ROUMNsrpabc,srv->vOUMNpabc",
        coeff_xyz_rixs,
        sigma,
        optimize="optimal",
    )
    # Dominance is based on absolute contribution of a channel divided by the
    # total absolute weight over all raw O,U,M,N channels at the same parameter.
    weights = np.abs(contracted)
    total_weight = weights.sum(axis=(1, 2, 3, 4))

    if polarisation_indices is not None:
        keep = np.array(polarisation_indices, dtype=int)
        mask = np.zeros(weights.shape[5], dtype=bool)
        mask[keep] = True
    else:
        mask = np.ones(weights.shape[5], dtype=bool)

    if classify == "symmetry":
        if symmetry_channel_map is None:
            raise ValueError("symmetry_channel_map is required for classify='symmetry'")
        class_ids = sorted(int(class_id) for class_id in np.unique(symmetry_channel_map))
    elif classify == "channel":
        class_ids = None
    else:
        raise ValueError("classify must be 'channel' or 'symmetry'")

    rows = []
    if average_angles:
        # Reduce the search space from (v,p,a,b,c) to (v,p) while preserving
        # the same score definition: averaged channel weight / averaged total.
        weights = weights.mean(axis=(6, 7, 8), keepdims=True)
        total_weight = total_weight.mean(axis=(2, 3, 4), keepdims=True)

    for v, p, a, b, c in np.ndindex(
        weights.shape[0],
        weights.shape[5],
        weights.shape[6],
        weights.shape[7],
        weights.shape[8],
    ):
        if not mask[p]:
            continue

        channel_weights = weights[v, :, :, :, :, p, a, b, c]
        total = total_weight[v, p, a, b, c]
        if total == 0:
            continue

        if classify == "symmetry":
            # Sum raw O,U,M,N weights inside each symmetry class before ranking.
            group_weights = np.array(
                [
                    np.sum(channel_weights[symmetry_channel_map == class_id])
                    for class_id in class_ids
                ]
            )
            labels = symmetry_channel_labels or {}

            if best_per_class:
                if target_symmetry_classes:
                    target_indices = [
                        class_ids.index(class_id)
                        for class_id in target_symmetry_classes
                        if class_id in class_ids
                    ]
                else:
                    target_indices = list(range(len(class_ids)))
            elif target_symmetry_classes:
                target_indices = [
                    class_ids.index(class_id)
                    for class_id in target_symmetry_classes
                    if class_id in class_ids
                ]
            else:
                target_indices = [int(np.argmax(group_weights))]

            for group_index in target_indices:
                class_id = class_ids[group_index]
                value = group_weights[group_index]
                rows.append(
                    {
                        "score": float(value / total),
                        "channel_weight": float(value),
                        "total_weight": float(total),
                        "channel": labels.get(class_id, f"class {class_id}"),
                        "channel_index": class_id,
                        "classification": "symmetry",
                        "incident_energy": float(incident_energies[v]),
                        "v_index": v,
                        "polarisation": p,
                        "alpha_index": "avg" if average_angles else a,
                        "two_theta_index": "avg" if average_angles else b,
                        "phi_index": "avg" if average_angles else c,
                        "alpha": "avg" if average_angles else axis_values(alpha, a),
                        "two_theta": "avg"
                        if average_angles
                        else axis_values(two_theta, b),
                        "phi": "avg" if average_angles else axis_values(phi, c),
                    }
                )
        elif target_channels:
            for channel in target_channels:
                value = channel_weights[channel]
                rows.append(
                    {
                        "score": float(value / total),
                        "channel_weight": float(value),
                        "total_weight": float(total),
                        "channel": channel_label(channel),
                        "channel_index": channel,
                        "classification": "channel",
                        "incident_energy": float(incident_energies[v]),
                        "v_index": v,
                        "polarisation": p,
                        "alpha_index": "avg" if average_angles else a,
                        "two_theta_index": "avg" if average_angles else b,
                        "phi_index": "avg" if average_angles else c,
                        "alpha": "avg" if average_angles else axis_values(alpha, a),
                        "two_theta": "avg"
                        if average_angles
                        else axis_values(two_theta, b),
                        "phi": "avg" if average_angles else axis_values(phi, c),
                    }
                )
        else:
            flat_index = int(np.argmax(channel_weights))
            channel = np.unravel_index(flat_index, channel_weights.shape)
            value = channel_weights[channel]
            rows.append(
                {
                    "score": float(value / total),
                    "channel_weight": float(value),
                    "total_weight": float(total),
                    "channel": channel_label(channel),
                    "channel_index": channel,
                    "classification": "channel",
                    "incident_energy": float(incident_energies[v]),
                    "v_index": v,
                    "polarisation": p,
                    "alpha_index": "avg" if average_angles else a,
                    "two_theta_index": "avg" if average_angles else b,
                    "phi_index": "avg" if average_angles else c,
                    "alpha": "avg" if average_angles else axis_values(alpha, a),
                    "two_theta": "avg"
                    if average_angles
                    else axis_values(two_theta, b),
                    "phi": "avg" if average_angles else axis_values(phi, c),
                }
            )

    rows.sort(key=lambda row: row["score"], reverse=True)
    if best_per_class:
        best_rows = {}
        for row in rows:
            key = (row["classification"], row["channel_index"])
            if key not in best_rows:
                best_rows[key] = row
        rows = sorted(
            best_rows.values(),
            key=lambda row: (
                row["channel_index"]
                if isinstance(row["channel_index"], int)
                else str(row["channel_index"])
            ),
        )
    if best_per_class:
        return rows, contracted
    return rows[:top], contracted


def symmetry_class_scores_averaged_angles(
    coeff_xyz_rixs,
    sigma,
    symmetry_channel_map,
    polarisation_indices=None,
    denominator_channels=None,
    denominator_symmetry_classes=None,
):
    """Return score[class, v, p] averaged over alpha/two-theta/phi.

    The score is the averaged absolute weight of one symmetry class divided by the
    averaged total absolute weight over either all raw O,U,M,N channels or a
    selected denominator subset.
    """
    contracted = np.einsum(
        "ROUMNsrpabc,srv->vOUMNpabc",
        coeff_xyz_rixs,
        sigma,
        optimize="optimal",
    )
    weights = np.abs(contracted)
    weights = weights.mean(axis=(6, 7, 8))

    denominator_mask = np.ones(symmetry_channel_map.shape, dtype=bool)
    if denominator_channels:
        denominator_mask = np.zeros(symmetry_channel_map.shape, dtype=bool)
        for channel in denominator_channels:
            denominator_mask[channel] = True
    if denominator_symmetry_classes:
        class_mask = np.isin(symmetry_channel_map, denominator_symmetry_classes)
        denominator_mask = denominator_mask & class_mask
    if not np.any(denominator_mask):
        raise ValueError("The selected score denominator contains no spin channels.")

    total_weight = weights[:, denominator_mask, :].sum(axis=1)

    class_ids = sorted(int(class_id) for class_id in np.unique(symmetry_channel_map))
    scores = np.zeros((len(class_ids), weights.shape[0], weights.shape[5]))
    for class_pos, class_id in enumerate(class_ids):
        class_weight = weights[:, symmetry_channel_map == class_id, :].sum(axis=1)
        scores[class_pos] = np.divide(
            class_weight,
            total_weight,
            out=np.zeros_like(class_weight, dtype=float),
            where=total_weight != 0,
        )

    if polarisation_indices is not None:
        polarisations = list(polarisation_indices)
    else:
        polarisations = list(range(scores.shape[2]))
    return scores, class_ids, polarisations


def plot_symmetry_class_scores_averaged_angles(
    coeff_xyz_rixs,
    sigma,
    incident_energies,
    symmetry_channel_map,
    symmetry_channel_labels=None,
    polarisation_indices=None,
    denominator_channels=None,
    denominator_symmetry_classes=None,
    output_dir=None,
    show=True,
):
    """Plot all symmetry-class scores vs incident energy, one figure per polarisation."""
    import matplotlib.pyplot as plt

    scores, class_ids, polarisations = symmetry_class_scores_averaged_angles(
        coeff_xyz_rixs,
        sigma,
        symmetry_channel_map,
        polarisation_indices=polarisation_indices,
        denominator_channels=denominator_channels,
        denominator_symmetry_classes=denominator_symmetry_classes,
    )
    labels = symmetry_channel_labels or {}
    figures = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for p in polarisations:
        fig, ax = plt.subplots(figsize=(8, 5))
        for class_pos, class_id in enumerate(class_ids):
            ax.plot(
                incident_energies,
                scores[class_pos, :, p],
                label=labels.get(class_id, f"class {class_id}"),
            )

        denominator_label = "selected channels" if (
            denominator_channels or denominator_symmetry_classes
        ) else "all channels"
        ax.set_title(
            f"Symmetry-class dominance scores, polarisation {p}\n"
            f"normalised by {denominator_label}"
        )
        ax.set_xlabel("Incident energy [eV]")
        ax.set_ylabel("Score averaged over angles")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        figures.append(fig)

        if output_dir is not None:
            fig.savefig(
                output_dir / f"symmetry_class_scores_pol_{p}.png",
                dpi=200,
                bbox_inches="tight",
            )

    if show:
        plt.show()
    return figures


def plot_fundamental_xray_spectra(
    incident_energies,
    prefactors,
    xas=None,
    symmetry_class=None,
    output_path=None,
    show=True,
):
    """Plot selected fundamental X-ray prefactor combinations vs incident energy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    if symmetry_class == "D4h" and {"eg2", "eu1"}.issubset(prefactors):
        eg2 = prefactors["eg2"]
        eu1 = prefactors["eu1"]
        curves = [
            (r"||$\sigma_{eg2}$||^2", np.abs(eg2) ** 2),
            (r"||$\sigma_{eu1}$||^2", np.abs(eu1) ** 2),
            (r"Im($\sigma_{eu1}^* \cdot \sigma_{eg2}$)", np.imag(np.conj(eu1) * eg2)),
            (r"-Im($\sigma_{eu1}^* \cdot \sigma_{eg2}$)", -np.imag(np.conj(eu1) * eg2)),
        ]
    elif symmetry_class == "Oh" and {"eg", "t1u", "t2g"}.issubset(prefactors):
        eg = prefactors["eg"]
        t1u = prefactors["t1u"]
        t2g = prefactors["t2g"]
        curves = [
            (r"||$\sigma_{eg}$||^2", np.abs(eg) ** 2),
            (r"||$\sigma_{t1u}$||^2", np.abs(t1u) ** 2),
            (r"||$\sigma_{t2g}$||^2", np.abs(t2g) ** 2),
            (r"Im($\sigma_{t2g}^* \cdot \sigma_{t1u}$)", np.imag(np.conj(t2g) * t1u)),
        ]
    else:
        curves = [
            (rf"||$\sigma_{{{name}}}$||^2", np.abs(value) ** 2)
            for name, value in prefactors.items()
        ]

    for label, values in curves:
        ax.plot(incident_energies, values, label=label)

    if xas is not None:
        xas = np.asarray(xas)
        max_curve = max(np.max(np.abs(values)) for _, values in curves)
        max_xas = np.max(np.abs(xas))
        scale = max_curve / max_xas if max_xas > 0 else 1
        ax.plot(
            incident_energies,
            np.abs(xas) * scale,
            color="gray",
            linestyle="--",
            label="XAS",
        )

    ax.set_xlabel("Incident energy [eV]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def _validate_rixs_spectrum(name, spectrum, incident_energies):
    """Check the saved RIXS spectrum convention used by LSW.py."""
    if spectrum is None:
        raise ValueError(f"Material does not contain {name}. Re-run with RIXS spectra saved.")
    if spectrum.ndim != 7:
        raise ValueError(
            f"Expected {name} with axes R,p,v,w,a,b,c; got shape {spectrum.shape}"
        )
    if spectrum.shape[2] != len(incident_energies):
        raise ValueError(
            f"{name} incident-energy axis has length {spectrum.shape[2]}, "
            f"but {len(incident_energies)} incident energies were loaded."
        )
    spectrum = np.asarray(spectrum)
    if np.iscomplexobj(spectrum):
        max_real = np.max(np.abs(np.real(spectrum)))
        max_imag = np.max(np.abs(np.imag(spectrum)))
        if max_imag > 1e-5 * max(max_real, 1.0):
            print(
                f"Warning: {name} has a non-negligible imaginary component "
                f"(max imag={max_imag:.6g}, max real={max_real:.6g}); plotting real part."
            )
        return np.real(spectrum)
    return spectrum


def magnon_channel_curves_by_polarisation(spectrum, polarisation_indices):
    """Integrate RIXS spectrum over energy loss and average over domains/angles.

    Input axes are R,p,v,w,a,b,c. The result for each polarisation keeps the
    incident-energy axis v and reduces the angle grid a,b,c to mean/std/min/max
    curves after summing over transferred energy w and averaging over R.
    """
    curves = {}
    for p in polarisation_indices:
        if p < 0 or p >= spectrum.shape[1]:
            raise IndexError(
                f"Polarisation index {p} is outside the available range "
                f"0..{spectrum.shape[1] - 1}."
            )
        vabc = spectrum[:, p].sum(axis=2).mean(axis=0)
        angle_axes = tuple(range(1, vabc.ndim))
        curves[p] = {
            "mean": vabc.mean(axis=angle_axes),
            "std": vabc.std(axis=angle_axes),
            "min": vabc.min(axis=angle_axes),
            "max": vabc.max(axis=angle_axes),
        }
    return curves


def plot_one_two_magnon_channel_comparison(
    incident_energies,
    one_magnon_spectrum,
    two_magnon_spectrum,
    polarisation_indices=None,
    output_dir=None,
    show=True,
):
    """Plot integrated one- and two-magnon RIXS channels vs incident energy."""
    import matplotlib.pyplot as plt

    one_magnon_spectrum = _validate_rixs_spectrum(
        "rixs_spectrum_2", one_magnon_spectrum, incident_energies
    )
    two_magnon_spectrum = _validate_rixs_spectrum(
        "rixs_spectrum_4", two_magnon_spectrum, incident_energies
    )
    if one_magnon_spectrum.shape[1] != two_magnon_spectrum.shape[1]:
        raise ValueError(
            "One- and two-magnon spectra have different polarisation axes: "
            f"{one_magnon_spectrum.shape[1]} vs {two_magnon_spectrum.shape[1]}."
        )

    if polarisation_indices is None:
        polarisation_indices = list(range(one_magnon_spectrum.shape[1]))
    else:
        polarisation_indices = list(polarisation_indices)

    one_curves = magnon_channel_curves_by_polarisation(
        one_magnon_spectrum, polarisation_indices
    )
    two_curves = magnon_channel_curves_by_polarisation(
        two_magnon_spectrum, polarisation_indices
    )
    figures = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for p in polarisation_indices:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        one = one_curves[p]
        two = two_curves[p]

        ax.plot(
            incident_energies,
            one["mean"],
            color="#1f77b4",
            linewidth=2.3,
            label="one magnon",
        )
        ax.plot(
            incident_energies,
            two["mean"],
            color="#d62728",
            linewidth=2.3,
            label="two magnon",
        )
        ax.fill_between(
            incident_energies,
            one["mean"] - one["std"],
            one["mean"] + one["std"],
            color="#1f77b4",
            alpha=0.18,
            linewidth=0,
            label="one magnon angle std",
        )
        ax.fill_between(
            incident_energies,
            two["mean"] - two["std"],
            two["mean"] + two["std"],
            color="#d62728",
            alpha=0.18,
            linewidth=0,
            label="two magnon angle std",
        )
        ax.plot(
            incident_energies,
            one["min"],
            color="#1f77b4",
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
        )
        ax.plot(
            incident_energies,
            one["max"],
            color="#1f77b4",
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
        )
        ax.plot(
            incident_energies,
            two["min"],
            color="#d62728",
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
        )
        ax.plot(
            incident_energies,
            two["max"],
            color="#d62728",
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
        )
        ax.set_title(f"One- and two-magnon RIXS channels, polarisation {p}")
        ax.set_xlabel("Incident energy [eV]")
        ax.set_ylabel("Integrated intensity [a.u.]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures.append(fig)

        if output_dir is not None:
            fig.savefig(
                output_dir / f"one_two_magnon_channels_pol_{p}.png",
                dpi=200,
                bbox_inches="tight",
            )

    if show:
        plt.show()
    return figures


def print_rows(rows):
    """Print a compact terminal table for the ranked parameter sets."""
    header = (
        "rank  score     channel           E_in[eV]  pol  "
        "alpha_idx two_theta_idx phi_idx  channel_weight"
    )
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4}  {row['score']:.5f}  {row['channel']:<16}  "
            f"{row['incident_energy']:>8.3f}  {row['polarisation']:>3}  "
            f"{row['alpha_index']:>9} {row['two_theta_index']:>13} "
            f"{row['phi_index']:>7}  {row['channel_weight']:.6g}"
        )


def main():
    """CLI entry point for scanning saved RIXS tensors."""
    parser = argparse.ArgumentParser(
        description=(
            "Find incident-energy and polarisation points where one spin channel "
            "dominates the RIXS tensor."
        )
    )
    parser.add_argument("--material", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--prefactors", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--polarisations",
        type=int,
        nargs="*",
        default=None,
        help="Polarisation indices to include. Defaults to config.rixs_config.polarisations_to_sum.",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help="Target channel as four labels/indices, e.g. --channel Sx,Sy,Sz,1. Repeatable.",
    )
    parser.add_argument(
        "--classify",
        choices=("channel", "symmetry"),
        default="channel",
        help="Rank raw O,U,M,N channels or grouped symmetry-channel classes.",
    )
    parser.add_argument(
        "--symmetry-class",
        type=int,
        action="append",
        default=None,
        help="Target symmetry-channel class id. Repeatable; only used with --classify symmetry.",
    )
    parser.add_argument(
        "--best-per-class",
        action="store_true",
        help="Report the best parameter set for each symmetry class/requested channel instead of global winners.",
    )
    parser.add_argument(
        "--plot-symmetry-scores",
        action="store_true",
        help="Plot all symmetry-class scores vs incident energy, averaged over angles.",
    )
    parser.add_argument(
        "--score-denominator-channel",
        action="append",
        default=None,
        help=(
            "Raw O,U,M,N channel included in the plot-score denominator, "
            "e.g. Sx,Sz,Sz,Sy. Repeatable."
        ),
    )
    parser.add_argument(
        "--score-denominator-symmetry-class",
        type=int,
        action="append",
        default=None,
        help=(
            "Symmetry class id included in the plot-score denominator. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--plot-output-dir",
        type=Path,
        default=None,
        help="Optional directory for saving generated PNG plots.",
    )
    parser.add_argument(
        "--plot-fund-xray",
        action="store_true",
        help="Plot fundamental X-ray prefactor combinations vs incident energy.",
    )
    parser.add_argument(
        "--plot-magnon-channel-comparison",
        action="store_true",
        help=(
            "Plot integrated one- and two-magnon RIXS channels vs incident energy "
            "for each polarisation. Defaults to all saved polarisations unless "
            "--polarisations is supplied."
        ),
    )
    parser.add_argument(
        "--fund-xray-output",
        type=Path,
        default=None,
        help="Optional PNG path for the fundamental X-ray spectra plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows; useful together with --plot-output-dir.",
    )
    parser.add_argument(
        "--average-angles",
        action="store_true",
        help="Average over alpha/two-theta/phi and rank only incident energy plus polarisation.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    material_path = args.material or newest_material_file()
    material = load_material(material_path)

    config = load_analysis_config(material, args.config)
    rixs_config = config_value(config, "rixs_config", {})
    incident_energies = incident_energies_from_rixs_config(rixs_config)
    polarisations = (
        args.polarisations
        if args.polarisations is not None
        else config_value(rixs_config, "polarisations_to_sum")
    )
    prefactor_file = (
        args.prefactors
        or getattr(material, "fund_xray_prefactors_filename", None)
        or Path("input_data/prefactors_results_ligand_field.dat")
    )
    names = prefactor_names(material.symmetry_class)
    loaded_energies, prefactors = load_prefactors(prefactor_file, incident_energies, names)
    sigma = sigma_prefactors(prefactors)
    symmetry_channel_map = None
    symmetry_channel_labels = None
    if args.classify == "symmetry" or args.plot_symmetry_scores:
        # The class ids are generated from the material's own symmetry grouping,
        # so they match the labels used in the existing RIXS diagnostic plots.
        symmetry_channel_map, symmetry_channel_labels = build_symmetry_channel_map(material)

    target_channels = [parse_channel(channel) for channel in args.channel or []]
    denominator_channels = [
        parse_channel(channel) for channel in args.score_denominator_channel or []
    ]
    rows, _ = analyze_dominant_channels(
        material.coeff_xyz_rixs,
        sigma,
        loaded_energies,
        polarisation_indices=polarisations,
        target_channels=target_channels,
        target_symmetry_classes=args.symmetry_class,
        top=args.top,
        average_angles=args.average_angles,
        classify=args.classify,
        best_per_class=args.best_per_class,
        symmetry_channel_map=symmetry_channel_map,
        symmetry_channel_labels=symmetry_channel_labels,
        alpha=config_value(rixs_config, "alpha"),
        two_theta=config_value(rixs_config, "two_theta"),
        phi=config_value(rixs_config, "phi"),
    )

    print(f"Material: {material_path}")
    print(
        "Material type: "
        f"{type(material).__name__}, name={getattr(material, 'name', 'unknown')}, "
        f"symmetry={getattr(material, 'symmetry_class', 'unknown')}"
    )
    print(f"Prefactors: {prefactor_file}")
    print(f"Polarisations: {polarisations}")
    print_rows(rows)

    if args.plot_fund_xray:
        plot_fundamental_xray_spectra(
            loaded_energies,
            prefactors,
            xas=getattr(material, "others", {}).get("XAS"),
            symmetry_class=material.symmetry_class,
            output_path=args.fund_xray_output,
            show=not args.no_show,
        )

    if args.plot_magnon_channel_comparison:
        comparison_polarisations = (
            args.polarisations
            if args.polarisations is not None
            else list(range(material.rixs_spectrum_2.shape[1]))
        )
        plot_one_two_magnon_channel_comparison(
            loaded_energies,
            material.rixs_spectrum_2,
            material.rixs_spectrum_4,
            polarisation_indices=comparison_polarisations,
            output_dir=args.plot_output_dir,
            show=not args.no_show,
        )

    if args.plot_symmetry_scores:
        plot_symmetry_class_scores_averaged_angles(
            material.coeff_xyz_rixs,
            sigma,
            loaded_energies,
            symmetry_channel_map,
            symmetry_channel_labels=symmetry_channel_labels,
            polarisation_indices=polarisations,
            denominator_channels=denominator_channels,
            denominator_symmetry_classes=args.score_denominator_symmetry_class,
            output_dir=args.plot_output_dir,
            show=not args.no_show,
        )

    if args.save is not None:
        import csv

        with open(args.save, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "score",
                "channel_weight",
                "total_weight",
                "channel",
                "channel_index",
                "classification",
                "incident_energy",
                "v_index",
                "polarisation",
                "alpha_index",
                "two_theta_index",
                "phi_index",
                "alpha",
                "two_theta",
                "phi",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {args.save}")


if __name__ == "__main__":
    main()
