from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy


DATA_PATH = Path(__file__).with_name("SNe data.csv")
PLOT_OUTPUT_PATH = Path(__file__).with_name("modulus_vs_redshift.pdf")
LOW_Z_PLOT_OUTPUT_PATH = Path(__file__).with_name(
    "distance_vs_redshift_low_z.pdf"
)
LOW_Z_THRESHOLD = 0.03
PLOT_ENTRY_LIMIT = 600


def load_modulus_data(csv_path: Path = DATA_PATH) -> np.ndarray:
    """Load the supernova catalog into a structured numpy array."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    return np.genfromtxt(
        csv_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        autostrip=True,
    )


def modulus_to_distance(
    modulus: np.ndarray, modulus_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert modulus (and its error) to distance and propagated error."""
    distance = np.power(10.0, (modulus / 5.0) +1.0)
    distance_error = (np.log(10.0) / 5.0) * distance * modulus_error
    return distance, distance_error


def plot_modulus_vs_redshift(
    redshift: np.ndarray,
    modulus: np.ndarray,
    modulus_error: np.ndarray,
    output_path: Path,
) -> None:
    """Plot distance modulus (with uncertainties) vs redshift."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        redshift,
        modulus,
        yerr=modulus_error,
        fmt="o",
        markersize=4,
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
    )
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance Modulus (mag)")
    ax.set_title(
        f"Distance Modulus vs. Redshift (first {len(redshift)} entries)"
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_distance_vs_redshift(
    redshift: np.ndarray,
    distance: np.ndarray,
    distance_error: np.ndarray,
    output_path: Path,
) -> None:
    """Plot physical distance (with uncertainties) vs redshift."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        redshift,
        distance,
        yerr=distance_error,
        fmt="o",
        markersize=4,
        color="tab:orange",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
    )
    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance (pc)")
    ax.set_title(
        f"Distance vs. Redshift (subset of {len(redshift)} low-z entries)"
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main() -> None:
    data = load_modulus_data()
    subset = data[: min(len(data), PLOT_ENTRY_LIMIT)]
    if len(subset) == 0:
        raise RuntimeError("No supernova data available to process.")

    distances, distance_errors = modulus_to_distance(
        subset["Distance_Modulus"],
        subset["Distance_Modulus_Error"],
    )
    redshift = subset["Redshift"]
    modulus = subset["Distance_Modulus"]
    modulus_error = subset["Distance_Modulus_Error"]

    print(f"Loaded {len(data)} supernovae using SciPy {scipy.__version__}.")
    print(f"Plotting first {len(subset)} entries to {PLOT_OUTPUT_PATH.name}.")
    print("First entry as a quick check:")
    print(
        f"{subset['Supernova_Name'][0]} | "
        f"μ = {subset['Distance_Modulus'][0]:.3f} ± {subset['Distance_Modulus_Error'][0]:.3f} -> "
        f"d = {distances[0]:.3e} ± {distance_errors[0]:.3e} pc"
    )
    plot_modulus_vs_redshift(
        redshift,
        modulus,
        modulus_error,
        PLOT_OUTPUT_PATH,
    )
    print(f"Saved plot to {PLOT_OUTPUT_PATH.resolve()}")

    low_z_mask = redshift < LOW_Z_THRESHOLD
    if np.any(low_z_mask):
        plot_distance_vs_redshift(
            redshift[low_z_mask],
            distances[low_z_mask],
            distance_errors[low_z_mask],
            LOW_Z_PLOT_OUTPUT_PATH,
        )
        print(
            f"Saved low-z (< {LOW_Z_THRESHOLD}) distance plot to "
            f"{LOW_Z_PLOT_OUTPUT_PATH.resolve()}"
        )
    else:
        print(
            f"No entries below redshift {LOW_Z_THRESHOLD}; "
            "skipping low-z plot."
        )


if __name__ == "__main__":
    main()

