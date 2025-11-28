from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy


DATA_PATH = Path(__file__).with_name("SNe data.csv")
PLOT_OUTPUT_PATH = Path(__file__).with_name("modulus_vs_log_redshift.pdf")
PLOT_ENTRY_LIMIT = 100


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


def plot_modulus_vs_log_redshift(
    redshift: np.ndarray,
    modulus: np.ndarray,
    modulus_error: np.ndarray,
    output_path: Path,
) -> None:
    """Plot distance modulus (with uncertainties) vs log10(redshift)."""
    positive_mask = redshift > 0.0
    if not np.any(positive_mask):
        raise ValueError("No positive redshift values available for logarithm.")

    log_redshift = np.log10(redshift[positive_mask])
    filtered_modulus = modulus[positive_mask]
    filtered_errors = modulus_error[positive_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        log_redshift,
        filtered_modulus,
        yerr=filtered_errors,
        fmt="o",
        markersize=4,
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
    )
    ax.set_xlabel("log10(Redshift)")
    ax.set_ylabel("Distance Modulus (mag)")
    ax.set_title(f"Distance Modulus vs. log10(Redshift) (first {len(log_redshift)} entries)")
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
    plot_modulus_vs_log_redshift(
        redshift,
        modulus,
        modulus_error,
        PLOT_OUTPUT_PATH,
    )
    print(f"Saved plot to {PLOT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

