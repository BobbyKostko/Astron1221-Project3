from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate, optimize


DATA_PATH = Path(__file__).with_name("SNe data.csv")
PLOT_OUTPUT_PATH = Path(__file__).with_name("modulus_vs_redshift.pdf")
COSMO_MODULUS_PLOT_OUTPUT_PATH = Path(__file__).with_name(
    "distance_modulus_vs_redshift_fit.pdf"
)
LOW_Z_PLOT_OUTPUT_PATH = Path(__file__).with_name(
    "distance_vs_redshift_low_z.pdf"
)
LOW_Z_THRESHOLD = 0.03
PLOT_ENTRY_LIMIT = 600
SPEED_OF_LIGHT_KM_S = 299_792.458
PARSECS_PER_MEGAPARSEC = 1_000_000.0
DEFAULT_H0_KM_S_MPC = 70.0
DEFAULT_H0_PC_INV = DEFAULT_H0_KM_S_MPC / (
    SPEED_OF_LIGHT_KM_S * PARSECS_PER_MEGAPARSEC
)


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
    distance = np.power(10.0, (modulus / 5.0) + 1.0)
    distance_error = (np.log(10.0) / 5.0) * distance * modulus_error
    return distance, distance_error


def distance_to_modulus(distance: np.ndarray) -> np.ndarray:
    """Convert distances (pc) back to distance modulus."""
    distance = np.asarray(distance, dtype=float)
    if np.any(distance <= 0):
        raise ValueError("Distance must be positive to compute modulus.")
    return 5.0 * (np.log10(distance) - 1.0)


def hubble_law(redshift: np.ndarray, hubble_constant: float) -> np.ndarray:
    """Simple linear relation D = z / H0 as requested."""
    return redshift / hubble_constant


def fit_hubble_constant(
    redshift: np.ndarray,
    distance: np.ndarray,
    distance_error: np.ndarray,
) -> tuple[float, float]:
    """Fit the Hubble-like constant using scipy.optimize.curve_fit."""
    mask = redshift > 0.0
    if not np.any(mask):
        raise ValueError("Need positive redshift values to fit H0.")

    z = redshift[mask]
    d = distance[mask]
    sigma = distance_error[mask]

    ratio = np.mean(z / d)
    initial_guess = ratio if np.isfinite(ratio) and ratio > 0 else 1.0

    popt, pcov = optimize.curve_fit(
        hubble_law,
        z,
        d,
        sigma=sigma,
        absolute_sigma=True,
        p0=(initial_guess,),
        maxfev=10000,
    )
    h0 = popt[0]
    h0_err = float(np.sqrt(np.diag(pcov))[0]) if pcov.size else float("nan")
    return h0, h0_err


def luminosity_distance_flat(
    redshift: float,
    omega_m: float,
    omega_lambda: float,
    hubble_constant_pc_inv: float,
) -> float:
    """Evaluate d_L(z) for a flat Universe with matter and dark energy only."""
    if hubble_constant_pc_inv <= 0:
        raise ValueError("H0 must be positive for luminosity distance.")
    if redshift < 0:
        raise ValueError("Redshift must be non-negative.")
    if omega_m + omega_lambda <= 0:
        raise ValueError("Ω_M + Ω_Λ must be positive.")

    def integrand(z_prime: float) -> float:
        return 1.0 / np.sqrt(omega_m * (1.0 + z_prime) ** 3 + omega_lambda)

    integral, _ = integrate.quad(
        integrand,
        0.0,
        float(redshift),
        epsabs=1e-8,
        epsrel=1e-8,
        limit=200,
    )
    return (1.0 + redshift) * integral / hubble_constant_pc_inv


def predict_luminosity_distance(
    redshift: np.ndarray,
    omega_m: float,
    omega_lambda: float,
    hubble_constant_pc_inv: float,
) -> np.ndarray:
    """Vectorized helper that maps an array of redshifts to d_L(z)."""
    redshift = np.asarray(redshift, dtype=float)
    distances = np.empty_like(redshift)
    for idx, z in enumerate(redshift):
        distances[idx] = luminosity_distance_flat(
            z, omega_m, omega_lambda, hubble_constant_pc_inv
        )
    return distances


def fit_density_parameters(
    redshift: np.ndarray,
    distance: np.ndarray,
    distance_error: np.ndarray,
    hubble_constant_pc_inv: float,
) -> tuple[float, float, optimize.OptimizeResult]:
    """Fit Ω_M and Ω_Λ by minimizing residuals between data and d_L(z)."""
    if hubble_constant_pc_inv is None or hubble_constant_pc_inv <= 0:
        raise ValueError("Need a positive H0 (in pc^-1) to fit densities.")

    mask = (
        np.isfinite(redshift)
        & np.isfinite(distance)
        & (redshift >= 0.0)
        & (distance > 0.0)
    )
    if not np.any(mask):
        raise ValueError("No finite redshift/distance pairs available.")

    z = redshift[mask]
    d = distance[mask]
    sigma = distance_error[mask]
    finite_sigma = sigma[(sigma > 0) & np.isfinite(sigma)]
    fallback_sigma = np.median(finite_sigma) if finite_sigma.size else 1.0
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 1.0
    sigma = np.where(
        (sigma > 0) & np.isfinite(sigma),
        sigma,
        fallback_sigma,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        omega_m, omega_lambda = params
        if omega_m + omega_lambda <= 0:
            return np.full_like(z, 1e9, dtype=float)
        try:
            model = predict_luminosity_distance(
                z,
                omega_m,
                omega_lambda,
                hubble_constant_pc_inv,
            )
        except ValueError:
            return np.full_like(z, 1e9, dtype=float)
        return (model - d) / sigma

    initial_guess = np.array([0.3, 0.7])
    bounds = (
        np.array([1e-4, 1e-4]),
        np.array([3.0, 3.0]),
    )
    result = optimize.least_squares(
        residuals,
        x0=initial_guess,
        bounds=bounds,
        max_nfev=200,
    )
    if not result.success:
        raise RuntimeError(f"Cosmological fit did not converge: {result.message}")
    omega_m_fit, omega_lambda_fit = result.x
    return omega_m_fit, omega_lambda_fit, result


def to_km_s_per_mpc(
    h0_pc_inv: float, h0_pc_inv_error: float | None
) -> tuple[float, float | None]:
    """Convert from pc⁻¹ units to the conventional km s⁻¹ Mpc⁻¹."""
    factor = SPEED_OF_LIGHT_KM_S * PARSECS_PER_MEGAPARSEC
    converted = h0_pc_inv * factor
    converted_err = (
        h0_pc_inv_error * factor if h0_pc_inv_error is not None else None
    )
    return converted, converted_err


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
    best_fit_h0: float | None = None,
    best_fit_h0_error: float | None = None,
    best_fit_h0_km_s_mpc: float | None = None,
    best_fit_h0_km_s_mpc_error: float | None = None,
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

    if best_fit_h0 is not None and len(redshift) > 1:
        z_grid = np.linspace(np.min(redshift), np.max(redshift), 200)
        d_grid = hubble_law(z_grid, best_fit_h0)
        label_lines = [r"$D = z / H_0$ fit"]
        if best_fit_h0_km_s_mpc is not None:
            km_line = (
                f"H0 = {best_fit_h0_km_s_mpc:.2f} km s$^{{-1}}$ Mpc$^{{-1}}$"
            )
            if (
                best_fit_h0_km_s_mpc_error is not None
                and np.isfinite(best_fit_h0_km_s_mpc_error)
            ):
                km_line += f" ± {best_fit_h0_km_s_mpc_error:.2f}"
            label_lines.append(km_line)
        else:
            label_lines.append(f"H0 = {best_fit_h0:.3e} pc$^{{-1}}$")

        ax.plot(z_grid, d_grid, color="black", linestyle="--", label="\n".join(label_lines))
        ax.legend()

    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance (pc)")
    ax.set_title(
        f"Distance vs. Redshift (subset of {len(redshift)} low-z entries)"
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_cosmological_modulus_fit(
    redshift: np.ndarray,
    modulus: np.ndarray,
    modulus_error: np.ndarray,
    omega_m: float,
    omega_lambda: float,
    hubble_constant_pc_inv: float,
    output_path: Path,
) -> None:
    """Plot distance modulus with best-fit Ω parameters over the data."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        redshift,
        modulus,
        yerr=modulus_error,
        fmt="o",
        markersize=4,
        color="tab:green",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
    )

    if len(redshift) >= 2:
        z_grid = np.linspace(np.min(redshift), np.max(redshift), 400)
        model_distance = predict_luminosity_distance(
            z_grid,
            omega_m,
            omega_lambda,
            hubble_constant_pc_inv,
        )
        model_modulus = distance_to_modulus(model_distance)
        label_lines = [
            r"Best-fit cosmological $\mu(z)$",
            rf"$\Omega_M = {omega_m:.3f}$",
            rf"$\Omega_\Lambda = {omega_lambda:.3f}$",
        ]
        h0_km_s_mpc, _ = to_km_s_per_mpc(hubble_constant_pc_inv, None)
        label_lines.append(rf"$H_0 = {h0_km_s_mpc:.2f}$ km s$^{{-1}}$ Mpc$^{{-1}}$")
        ax.plot(z_grid, model_modulus, color="black", linewidth=2, label="\n".join(label_lines))
        ax.legend()

    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance Modulus (mag)")
    ax.set_title("Distance Modulus vs. Redshift with Cosmological Fit")
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

    best_fit_h0 = None
    best_fit_h0_err = None
    best_fit_h0_km_s_mpc = None
    best_fit_h0_km_s_mpc_err = None

    low_z_mask = redshift < LOW_Z_THRESHOLD
    if np.any(low_z_mask):
        low_z = redshift[low_z_mask]
        low_d = distances[low_z_mask]
        low_d_err = distance_errors[low_z_mask]

        try:
            best_fit_h0, best_fit_h0_err = fit_hubble_constant(
                low_z,
                low_d,
                low_d_err,
            )
            best_fit_h0_km_s_mpc, best_fit_h0_km_s_mpc_err = to_km_s_per_mpc(
                best_fit_h0,
                best_fit_h0_err,
            )
            print(
                "Best-fit H0 for low-z subset: "
                f"{best_fit_h0:.3e} pc^-1 "
                f"(± {best_fit_h0_err:.3e}) -> "
                f"{best_fit_h0_km_s_mpc:.2f} km s^-1 Mpc^-1"
                + (
                    f" ± {best_fit_h0_km_s_mpc_err:.2f}"
                    if best_fit_h0_km_s_mpc_err is not None
                    else ""
                )
            )
        except Exception as exc:  # pragma: no cover - user data dependent
            best_fit_h0 = None
            best_fit_h0_err = None
            best_fit_h0_km_s_mpc = None
            best_fit_h0_km_s_mpc_err = None
            print(f"Could not fit H0 for low-z subset: {exc}")

        plot_distance_vs_redshift(
            low_z,
            low_d,
            low_d_err,
            LOW_Z_PLOT_OUTPUT_PATH,
            best_fit_h0=best_fit_h0,
            best_fit_h0_error=best_fit_h0_err,
            best_fit_h0_km_s_mpc=best_fit_h0_km_s_mpc,
            best_fit_h0_km_s_mpc_error=best_fit_h0_km_s_mpc_err,
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

    h0_for_cosmo = (
        best_fit_h0 if best_fit_h0 is not None else DEFAULT_H0_PC_INV
    )
    if best_fit_h0 is None:
        print(
            "No reliable low-z H0 fit; defaulting to "
            f"{DEFAULT_H0_KM_S_MPC:.1f} km s^-1 Mpc^-1 "
            "(converted to pc^-1) for cosmological optimization."
        )

    try:
        omega_m_fit, omega_lambda_fit, _ = fit_density_parameters(
            redshift,
            distances,
            distance_errors,
            h0_for_cosmo,
        )
        print(
            "Best-fit density parameters from cosmological d_L(z): "
            f"Ω_M = {omega_m_fit:.3f}, Ω_Λ = {omega_lambda_fit:.3f}"
        )
        plot_cosmological_modulus_fit(
            redshift,
            modulus,
            modulus_error,
            omega_m_fit,
            omega_lambda_fit,
            h0_for_cosmo,
            COSMO_MODULUS_PLOT_OUTPUT_PATH,
        )
        print(
            "Saved cosmological distance-modulus plot to "
            f"{COSMO_MODULUS_PLOT_OUTPUT_PATH.resolve()}"
        )
    except Exception as exc:
        print(f"Could not perform cosmological fit: {exc}")


if __name__ == "__main__":
    main()

