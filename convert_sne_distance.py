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
LOW_Z_SIGMA_CLIP_THRESHOLD = 3.0
LOW_Z_SIGMA_CLIP_MAX_ITER = 5
COSMO_SIGMA_CLIP_THRESHOLD = 3.0
COSMO_SIGMA_CLIP_MAX_ITER = 5


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


def reduced_chi_squared(
    observed: np.ndarray,
    model: np.ndarray,
    sigma: np.ndarray,
    n_params: int,
) -> float:
    """Compute reduced χ² with basic handling of invalid uncertainties."""
    if n_params < 0:
        raise ValueError("Number of fit parameters must be non-negative.")

    observed = np.asarray(observed, dtype=float)
    model = np.asarray(model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    finite_sigma = sigma[(sigma > 0) & np.isfinite(sigma)]
    fallback_sigma = np.median(finite_sigma) if finite_sigma.size else 1.0
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 1.0
    sigma = np.where(
        (sigma > 0) & np.isfinite(sigma),
        sigma,
        fallback_sigma,
    )

    mask = np.isfinite(observed) & np.isfinite(model)
    if not np.any(mask):
        return float("nan")

    observed = observed[mask]
    model = model[mask]
    sigma = sigma[mask]

    dof = len(observed) - n_params
    if dof <= 0:
        return float("nan")

    chi_squared = np.sum(((observed - model) / sigma) ** 2)
    return chi_squared / dof


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


def _residual_scale(residuals: np.ndarray) -> float:
    """Return a robust scale estimate (std with MAD fallback)."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return float("nan")
    if residuals.size > 1:
        std = np.std(residuals, ddof=1)
        if np.isfinite(std) and std > 0:
            return std
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    scale = 1.4826 * mad
    return scale if np.isfinite(scale) and scale > 0 else float("nan")


def _sigma_clip_outliers(
    redshift: np.ndarray,
    distance: np.ndarray,
    distance_error: np.ndarray,
    valid_mask: np.ndarray,
    model_predictor,
    min_required_points: int,
    sigma_threshold: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic iterative sigma clipper that removes large-residual data points.

    Parameters
    ----------
    redshift, distance, distance_error : arrays of equal length.
    valid_mask : boolean array indicating which rows are eligible for clipping.
    model_predictor : callable taking (z, d, err) subsets and returning model
        distances for those redshifts.
    min_required_points : int
        Stop if fewer than this many points remain.
    """

    redshift = np.asarray(redshift, dtype=float)
    distance = np.asarray(distance, dtype=float)
    distance_error = np.asarray(distance_error, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if redshift.shape != valid_mask.shape:
        raise ValueError("valid_mask must match the shape of redshift.")

    keep_mask = valid_mask.copy()
    outlier_mask = np.zeros_like(valid_mask, dtype=bool)

    if np.count_nonzero(keep_mask) < max(min_required_points, 1):
        return keep_mask, outlier_mask

    sigma_threshold = float(sigma_threshold)
    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be positive.")
    max_iterations = max(1, int(max_iterations))
    min_required_points = max(1, int(min_required_points))

    for _ in range(max_iterations):
        idx = np.where(keep_mask)[0]
        if idx.size < min_required_points:
            break
        try:
            model = model_predictor(
                redshift[idx],
                distance[idx],
                distance_error[idx],
            )
        except Exception:
            break

        residuals = distance[idx] - model
        scale = _residual_scale(residuals)
        if not np.isfinite(scale) or scale <= 0:
            break

        standardized = residuals / scale
        flagged = np.abs(standardized) > sigma_threshold
        if not np.any(flagged):
            break

        flagged_indices = idx[flagged]
        keep_mask[flagged_indices] = False
        outlier_mask[flagged_indices] = True

    return keep_mask, outlier_mask


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
        markersize=3,
        color="tab:blue",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
        alpha=0.65,
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
    rejected_points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Plot physical distance vs redshift, residuals, and optional outliers."""
    show_fit = best_fit_h0 is not None and len(redshift) > 1
    if show_fit:
        fig, (ax, ax_resid) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7, 7),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax_resid = None

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

    if rejected_points is not None:
        rej_z, rej_d, rej_err = rejected_points
        if len(rej_z):
            ax.errorbar(
                rej_z,
                rej_d,
                yerr=rej_err,
                fmt="x",
                markersize=5,
                color="tab:red",
                ecolor="tab:red",
                elinewidth=1,
                capsize=2,
                linestyle="none",
                label=rf"Rejected (>{LOW_Z_SIGMA_CLIP_THRESHOLD:.0f}$\sigma$)",
                alpha=0.85,
            )

    model_at_data = None
    chi_sq_red = float("nan")
    if show_fit:
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
        model_at_data = hubble_law(redshift, best_fit_h0)
        chi_sq_red = reduced_chi_squared(
            distance,
            model_at_data,
            distance_error,
            n_params=1,
        )
        if np.isfinite(chi_sq_red):
            label_lines.append(rf"$\chi^2_\nu = {chi_sq_red:.2f}$")

        ax.plot(
            z_grid,
            d_grid,
            color="black",
            linestyle="--",
            label="\n".join(label_lines),
        )

    if show_fit and ax_resid is not None and model_at_data is not None:
        residuals = distance - model_at_data
        ax_resid.errorbar(
            redshift,
            residuals,
            yerr=distance_error,
            fmt="o",
            markersize=4,
            color="tab:purple",
            ecolor="tab:gray",
            elinewidth=1,
            capsize=3,
            linestyle="none",
        )
        ax_resid.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax_resid.set_ylabel("Residual (pc)")
        ax_resid.grid(True, which="both", alpha=0.3)
        ax_resid.set_xlabel("Redshift (z)")
    else:
        ax.set_xlabel("Redshift (z)")

    ax.set_ylabel("Distance (pc)")
    ax.set_title(
        f"Distance vs. Redshift (subset of {len(redshift)} low-z entries)"
    )
    ax.grid(True, which="both", alpha=0.3)
    if show_fit:
        ax.tick_params(labelbottom=False)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
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
    rejected_points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """
    Plot distance modulus with best-fit Ω parameters, reference models,
    and optional highlighted outliers.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        redshift,
        modulus,
        yerr=modulus_error,
        fmt="o",
        markersize=2.5,
        color="tab:green",
        ecolor="tab:gray",
        elinewidth=1,
        capsize=3,
        linestyle="none",
        alpha=0.55,
    )

    if rejected_points is not None:
        rej_z, rej_mod, rej_err = rejected_points
        if len(rej_z):
            ax.errorbar(
                rej_z,
                rej_mod,
                yerr=rej_err,
                fmt="x",
                markersize=4,
                color="tab:red",
                ecolor="tab:red",
                elinewidth=1,
                capsize=2,
                linestyle="none",
                label=rf"Rejected (>{COSMO_SIGMA_CLIP_THRESHOLD:.0f}$\sigma$)",
                alpha=0.85,
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
        model_distance_data = predict_luminosity_distance(
            redshift,
            omega_m,
            omega_lambda,
            hubble_constant_pc_inv,
        )
        model_modulus_data = distance_to_modulus(model_distance_data)
        label_lines = [
            r"Best-fit cosmological $\mu(z)$",
            rf"$\Omega_M = {omega_m:.3f}$",
            rf"$\Omega_\Lambda = {omega_lambda:.3f}$",
        ]
        h0_km_s_mpc, _ = to_km_s_per_mpc(hubble_constant_pc_inv, None)
        label_lines.append(rf"$H_0 = {h0_km_s_mpc:.2f}$ km s$^{{-1}}$ Mpc$^{{-1}}$")
        chi_sq_red = reduced_chi_squared(
            modulus,
            model_modulus_data,
            modulus_error,
            n_params=2,
        )
        if np.isfinite(chi_sq_red):
            label_lines.append(rf"$\chi^2_\nu = {chi_sq_red:.2f}$")
        ax.plot(
            z_grid,
            model_modulus,
            color="black",
            linewidth=2,
            label="\n".join(label_lines),
        )

        hypothetical_pairs = [
            (1.0, 0.0, r"(1, 0)"),
            (0.5, 0.5, r"(0.5, 0.5)"),
            (0.0, 1.0, r"(0, 1)"),
        ]
        styles = [
            ("tab:red", "--"),
            ("tab:purple", ":"),
            ("tab:orange", "-."),
        ]
        for (omega_m_ref, omega_lambda_ref, label_suffix), (color, style) in zip(
            hypothetical_pairs, styles
        ):
            try:
                ref_distance = predict_luminosity_distance(
                    z_grid,
                    omega_m_ref,
                    omega_lambda_ref,
                    hubble_constant_pc_inv,
                )
                ref_modulus = distance_to_modulus(ref_distance)
            except ValueError:
                continue
            ax.plot(
                z_grid,
                ref_modulus,
                color=color,
                linestyle=style,
                linewidth=1.2,
                label=rf"$(\Omega_M,\Omega_\Lambda) = {label_suffix}$",
            )

    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance Modulus (mag)")
    ax.set_title("Distance Modulus vs. Redshift with Cosmological Fit")
    ax.grid(True, which="both", alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
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

        low_valid_mask = (
            np.isfinite(low_z)
            & np.isfinite(low_d)
            & np.isfinite(low_d_err)
            & (low_d_err >= 0.0)
        )

        def _low_z_model(
            z_subset: np.ndarray,
            d_subset: np.ndarray,
            err_subset: np.ndarray,
        ) -> np.ndarray:
            h0, _ = fit_hubble_constant(z_subset, d_subset, err_subset)
            return hubble_law(z_subset, h0)

        inlier_mask, outlier_mask = _sigma_clip_outliers(
            low_z,
            low_d,
            low_d_err,
            low_valid_mask,
            _low_z_model,
            min_required_points=2,
            sigma_threshold=LOW_Z_SIGMA_CLIP_THRESHOLD,
            max_iterations=LOW_Z_SIGMA_CLIP_MAX_ITER,
        )
        valid_mask = inlier_mask | outlier_mask
        if not np.any(valid_mask):
            print(
                "No finite low-z entries survived quality checks; "
                "skipping low-z fit and plot."
            )
        else:
            filtered_low_z = low_z[inlier_mask]
            filtered_low_d = low_d[inlier_mask]
            filtered_low_d_err = low_d_err[inlier_mask]
            num_outliers = int(np.count_nonzero(outlier_mask))
            rejected_points = None

            if filtered_low_z.size >= 2:
                if num_outliers:
                    rejected_points = (
                        low_z[outlier_mask],
                        low_d[outlier_mask],
                        low_d_err[outlier_mask],
                    )
                    print(
                        f"Removed {num_outliers} low-z outlier(s) "
                        f"using {LOW_Z_SIGMA_CLIP_THRESHOLD:.0f}σ residual clipping."
                    )
            else:
                filtered_low_z = low_z[valid_mask]
                filtered_low_d = low_d[valid_mask]
                filtered_low_d_err = low_d_err[valid_mask]
                rejected_points = None
                if num_outliers:
                    print(
                        "Not enough low-z points after clipping; "
                        "using all valid measurements for the H0 fit instead."
                    )

            try:
                best_fit_h0, best_fit_h0_err = fit_hubble_constant(
                    filtered_low_z,
                    filtered_low_d,
                    filtered_low_d_err,
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

            if best_fit_h0 is not None:
                plot_distance_vs_redshift(
                    filtered_low_z,
                    filtered_low_d,
                    filtered_low_d_err,
                    LOW_Z_PLOT_OUTPUT_PATH,
                    best_fit_h0=best_fit_h0,
                    best_fit_h0_error=best_fit_h0_err,
                    best_fit_h0_km_s_mpc=best_fit_h0_km_s_mpc,
                    best_fit_h0_km_s_mpc_error=best_fit_h0_km_s_mpc_err,
                    rejected_points=rejected_points,
                )
            else:
                plot_distance_vs_redshift(
                    filtered_low_z,
                    filtered_low_d,
                    filtered_low_d_err,
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

    h0_for_cosmo = (
        best_fit_h0 if best_fit_h0 is not None else DEFAULT_H0_PC_INV
    )
    if best_fit_h0 is None:
        print(
            "No reliable low-z H0 fit; defaulting to "
            f"{DEFAULT_H0_KM_S_MPC:.1f} km s^-1 Mpc^-1 "
            "(converted to pc^-1) for cosmological optimization."
        )

    cosmo_valid_mask = (
        np.isfinite(redshift)
        & np.isfinite(distances)
        & np.isfinite(distance_errors)
        & (redshift >= 0.0)
        & (distances > 0.0)
        & (distance_errors >= 0.0)
    )

    def _cosmo_model(
        z_subset: np.ndarray,
        d_subset: np.ndarray,
        err_subset: np.ndarray,
    ) -> np.ndarray:
        omega_m, omega_lambda, _ = fit_density_parameters(
            z_subset,
            d_subset,
            err_subset,
            h0_for_cosmo,
        )
        return predict_luminosity_distance(
            z_subset,
            omega_m,
            omega_lambda,
            h0_for_cosmo,
        )

    try:
        cosmo_inlier_mask, cosmo_outlier_mask = _sigma_clip_outliers(
            redshift,
            distances,
            distance_errors,
            cosmo_valid_mask,
            _cosmo_model,
            min_required_points=3,
            sigma_threshold=COSMO_SIGMA_CLIP_THRESHOLD,
            max_iterations=COSMO_SIGMA_CLIP_MAX_ITER,
        )
    except Exception as exc:  # pragma: no cover - user data dependent
        print(f"Could not run cosmological sigma clipping: {exc}")
        cosmo_inlier_mask = cosmo_valid_mask.copy()
        cosmo_outlier_mask = np.zeros_like(redshift, dtype=bool)

    cosmo_valid_mask_combined = cosmo_inlier_mask | cosmo_outlier_mask
    if not np.any(cosmo_valid_mask_combined):
        raise RuntimeError("No valid entries available for cosmological fit.")

    cosmo_filtered_mask = cosmo_inlier_mask.copy()
    num_cosmo_inliers = int(np.count_nonzero(cosmo_filtered_mask))
    num_cosmo_outliers = int(np.count_nonzero(cosmo_outlier_mask))
    use_all_cosmo_data = False

    if num_cosmo_inliers < 3:
        cosmo_filtered_mask = cosmo_valid_mask_combined
        use_all_cosmo_data = True
        if num_cosmo_outliers:
            print(
                "Not enough cosmological points after clipping; "
                "using all valid entries for the Ω fit instead."
            )
        num_cosmo_outliers = 0

    cosmo_rejected_points = None
    if num_cosmo_outliers and not use_all_cosmo_data:
        cosmo_rejected_points = (
            redshift[cosmo_outlier_mask],
            modulus[cosmo_outlier_mask],
            modulus_error[cosmo_outlier_mask],
        )
        print(
            f"Removed {num_cosmo_outliers} cosmological outlier(s) "
            f"using {COSMO_SIGMA_CLIP_THRESHOLD:.0f}σ residual clipping."
        )

    filtered_redshift = redshift[cosmo_filtered_mask]
    filtered_distances = distances[cosmo_filtered_mask]
    filtered_distance_errors = distance_errors[cosmo_filtered_mask]
    filtered_modulus = modulus[cosmo_filtered_mask]
    filtered_modulus_error = modulus_error[cosmo_filtered_mask]

    try:
        omega_m_fit, omega_lambda_fit, _ = fit_density_parameters(
            filtered_redshift,
            filtered_distances,
            filtered_distance_errors,
            h0_for_cosmo,
        )
        print(
            "Best-fit density parameters from cosmological d_L(z): "
            f"Ω_M = {omega_m_fit:.3f}, Ω_Λ = {omega_lambda_fit:.3f}"
        )
        plot_cosmological_modulus_fit(
            filtered_redshift,
            filtered_modulus,
            filtered_modulus_error,
            omega_m_fit,
            omega_lambda_fit,
            h0_for_cosmo,
            COSMO_MODULUS_PLOT_OUTPUT_PATH,
            rejected_points=cosmo_rejected_points,
        )
        print(
            "Saved cosmological distance-modulus plot to "
            f"{COSMO_MODULUS_PLOT_OUTPUT_PATH.resolve()}"
        )
    except Exception as exc:
        print(f"Could not perform cosmological fit: {exc}")


if __name__ == "__main__":
    main()

