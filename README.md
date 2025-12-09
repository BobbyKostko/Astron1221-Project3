# Astron1221-Project3

Hubble diagram analysis using Type Ia supernova data from the Union2.1 catalog.

## Overview

This project analyzes Type Ia supernova observations to:
- Construct a Hubble diagram showing the relationship between distance and redshift
- Measure the Hubble constant (H₀) from low-redshift supernovae
- Fit cosmological parameters (Ω_M, Ω_Λ) using the full redshift range
- Investigate the accelerated expansion of the universe and dark energy

The analysis uses publicly available data from the Supernova Cosmology Project's Union2.1 compilation, containing distance modulus and redshift measurements for 580 Type Ia supernovae.

## Scientific Background

Type Ia supernovae serve as "standard candles" in cosmology because they reach nearly the same peak brightness regardless of the particular star system. This property allows astronomers to:

1. **Measure distances**: By comparing observed brightness to known intrinsic brightness, we can calculate the luminosity distance
2. **Measure expansion**: The redshift of light from these supernovae reveals how fast the universe is expanding

The 1998 discovery that the universe's expansion is accelerating (not slowing down) led to the concept of dark energy and won the 2011 Nobel Prize in Physics.

## Project Structure

```
Astron1221-Project3/
├── SNe_Hubble.py              # Main analysis module with core functions
├── Project3_Kostko.ipynb            # Jupyter notebook with detailed analysis
├── SNe data.csv               # Supernova catalog (Union2.1 data)
├── Raw text SNe data.csv      # Raw data file
├── modulus_vs_redshift.pdf    # Distance modulus vs redshift plot
├── distance_vs_redshift_low_z.pdf  # Low-z Hubble diagram with fit
└── distance_modulus_vs_redshift_fit.pdf  # Cosmological fit results
```

## Features

### Core Functionality (`SNe_Hubble.py`)

- **Data Loading**: Loads supernova catalog from CSV format
- **Distance Conversions**: Converts between distance modulus and luminosity distance
- **Hubble Constant Fitting**: Fits H₀ from low-redshift (z < 0.03) supernovae using linear Hubble law
- **Cosmological Parameter Fitting**: Fits Ω_M and Ω_Λ using full luminosity distance relation for a flat universe
- **Outlier Rejection**: Implements iterative sigma-clipping to remove statistical outliers
- **Visualization**: Generates publication-quality plots with error bars, fits, and residuals

### Key Functions

- `load_modulus_data()`: Loads supernova catalog
- `modulus_to_distance()`: Converts distance modulus to physical distance
- `fit_hubble_constant()`: Fits Hubble constant from low-z data
- `fit_density_parameters()`: Fits matter and dark energy density parameters
- `luminosity_distance_flat()`: Computes luminosity distance for flat universe cosmology
- `plot_modulus_vs_redshift()`: Creates distance modulus visualization
- `plot_distance_vs_redshift()`: Creates Hubble diagram with fit
- `plot_cosmological_modulus_fit()`: Creates cosmological fit visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Astron1221-Project3
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

Execute the main analysis script to generate all plots:

```bash
python SNe_Hubble.py
```

This will:
- Load the supernova data
- Generate a distance modulus vs redshift plot
- Fit the Hubble constant from low-redshift data
- Fit cosmological parameters (Ω_M, Ω_Λ) from the full dataset
- Save three PDF plots to the current directory

### Using the Jupyter Notebook

For interactive analysis and detailed explanations:

```bash
jupyter notebook Project3_Kostko.ipynb
```

The notebook provides:
- Detailed scientific background
- Step-by-step analysis walkthrough
- Visualization of results
- Discussion of cosmological implications

### Using as a Module

Import functions from `SNe_Hubble.py` in your own scripts:

```python
import SNe_Hubble as sne

# Load data
data = sne.load_modulus_data()

# Convert distance modulus to physical distance
modulus = data["Distance_Modulus"]
modulus_error = data["Distance_Modulus_Error"]
distances, distance_errors = sne.modulus_to_distance(modulus, modulus_error)

# Fit Hubble constant
redshift = data["Redshift"]
h0, h0_err = sne.fit_hubble_constant(redshift, distances, distance_errors)
```

## Data Format

The `SNe data.csv` file should contain the following columns:
- `Supernova_Name`: Name/identifier of the supernova
- `Redshift`: Redshift value (z)
- `Distance_Modulus`: Distance modulus (μ) in magnitudes
- `Distance_Modulus_Error`: Uncertainty in distance modulus

## Output

The script generates three PDF plots:

1. **`modulus_vs_redshift.pdf`**: Distance modulus vs redshift for all supernovae
2. **`distance_vs_redshift_low_z.pdf`**: Hubble diagram for low-redshift supernovae with linear fit and residuals
3. **`distance_modulus_vs_redshift_fit.pdf`**: Full redshift range with cosmological fit, showing best-fit Ω_M and Ω_Λ values

## Methodology

### Low-Redshift Analysis
For supernovae with z < 0.03, the analysis uses the simple linear Hubble law:
```
D = z / H₀
```
where D is distance and H₀ is the Hubble constant.

### Cosmological Analysis
For the full redshift range, the analysis uses the luminosity distance relation for a flat universe:
```
d_L(z) = (1+z) / H₀ × ∫[0 to z] dz' / √[Ω_M(1+z')³ + Ω_Λ]
```
where Ω_M is the matter density parameter and Ω_Λ is the dark energy density parameter.

The code fits these parameters by minimizing the chi-squared statistic between observed and predicted distances.

## Dependencies

- **numpy**: Numerical computations and array operations
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing (integration and optimization)

See `requirements.txt` for specific version requirements.

## References

- Supernova Cosmology Project: Union2.1 compilation
- Perlmutter et al. (1999) - Discovery of cosmic acceleration
- Riess et al. (1998) - Independent confirmation of acceleration

## License

This project is part of an academic coursework (Astron1221). Please cite appropriately if using this code for research purposes.

## Author

Created for Astron1221 Project 3.

## Acknowledgments

Data from the Supernova Cosmology Project's Union2.1 catalog.
