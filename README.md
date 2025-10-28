# Thesis Code Repository – Edge-Based SIR Comparison

This repository contains the complete and reproducible code used in my thesis
to compare **Degree Stratified Mass Action (DSMA)** and **Configuration Model (CM)** epidemic models in edge-based formulation (Volz Miller 2012).

Both models are matched on the basic reproduction number \( R_0 \) and share identical
degree distributions under degree-proportionate mixing.

---

## Quick Start

```bash
# Python ≥3.10 recommended
pip install numpy pandas scipy matplotlib

# 1. Run the solver to generate data
python solver.py

# 2. Plot R0-dependent differences (requires simulation_results.csv)
python absolute_difference_plot.py

# 3. Plot epidemic trajectories for selected R0 values
python incidence_plots.py
```

---

## File Overview

### 1. solver.py
- Core simulation engine.
- Integrates both DSMA and CM edge based ODE systems.
- Saves two datasets:
  - **simulation_results.csv**
    - Summary over R0 × degree distributions.
  - **trajectories.csv**
    - Time-resolved trajectories for selected R0 values.
- Creates subfolders under `plots/` automatically.

### 2. absolute_difference_plot.py
- Loads `simulation_results.csv`.
- Computes and visualizes CM–MA differences in:
  - Final epidemic size
  - Peak prevalence
  - Time to peak
- Default: power-law degree distribution with α = 2.5.
- Outputs to `plots/alpha25/*.png`.

### 3. incidence_plots.py
- Loads `trajectories.csv`.
- Plots DSMA vs CM trajectories for R0 = 1.2, 1.6, 2.0.
- Top row: Prevalence \( I(t) \)
- Bottom row: Recovered \( R(t) \)
- Saves combined figure to:
  - `plots/trajectories/trajectories_alpha2.5.png`
  - `plots/trajectories/trajectories_alpha2.5.pdf`

---

## Outputs

### simulation_results.csv
| Column | Description |
|--------|--------------|
| distribution | Degree distribution label |
| alpha | Power-law exponent (if applicable) |
| kmax | Maximum degree |
| R0_target | Target basic reproduction number |
| beta_ma | Transmission rate (DSMA) |
| beta_cm | Transmission rate ( CM) |
| final_EBMA | Final recovered fraction (DSMA) |
| peak_EBMA | Peak prevalence (DSMA) |
| tpeak_EBMA | Time to peak (DSMA) |
| final_Static | Final recovered fraction ( M) |
| peak_Static | Peak prevalence (CM) |
| tpeak_Static | Time to peak (CM) |

### trajectories.csv
| Column | Description |
|--------|--------------|
| time | Simulation time (in 1/γ units) |
| I | Infectious fraction |
| R | Recovered fraction |
| model | "DSMA" or "CM" |
| R0 | Target R0 value |
| alpha | Power-law exponent used |

### Figures
- `plots/alpha25/*.png` — R0-based comparison plots.
- `plots/trajectories/trajectories_alpha2.5.(png|pdf)` — time trajectories.

---

## Parameters (editable in solver.py)

| Parameter | Meaning | Default |
|------------|----------|----------|
| gamma | Recovery rate | 1.0 |
| rho_seed | Initial infection fraction | 1e-6 |
| kmax | Max degree for power-law | 30 |
| alpha | Power-law exponent | {2.0, 2.5, 3.0} |
| R0 grid | Range of R0 for simulation table | 0.5 → 3.0 (step 0.01) |
| Trajectory R0 values | Shown in plots | [1.2, 1.6, 2.0] |

---

## Notes

- **R0 matching**:
  - DSMA: \( \beta_{MA} = \gamma \, R_0 / (\langle k^2 \rangle / \langle k \rangle) \)
  - CM: \( R_0 = T \, \rho(M) \), with transmissibility \( T = \beta_{CM} / (\beta_{CM} + \gamma) \)
- CM results are skipped if \( R_0 \) exceeds its spectral ceiling.
- All runs assume degree-proportionate mixing.

---

**Author:** Ezra  
