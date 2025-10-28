# Compact DSMA & CM simulator
# - Matches R0 across models
# - Computes mean infected degree ⟨k⟩_I for both models
# - Exports trajectories.csv and simulation_results.csv

import numpy as np
import pandas as pd
import os
from numpy.linalg import eigvals
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# -----------------------------
# Degree distributions
# -----------------------------
def degree_distribution(dist_type, kmax, alpha=None, poisson_mean=None):
    k = np.arange(1, kmax + 1)
    if dist_type == "truncated_powerlaw":
        if alpha is None:
            raise ValueError("alpha required for truncated_powerlaw")
        Pk = k**(-alpha); Pk = Pk / Pk.sum()
        return k, Pk
    elif dist_type == "uniform_1_5":
        k = np.arange(1, 6)
        Pk = np.ones_like(k, float) / 5.0
        return k, Pk
    elif dist_type == "homogeneous_5":
        return np.array([5]), np.array([1.0])
    else:
        raise ValueError("Unknown dist_type")

def get_kPk(dist_type, kmax=None, alpha=None, poisson_mean=None):
    return degree_distribution(dist_type, kmax, alpha, poisson_mean)

# -----------------------------
# Proportionate-mixing kernels
# -----------------------------
def make_Q(Pk, k):
    # Q_{j|k} columns-identical proportionate mixing
    kbar = np.sum(k * Pk)
    base = (k * Pk) / kbar
    return np.tile(base[:, None], (1, len(k)))  # shape (j,k)

def Q_k_given_j_from_Q_j_given_k(Q_jk, k, Pk):
    # Bayes on stubs, columns sum to 1
    stub_prior = k * Pk
    weighted = Q_jk * stub_prior[None, :]
    denom = weighted.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return (weighted.T) / denom.T

# -----------------------------
# R0 calculators & beta matchers
# -----------------------------
def R0_EBMA(beta, gamma, Q, k, Pk):
    Q_kj = Q_k_given_j_from_Q_j_given_k(Q, k, Pk)
    B = Q_kj * k[None, :]           # B_{k,j} = j * Q_{k|j}
    rhoB = float(np.max(np.real(eigvals(B))))
    return (beta / gamma) * rhoB

def R0_StaticCM(beta, gamma, Q, k, Pk):
    Q_kj = Q_k_given_j_from_Q_j_given_k(Q, k, Pk)
    j_minus_1 = np.maximum(k - 1, 0)
    M = Q_kj * j_minus_1[None, :]
    rhoM = float(np.max(np.real(eigvals(M))))
    T = beta / (beta + gamma)
    return T * rhoM, rhoM

def beta_ma_for_target_R0(R0_target, gamma, Q, k, Pk):
    f = lambda beta: R0_EBMA(beta, gamma, Q, k, Pk) - R0_target
    return brentq(f, 1e-12, 1e3)

def beta_cm_for_target_R0(R0_target, gamma, Q, k, Pk):
    _, rhoM = R0_StaticCM(beta=1.0, gamma=gamma, Q=Q, k=k, Pk=Pk)
    x = max(0.0, min(R0_target / rhoM, 0.999))  # clamp
    return 0.0 if x <= 0 else gamma * x / (1.0 - x)

# -----------------------------
# Seeding
# -----------------------------
def compute_theta0(Pk, k, rho_seed):
    f = lambda th: np.sum(Pk * th**k) - (1.0 - rho_seed)
    return brentq(f, 0.0, 1.0 - 1e-12)

# -----------------------------
# ODE RHS
# -----------------------------
def rhs_EBMA(t, y, beta, gamma, Q, k, Pk):
    K = len(k)
    theta = y[:K]
    pi_R  = y[K:2*K]
    R     = y[-1]

    Q_kj = Q_k_given_j_from_Q_j_given_k(Q, k, Pk)
    theta_pow = theta**k
    pi_S = (Q_kj.T @ theta_pow)
    pi_I = 1.0 - pi_S - pi_R

    dtheta = -beta * theta * pi_I
    dpi_R  =  gamma * pi_I

    S = np.sum(Pk * theta_pow)
    I = 1.0 - S - R
    dR = gamma * I
    return np.concatenate([dtheta, dpi_R, [dR]])

def rhs_StaticCM(t, y, beta, gamma, Q, k, Pk):
    K = len(k)
    theta = y[:K]
    phi_R = y[K:2*K]
    R     = y[-1]

    Phi   = (Q.T @ theta)
    phi_S = Phi**(k - 1)
    phi_I = theta - phi_S - phi_R

    dtheta = -beta * phi_I
    dphi_R =  gamma * phi_I

    S = np.sum(Pk * (Q.T @ theta)**k)
    I = 1.0 - S - R
    dR = gamma * I
    return np.concatenate([dtheta, dphi_R, [dR]])

# -----------------------------
# Trajectories (DSMA & CM) + mean ⟨k⟩_I
# -----------------------------
def get_trajectories(R0_target, alpha, kmax=30, gamma=1.0, rho_seed=1e-6,
                     t_span=(0, 150), t_points=4000):
    k, Pk = degree_distribution("truncated_powerlaw", kmax, alpha)
    kbar = np.sum(k * Pk)
    Q = make_Q(Pk, k)
    Q_kj = Q_k_given_j_from_Q_j_given_k(Q, k, Pk)

    beta_ma = beta_ma_for_target_R0(R0_target, gamma, Q, k, Pk)
    beta_cm = beta_cm_for_target_R0(R0_target, gamma, Q, k, Pk)

    theta0 = compute_theta0(Pk, k, rho_seed)
    theta0_vec = np.full_like(k, theta0, float)

    y0_ebma = np.concatenate([theta0_vec, np.zeros_like(k, float), [0.0]])
    y0_cm   = np.concatenate([theta0_vec, np.zeros_like(k, float), [0.0]])

    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Integrate
    sol_e = solve_ivp(rhs_EBMA, t_span, y0_ebma, t_eval=t_eval,
                      args=(beta_ma, gamma, Q, k, Pk), rtol=1e-8, atol=1e-10)
    sol_c = solve_ivp(rhs_StaticCM, t_span, y0_cm, t_eval=t_eval,
                      args=(beta_cm, gamma, Q, k, Pk), rtol=1e-8, atol=1e-10)

    # EBMA series
    theta_t = sol_e.y[:len(k), :].T
    R_e = sol_e.y[-1]
    S_e = np.sum(Pk * (theta_t**k), axis=1)
    I_e = 1.0 - S_e - R_e

    piR_t = sol_e.y[len(k):2*len(k), :].T
    pi_S  = (Q_kj.T @ (theta_t**k).T).T
    pi_I  = 1.0 - pi_S - piR_t
    q = (k * Pk) / kbar
    pi_I_scalar = np.sum(pi_I * q[None, :], axis=1)
    mean_kI_e = np.where(I_e > 1e-10, (kbar * pi_I_scalar) / I_e, np.nan)

    df_e = pd.DataFrame({
        "time": sol_e.t, "I": I_e, "R": R_e,
        "mean_kI": mean_kI_e, "model": "DSMA",
        "alpha": alpha, "R0_target": R0_target
    })

    # CM series + mean ⟨k⟩_I
    theta_t = sol_c.y[:len(k), :].T
    R_c = sol_c.y[-1]
    S_c = np.sum(Pk * ((Q.T @ theta_t.T).T ** k), axis=1)
    I_c = 1.0 - S_c - R_c

    phiR_t = sol_c.y[len(k):2*len(k), :].T
    Phi_t = (Q.T @ theta_t.T).T
    Phi_t = np.clip(Phi_t, 0.0, 1.0)
    S_mat = np.clip(Phi_t**k, 0.0, 1.0)

    phiS_nei = np.clip(Phi_t**(k - 1), 0.0, 1.0)
    phiI_nei = theta_t - phiS_nei - phiR_t
    dtheta   = -beta_cm * phiI_nei
    dPhi     = (Q.T @ dtheta.T).T
    dSdt     = (k * np.clip(Phi_t**(k - 1), 0.0, 1.0)) * dPhi

    Tn, Kn = dSdt.shape
    exp_gt = np.exp(gamma * sol_c.t)[:, None]
    Ik = np.empty_like(S_mat)
    Ik[0, :] = 1.0 - S_mat[0, :]
    yint = exp_gt[0, :] * Ik[0, :]
    for i in range(1, Tn):
        dt_i = sol_c.t[i] - sol_c.t[i - 1]
        trap = 0.5 * dt_i * (exp_gt[i - 1, :] * dSdt[i - 1, :] + exp_gt[i, :] * dSdt[i, :])
        yint = yint - trap
        Ik[i, :] = yint / exp_gt[i, :]
    Ik = np.clip(Ik, 0.0, 1.0)

    num = np.sum((k * Pk) * Ik, axis=1)
    den = np.sum(Pk * Ik, axis=1)
    mean_kI_c = np.where(den > 1e-10, num / den, np.nan)

    df_c = pd.DataFrame({
        "time": sol_c.t, "I": I_c, "R": R_c,
        "mean_kI": mean_kI_c, "model": "CM",
        "alpha": alpha, "R0_target": R0_target
    })

    return df_e, df_c

# -----------------------------
# One-shot run for a config → summary row
# -----------------------------
def simulate_summary(config, dist_name_for_output=None):
    dist_type   = config["dist_type"]
    kmax        = config.get("kmax")
    alpha       = config.get("alpha")
    gamma       = config["gamma"]
    rho_seed    = config["rho_seed"]
    t_span      = config["t_span"]
    t_eval      = np.linspace(t_span[0], t_span[1], config["t_points"])

    k, Pk = get_kPk(dist_type, kmax, alpha)
    Q = make_Q(Pk, k)

    if "beta_ma" in config:
        beta_ma = config["beta_ma"]
        R0_target = R0_EBMA(beta_ma, gamma, Q, k, Pk)
    else:
        R0_target = config["R0_target"]
        beta_ma = beta_ma_for_target_R0(R0_target, gamma, Q, k, Pk)

    beta_cm = beta_cm_for_target_R0(R0_target, gamma, Q, k, Pk)

    theta0 = compute_theta0(Pk, k, rho_seed)
    theta0_vec = np.full_like(k, theta0, float)
    y0_ebma = np.concatenate([theta0_vec, np.zeros_like(k, float), [0.0]])
    y0_cm   = np.concatenate([theta0_vec, np.zeros_like(k, float), [0.0]])

    sol_e = solve_ivp(rhs_EBMA, t_span, y0_ebma, t_eval=t_eval,
                      args=(beta_ma, gamma, Q, k, Pk), rtol=1e-8, atol=1e-10)
    sol_c = solve_ivp(rhs_StaticCM, t_span, y0_cm, t_eval=t_eval,
                      args=(beta_cm, gamma, Q, k, Pk), rtol=1e-8, atol=1e-10)

    def get_I_R(sol, model):
        theta_t = sol.y[:len(k), :].T
        R = sol.y[-1]
        if model == "EBMA":
            S = np.sum(Pk * (theta_t**k), axis=1)
        else:
            S = np.sum(Pk * ((Q.T @ theta_t.T).T ** k), axis=1)
        I = 1.0 - S - R
        return I, R

    I_e, R_e = get_I_R(sol_e, "EBMA")
    I_c, R_c = get_I_R(sol_c, "CM")

    def metrics(I, R, t):
        pidx = int(np.argmax(I))
        return float(R[-1]), float(I[pidx]), float(t[pidx])

    fs_e, ps_e, tp_e = metrics(I_e, R_e, sol_e.t)
    fs_s, ps_s, tp_s = metrics(I_c, R_c, sol_c.t)
    R0_measured = R0_EBMA(beta_ma, gamma, Q, k, Pk)

    return {
        "distribution": dist_name_for_output or dist_type,
        "alpha": alpha if dist_type == "truncated_powerlaw" else None,
        "kmax": int(k[-1]),
        "R0_target": float(R0_target),
        "R0_measured": float(R0_measured),
        "beta_ma": float(beta_ma),
        "beta_cm": float(beta_cm),
        "final_EBMA": fs_e,
        "peak_EBMA": ps_e,
        "tpeak_EBMA": tp_e,
        "final_Static": fs_s,
        "peak_Static": ps_s,
        "tpeak_Static": tp_s,
    }

# -----------------------------
# Main: write trajectories.csv and simulation_results.csv
# -----------------------------
if __name__ == "__main__":
    # Config (keep same knobs as your long solver; ε removed)
    CONFIG = {
        "dist_type": "truncated_powerlaw",
        "kmax": 50,
        "alpha": 2.5,
        "gamma": 1.0,
        "rho_seed": 1e-6,
        "t_span": (0.0, 400.0),
        "t_points": 40000,
    }

    # ---- Trajectories ----
    traj_file = "trajectories.csv"
    if os.path.exists(traj_file):
        os.remove(traj_file)

    dfs = []
    for R0 in [1.2, 1.6, 2.0]:
        df_e, df_c = get_trajectories(R0, alpha=2.5, kmax=30)
        dfs.extend([df_e, df_c])

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(traj_file, index=False)
    print(f"Saved trajectories to {traj_file}")

    # ---- Simulation results grid 
    R0_grid = np.round(np.arange(0.5, 3.0 + 1e-9, 0.01), 10)
    distributions = [
        {"dist_type": "truncated_powerlaw", "kmax": 30, "alpha": 2.0, "label": "Power law α=2.0, kmax=30"},
        {"dist_type": "truncated_powerlaw", "kmax": 30, "alpha": 2.5, "label": "Power law α=2.5, kmax=30"},
        {"dist_type": "truncated_powerlaw", "kmax": 30, "alpha": 3.0, "label": "Power law α=3.0, kmax=30"},
    ]

    results = []
    for dist in distributions:
        k, Pk = get_kPk(dist["dist_type"], dist["kmax"], dist["alpha"], None)
        Q = make_Q(Pk, k)
        _, rhoM = R0_StaticCM(beta=1.0, gamma=CONFIG["gamma"], Q=Q, k=k, Pk=Pk)
        R0_static_max = rhoM

        for R0 in R0_grid:
            if R0 > R0_static_max:
                # CM cannot reach this target R0 (T<=1 ceiling)
                continue

            cfg = CONFIG.copy()
            cfg["dist_type"] = dist["dist_type"]
            cfg["kmax"] = dist["kmax"]
            cfg["alpha"] = dist["alpha"]
            cfg["R0_target"] = R0
            # compute matching beta_ma inside simulate_summary
            res = simulate_summary(cfg, dist_name_for_output=dist["label"])
            results.append(res)

    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_results.csv", index=False)
    print("Saved simulation_results.csv")
