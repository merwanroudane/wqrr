"""
Wavelet Quantile Mediation & Moderation Regression.

Implements:
  - Direct effect:    Y ~ X
  - Interaction:      Y ~ X + Z + X·Z  (moderation)
  - Path a:           Z ~ X
  - Path b:           Y ~ X + Z        (coef on Z)
  - Indirect effect:  a × b            (mediation)

Based on Adebayo (2025):
  "Can ESG Uncertainty Alter the Emissions Impact of Renewable Energy
   Consumption?"  Statistical Journal of the IAOS.
  doi:10.1177/18747655261445375
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from .utils import modwt_mra, aggregate_bands, safe_print


@dataclass
class MediationResult:
    """Container for wavelet quantile mediation + moderation results."""
    direct: pd.DataFrame         # Y ~ X
    interaction: pd.DataFrame    # coef on X·Z in Y ~ X + Z + XZ
    path_a: pd.DataFrame         # Z ~ X
    path_b: pd.DataFrame         # coef on Z in Y ~ X + Z
    indirect: pd.DataFrame       # a × b with joint p-value
    quantiles: np.ndarray
    bands: list
    dep_name: str
    main_name: str
    mod_name: str
    n_obs: int

    def summary(self):
        """Print summary."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║  Wavelet Quantile Mediation & Moderation — Summary      ║",
            "╚══════════════════════════════════════════════════════════╝",
            "",
            f"  Dependent (Y) : {self.dep_name}",
            f"  Main (X)      : {self.main_name}",
            f"  Moderator (Z) : {self.mod_name}",
            f"  Observations  : {self.n_obs}",
            f"  Bands         : {self.bands}",
            "",
        ]
        for name, df in [("Direct", self.direct), ("Interaction", self.interaction),
                         ("Path a", self.path_a), ("Path b", self.path_b),
                         ("Indirect", self.indirect)]:
            n_sig = (df["p_value"] < 0.05).sum() if "p_value" in df.columns else 0
            lines.append(f"  {name:14s}  sig(5%)={n_sig}/{len(df)}")
        lines.append("")
        safe_print("\n".join(lines))
        return self


def _qr_coef(dep, indep_dict, tau, n_boot=100):
    """Run QR and return dict {var_name: (beta, pval)} for each regressor."""
    X_cols = list(indep_dict.keys())
    X_mat = np.column_stack([indep_dict[k] for k in X_cols])
    X_mat = sm.add_constant(X_mat)

    try:
        model = QuantReg(dep, X_mat)
        res = model.fit(q=tau, max_iter=1000, p_tol=1e-5)
        out = {}
        for i, name in enumerate(X_cols):
            idx = i + 1
            out[name] = (
                res.params[idx],
                res.bse[idx] if res.bse is not None else np.nan,
                res.pvalues[idx] if res.pvalues is not None else np.nan,
            )
        return out
    except Exception:
        return {k: (np.nan, np.nan, np.nan) for k in X_cols}


def wavelet_mediation(y, x, z, quantiles=None, wavelet="la8", J=None,
                      n_boot=100, dep_name="Y", main_name="X",
                      mod_name="Z", verbose=True):
    """
    Wavelet Quantile Mediation & Moderation Regression.

    Parameters
    ----------
    y : array-like
        Dependent variable (Y).
    x : array-like
        Main independent variable (X).
    z : array-like
        Mediator / Moderator variable (Z).
    quantiles : array-like or None
        Quantile grid.
    wavelet : str
    J : int or None
    n_boot : int
    dep_name, main_name, mod_name : str
        Variable names for labels.
    verbose : bool

    Returns
    -------
    MediationResult
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n_obs = len(y)

    if J is None:
        J = max(3, int(np.log2(n_obs)) - 3)

    if quantiles is None:
        quantiles = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

    if verbose:
        print(f"Wavelet Quantile Mediation/Moderation  (wf={wavelet}, J={J})")
        print(f"  {main_name} → {mod_name} → {dep_name}  (Mediation)")
        print(f"  {main_name}×{mod_name} → {dep_name}  (Moderation)")

    # Decompose
    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)
    det_x, _ = modwt_mra(x, wavelet=wavelet, level=J)
    det_z, _ = modwt_mra(z, wavelet=wavelet, level=J)

    bands_y = aggregate_bands(det_y)
    bands_x = aggregate_bands(det_x)
    bands_z = aggregate_bands(det_z)
    band_names = list(bands_y.keys())

    rec_direct, rec_int, rec_a, rec_b, rec_ind = [], [], [], [], []

    for band in band_names:
        yb = bands_y[band]
        xb = bands_x[band]
        zb = bands_z[band]
        xzb = xb * zb

        for tau in quantiles:
            # Direct: Y ~ X
            r = _qr_coef(yb, {"x": xb}, tau, n_boot)
            rec_direct.append({"quantile": tau, "band": band,
                               "beta": r["x"][0], "se": r["x"][1], "p_value": r["x"][2]})

            # Interaction: Y ~ X + Z + XZ  (report XZ coef)
            r = _qr_coef(yb, {"x": xb, "z": zb, "xz": xzb}, tau, n_boot)
            rec_int.append({"quantile": tau, "band": band,
                            "beta": r["xz"][0], "se": r["xz"][1], "p_value": r["xz"][2]})

            # Path a: Z ~ X
            r = _qr_coef(zb, {"x": xb}, tau, n_boot)
            a_beta, a_se, a_pval = r["x"]
            rec_a.append({"quantile": tau, "band": band,
                          "beta": a_beta, "se": a_se, "p_value": a_pval})

            # Path b: Y ~ X + Z (report Z coef)
            r = _qr_coef(yb, {"x": xb, "z": zb}, tau, n_boot)
            b_beta, b_se, b_pval = r["z"]
            rec_b.append({"quantile": tau, "band": band,
                          "beta": b_beta, "se": b_se, "p_value": b_pval})

            # Indirect: a × b
            indirect_beta = a_beta * b_beta
            joint_p = max(a_pval, b_pval) if not (np.isnan(a_pval) or np.isnan(b_pval)) else np.nan
            rec_ind.append({"quantile": tau, "band": band,
                            "beta": indirect_beta, "p_value": joint_p})

        if verbose:
            print(f"  ✓ Band {band} done")

    if verbose:
        print("  ✓ Mediation/Moderation completed.")

    return MediationResult(
        direct=pd.DataFrame(rec_direct),
        interaction=pd.DataFrame(rec_int),
        path_a=pd.DataFrame(rec_a),
        path_b=pd.DataFrame(rec_b),
        indirect=pd.DataFrame(rec_ind),
        quantiles=quantiles,
        bands=band_names,
        dep_name=dep_name,
        main_name=main_name,
        mod_name=mod_name,
        n_obs=n_obs,
    )
