"""
Multivariate Wavelet Quantile Regression (MWQR).

Supports multiple independent variables with MODWT band aggregation.

Based on Adebayo et al. (2025):
  "Unpacking policy ambiguities in residential and commercial renewable
   energy adoption: A novel multivariate wavelet quantile regression analysis"
  Applied Economics.  doi:10.1080/00036846.2025.2590632
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from .utils import modwt_mra, aggregate_bands, safe_print


@dataclass
class MWQRResult:
    """Container for Multivariate Wavelet Quantile Regression results."""
    coefficients: pd.DataFrame
    quantiles: np.ndarray
    bands: list
    dep_name: str
    indep_names: list
    n_obs: int
    wavelet: str
    J: int

    def get_variable(self, var_name):
        """Get results for a specific independent variable."""
        return self.coefficients[self.coefficients["variable"] == var_name].copy()

    def to_matrix(self, var_name):
        """Pivot results for one variable to (quantiles × bands) matrix."""
        sub = self.get_variable(var_name)
        return sub.pivot(index="quantile", columns="band", values="beta")

    def significance_matrix(self, var_name, alpha=0.05):
        """Stars matrix for one variable."""
        sub = self.get_variable(var_name)
        piv = sub.pivot(index="quantile", columns="band", values="p_value")
        stars = piv.copy()
        stars[piv >= 0.10] = ""
        stars[(piv < 0.10) & (piv >= 0.05)] = "*"
        stars[(piv < 0.05) & (piv >= 0.01)] = "**"
        stars[piv < 0.01] = "***"
        return stars

    def summary(self):
        """Print summary."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════╗",
            "║  Multivariate Wavelet Quantile Regression — Summary ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Dependent   : {self.dep_name}",
            f"  Regressors  : {', '.join(self.indep_names)}",
            f"  Wavelet     : {self.wavelet}   J = {self.J}",
            f"  Observations: {self.n_obs}",
            f"  Quantiles   : {len(self.quantiles)}",
            f"  Bands       : {self.bands}",
            "",
        ]
        for var in self.indep_names:
            sub = self.get_variable(var)
            for band in self.bands:
                bsub = sub[sub["band"] == band]
                nsig = (bsub["p_value"] < 0.05).sum()
                lines.append(f"  {var:10s} | {band:8s}  β̄={bsub['beta'].mean():+.4f}  sig(5%)={nsig}/{len(bsub)}")
        lines.append("")
        safe_print("\n".join(lines))
        return self


def multivariate_wqr(y, X_dict, quantiles=None, wavelet="la8", J=6,
                     n_boot=500, dep_name="Y", verbose=True):
    """
    Multivariate Wavelet Quantile Regression.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X_dict : dict
        ``{'VarName': array, …}``  independent variables.
    quantiles : array-like or None
        Quantile grid.
    wavelet : str
        Wavelet family (R name).
    J : int
        Decomposition levels.
    n_boot : int
        Bootstrap replications.
    dep_name : str
        Name of dependent variable for labelling.
    verbose : bool

    Returns
    -------
    MWQRResult
    """
    y = np.asarray(y, dtype=np.float64)
    var_names = list(X_dict.keys())
    X_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in X_dict.items()}
    n_obs = len(y)

    if quantiles is None:
        quantiles = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                              0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    quantiles = np.asarray(quantiles)

    if verbose:
        print(f"Multivariate WQR  (wf={wavelet}, J={J}, vars={var_names})")

    # MODWT decomposition
    det_y, _ = modwt_mra(y, wavelet=wavelet, level=J)
    det_X = {}
    for k, v in X_arrays.items():
        det_X[k], _ = modwt_mra(v, wavelet=wavelet, level=J)

    # Aggregate bands
    bands_y = aggregate_bands(det_y)
    bands_X = {k: aggregate_bands(det_X[k]) for k in var_names}
    band_names = list(bands_y.keys())

    records = []
    for band in band_names:
        yb = bands_y[band]
        # Build regressor matrix for this band
        X_cols = {}
        for k in var_names:
            X_cols[k] = bands_X[k][band]

        X_df = pd.DataFrame(X_cols)
        X_mat = sm.add_constant(X_df.values)

        for tau in quantiles:
            try:
                model = QuantReg(yb, X_mat)
                res = model.fit(q=tau, max_iter=1000, p_tol=1e-5)

                for vi, vname in enumerate(var_names):
                    idx = vi + 1  # skip constant
                    beta = res.params[idx]
                    se = res.bse[idx] if res.bse is not None else np.nan
                    pval = res.pvalues[idx] if res.pvalues is not None else np.nan

                    records.append({
                        "quantile": round(tau, 4),
                        "band": band,
                        "variable": vname,
                        "beta": beta,
                        "se": se,
                        "p_value": pval,
                    })
            except Exception:
                for vname in var_names:
                    records.append({
                        "quantile": round(tau, 4),
                        "band": band,
                        "variable": vname,
                        "beta": np.nan,
                        "se": np.nan,
                        "p_value": np.nan,
                    })

        if verbose:
            print(f"  ✓ Band {band} done")

    df = pd.DataFrame(records)
    return MWQRResult(
        coefficients=df,
        quantiles=quantiles,
        bands=band_names,
        dep_name=dep_name,
        indep_names=var_names,
        n_obs=n_obs,
        wavelet=wavelet,
        J=J,
    )
