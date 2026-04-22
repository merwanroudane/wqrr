"""
Quantile-on-Quantile (QQ) Regression.

Implements the methodology of Sim & Zhou (2015):
  "Oil prices, US stock return, and the dependence between their quantiles."
  Journal of Banking & Finance, 55, 1–12.

Converted from the R CRAN package ``QuantileOnQuantile`` (Roudane, 2024).
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional, List
from scipy import stats as sp_stats

try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    HAS_SM = True
except ImportError:
    HAS_SM = False

from .utils import check_function, safe_print


@dataclass
class QQResult:
    """Container for Quantile-on-Quantile regression results."""

    results: pd.DataFrame
    y_quantiles: np.ndarray
    x_quantiles: np.ndarray
    n_obs: int
    method: str = "Quantile-on-Quantile Regression"

    # ── Convenience methods ────────────────────────────────────────────

    def to_matrix(self, value="coefficient"):
        """
        Pivot results into a (len(y_quantiles) × len(x_quantiles)) matrix.

        Parameters
        ----------
        value : str
            One of 'coefficient', 'std_error', 't_value', 'p_value', 'r_squared'.

        Returns
        -------
        np.ndarray
        """
        df = self.results.dropna(subset=[value])
        mat = df.pivot(index="y_quantile", columns="x_quantile", values=value)
        return mat.values

    def to_dataframe(self):
        """Return full results DataFrame."""
        return self.results.copy()

    def summary(self):
        """Print a comprehensive summary."""
        r = self.results.dropna(subset=["coefficient"])
        n_sig05 = (r["p_value"] < 0.05).sum()
        n_sig01 = (r["p_value"] < 0.01).sum()

        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            "║   Quantile-on-Quantile Regression — Summary     ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            f"  Observations     : {self.n_obs}",
            f"  Y quantiles      : {len(self.y_quantiles)} levels  [{self.y_quantiles[0]:.2f} … {self.y_quantiles[-1]:.2f}]",
            f"  X quantiles      : {len(self.x_quantiles)} levels  [{self.x_quantiles[0]:.2f} … {self.x_quantiles[-1]:.2f}]",
            f"  Total cells      : {len(self.results)}",
            f"  Complete results : {len(r)}",
            "",
            "  ── Coefficient Statistics ──",
            f"    Mean   : {r['coefficient'].mean():.4f}",
            f"    Median : {r['coefficient'].median():.4f}",
            f"    Min    : {r['coefficient'].min():.4f}",
            f"    Max    : {r['coefficient'].max():.4f}",
            f"    Std    : {r['coefficient'].std():.4f}",
            "",
            "  ── R-squared ──",
            f"    Mean   : {r['r_squared'].mean():.4f}",
            f"    Median : {r['r_squared'].median():.4f}",
            "",
            "  ── Significance ──",
            f"    p < 0.05 : {n_sig05} / {len(r)}  ({100*n_sig05/max(len(r),1):.1f}%)",
            f"    p < 0.01 : {n_sig01} / {len(r)}  ({100*n_sig01/max(len(r),1):.1f}%)",
            "",
        ]
        safe_print("\n".join(lines))
        return self

    def export_csv(self, path, digits=4):
        """Export results to CSV."""
        df = self.results.copy()
        num_cols = ["coefficient", "std_error", "t_value", "p_value", "r_squared"]
        for c in num_cols:
            if c in df.columns:
                df[c] = df[c].round(digits)
        df.to_csv(path, index=False)

    def export_latex(self, value="coefficient", caption="QQ Regression Coefficients"):
        """Return a LaTeX table string of the coefficient matrix."""
        mat = self.to_matrix(value)
        y_q = sorted(self.results["y_quantile"].dropna().unique())
        x_q = sorted(self.results["x_quantile"].dropna().unique())

        header = " & ".join([f"{q:.2f}" for q in x_q])
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            r"\begin{tabular}{l" + "r" * len(x_q) + "}",
            r"\hline",
            r"$\theta \backslash \tau$ & " + header + r" \\",
            r"\hline",
        ]
        for i, yq in enumerate(y_q):
            if i < mat.shape[0]:
                vals = " & ".join([f"{v:.3f}" if not np.isnan(v) else "---" for v in mat[i]])
                lines.append(f"{yq:.2f} & {vals}" + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


def qq_regression(y, x,
                  y_quantiles=None,
                  x_quantiles=None,
                  min_obs=10,
                  se_method="boot",
                  n_boot=200,
                  verbose=True):
    """
    Quantile-on-Quantile Regression (Sim & Zhou, 2015).

    For each x-quantile τ, subsets data where x ≤ Q_x(τ) and runs
    quantile regression of y on x at each y-quantile θ.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    x : array-like
        Independent variable.
    y_quantiles : array-like or None
        Quantiles of y to estimate.  Default ``np.arange(0.05, 1.0, 0.05)``.
    x_quantiles : array-like or None
        Quantiles of x to condition on.  Default same as y_quantiles.
    min_obs : int
        Minimum observations for QR; below this a correlation fallback is used.
    se_method : str
        Standard-error method for quantreg: 'boot', 'iid', 'robust'.
    n_boot : int
        Bootstrap replications when ``se_method='boot'``.
    verbose : bool
        Print progress.

    Returns
    -------
    QQResult
    """
    if not HAS_SM:
        raise ImportError("statsmodels is required.  pip install statsmodels")

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if len(y) != len(x):
        raise ValueError("y and x must have equal length.")

    # Drop missing
    mask = np.isfinite(y) & np.isfinite(x)
    y_data, x_data = y[mask], x[mask]
    n_obs = len(y_data)

    if n_obs < 20:
        raise ValueError(f"Need ≥ 20 observations, got {n_obs}.")

    if y_quantiles is None:
        y_quantiles = np.arange(0.05, 1.0, 0.05)
    if x_quantiles is None:
        x_quantiles = np.arange(0.05, 1.0, 0.05)

    y_quantiles = np.asarray(y_quantiles)
    x_quantiles = np.asarray(x_quantiles)

    if verbose:
        print(f"Running Quantile-on-Quantile Regression …")
        print(f"  n = {n_obs},  Y quantiles = {len(y_quantiles)},  X quantiles = {len(x_quantiles)}")

    records = []
    total = len(y_quantiles) * len(x_quantiles)
    done = 0

    for x_tau in x_quantiles:
        x_thresh = np.quantile(x_data, x_tau)
        sel = x_data <= x_thresh
        y_sub = y_data[sel]
        x_sub = x_data[sel]

        for y_theta in y_quantiles:
            done += 1
            rec = {
                "y_quantile": round(y_theta, 4),
                "x_quantile": round(x_tau, 4),
                "coefficient": np.nan,
                "std_error": np.nan,
                "t_value": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "method": None,
            }
            try:
                if len(y_sub) >= min_obs:
                    X_mat = sm.add_constant(x_sub)
                    model = QuantReg(y_sub, X_mat)
                    res = model.fit(q=y_theta, max_iter=1000, p_tol=1e-5)

                    coef = res.params[1]
                    bse = res.bse[1] if res.bse is not None and len(res.bse) > 1 else np.nan
                    tval = res.tvalues[1] if res.tvalues is not None and len(res.tvalues) > 1 else np.nan
                    pval = res.pvalues[1] if res.pvalues is not None and len(res.pvalues) > 1 else np.nan

                    # Pseudo R²
                    y_pred = res.predict(X_mat)
                    rho_model = check_function(y_sub - y_pred, y_theta).sum()
                    y_med = np.quantile(y_sub, y_theta)
                    rho_null = check_function(y_sub - y_med, y_theta).sum()
                    r2 = max(0.0, 1 - rho_model / rho_null) if rho_null != 0 else 0.0

                    rec.update(coefficient=coef, std_error=bse,
                               t_value=tval, p_value=pval,
                               r_squared=r2, method="quantile_regression")
                else:
                    # Fallback: correlation approach
                    y_thresh = np.quantile(y_data, y_theta)
                    yb = (y_data <= y_thresh).astype(float)
                    xb = (x_data <= x_thresh).astype(float)
                    if yb.std() > 0 and xb.std() > 0:
                        corr = np.corrcoef(yb, xb)[0, 1]
                        nb = len(xb)
                        se = np.sqrt((1 - corr**2) / max(1, nb - 2))
                        tv = corr / se if se > 0 else np.nan
                        pv = 2 * sp_stats.t.sf(abs(tv), df=max(1, nb - 2))
                        rec.update(coefficient=corr, std_error=se,
                                   t_value=tv, p_value=pv,
                                   r_squared=corr ** 2, method="correlation")
            except Exception:
                pass

            records.append(rec)

            if verbose and done % max(1, total // 10) == 0:
                print(f"  Progress: {100 * done // total}%")

    if verbose:
        print("  ✓ QQ Regression completed.")

    df = pd.DataFrame(records)
    return QQResult(
        results=df,
        y_quantiles=y_quantiles,
        x_quantiles=x_quantiles,
        n_obs=n_obs,
    )
