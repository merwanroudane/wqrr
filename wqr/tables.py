"""
Publication-quality tables for WQR results.

Supports LaTeX, HTML, and console output with significance stars.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _stars(p):
    """Significance stars from p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def results_table(df, value_col="beta", pval_col="p_value",
                  row_col="quantile", col_col="level", digits=4):
    """
    Create a formatted results table with significance stars.

    Parameters
    ----------
    df : pd.DataFrame
    value_col, pval_col, row_col, col_col : str
    digits : int

    Returns
    -------
    pd.DataFrame  with formatted strings.
    """
    rows = sorted(df[row_col].unique())
    cols = sorted(df[col_col].unique(), key=str)

    table = pd.DataFrame(index=rows, columns=cols, dtype=str)
    for _, r in df.iterrows():
        v = r.get(value_col, np.nan)
        p = r.get(pval_col, np.nan)
        s = _stars(p)
        if np.isnan(v):
            cell = "—"
        else:
            cell = f"{v:.{digits}f}{s}"
        table.at[r[row_col], r[col_col]] = cell

    table.index.name = row_col
    return table


def export_latex(df, value_col="beta", pval_col="p_value",
                 row_col="quantile", col_col="level",
                 caption="Results", label="tab:results", digits=4):
    """
    Export results as a journal-ready LaTeX table.

    Returns
    -------
    str : LaTeX code
    """
    tbl = results_table(df, value_col, pval_col, row_col, col_col, digits)
    cols = list(tbl.columns)
    n_cols = len(cols)

    header = " & ".join([str(c) for c in cols])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * n_cols + "}",
        r"\hline\hline",
        f"$\\tau$ & {header}" + r" \\",
        r"\hline",
    ]

    for idx in tbl.index:
        vals = " & ".join([str(tbl.at[idx, c]) for c in cols])
        lines.append(f"{idx} & {vals}" + r" \\")

    lines += [
        r"\hline\hline",
        r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\footnotesize Note: ***, **, * denote significance at 1\%, 5\%, 10\% levels.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summary_statistics(data, var_names=None):
    """
    Descriptive statistics table suitable for journals.

    Parameters
    ----------
    data : pd.DataFrame or dict of arrays
    var_names : list or None

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if var_names is None:
        var_names = list(data.columns)

    records = []
    for v in var_names:
        s = data[v].dropna()
        records.append({
            "Variable": v,
            "Mean": f"{s.mean():.4f}",
            "Std. Dev.": f"{s.std():.4f}",
            "Min": f"{s.min():.4f}",
            "Max": f"{s.max():.4f}",
            "Skewness": f"{s.skew():.4f}",
            "Kurtosis": f"{s.kurtosis():.4f}",
            "Obs.": str(len(s)),
        })
    return pd.DataFrame(records)
