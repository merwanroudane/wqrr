"""
Shared utilities for the WQR package.
Wavelet decomposition, bandwidth selection, kernel helpers.
"""

import numpy as np
import pandas as pd
import pywt
from scipy import stats as sp_stats


# ── Wavelet Decomposition ──────────────────────────────────────────────────

def modwt_mra(x, wavelet="sym8", level=None, boundary="periodic"):
    """
    Maximal Overlap Discrete Wavelet Transform — Multiresolution Analysis.

    Mirrors R's ``waveslim::mra(x, wf, J, method='modwt')``.

    Parameters
    ----------
    x : array-like
        1-D signal.
    wavelet : str
        PyWavelets wavelet name.  Mapping from R names:
        la8 → sym4, la16 → sym8, d4 → db2, etc.
    level : int or None
        Decomposition depth (J).  If None uses ``pywt.swt_max_level(n)``.
    boundary : str
        'periodic' (zero-pad to next power of 2) or 'reflection'.

    Returns
    -------
    details : list of np.ndarray
        ``[D1, D2, …, DJ]``  detail coefficients at each level.
    smooth : np.ndarray
        ``SJ``  smooth (approximation) at the coarsest level.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Map common R wavelet names to pywt
    wf_map = {
        "la8": "sym4", "la16": "sym8", "la20": "sym10",
        "d4": "db2", "d6": "db3", "d8": "db4",
        "haar": "haar",
    }
    wname = wf_map.get(wavelet, wavelet)

    if level is None:
        level = pywt.swt_max_level(n)
    level = min(level, pywt.swt_max_level(n))

    # Stationary Wavelet Transform (= MODWT up to normalisation)
    coeffs = pywt.swt(x, wname, level=level, trim_approx=True)
    # coeffs[0] is approximation, coeffs[1:] are details from coarse to fine
    # We want details[0]=D1 (finest) … details[J-1]=DJ (coarsest)

    smooth = coeffs[0]
    details = list(reversed(coeffs[1:]))  # D1 (fine) first

    # Normalise by 1/sqrt(2^j) to get true MODWT scaling
    for j, d in enumerate(details):
        details[j] = d / np.sqrt(2)

    smooth = smooth / np.sqrt(2)

    return details, smooth


def aggregate_bands(details, band_spec=None):
    """
    Aggregate wavelet detail levels into Short / Medium / Long bands.

    Parameters
    ----------
    details : list of np.ndarray
        Detail levels D1, D2, …, DJ from ``modwt_mra``.
    band_spec : dict or None
        Mapping of band name → list of 0-based level indices.
        Defaults to ``{'Short': [0,1], 'Medium': [2,3], 'Long': [4,…]}``.

    Returns
    -------
    bands : dict
        ``{'Short': array, 'Medium': array, 'Long': array}``.
    """
    J = len(details)
    if band_spec is None:
        short_idx  = [i for i in range(min(2, J))]
        medium_idx = [i for i in range(2, min(4, J))]
        long_idx   = [i for i in range(4, J)]
        # Fallbacks
        if not medium_idx and J > 2:
            medium_idx = [2]
        if not long_idx and J > 3:
            long_idx = [J - 1]
        band_spec = {"Short": short_idx, "Medium": medium_idx, "Long": long_idx}

    bands = {}
    for name, indices in band_spec.items():
        if indices:
            bands[name] = np.sum([details[i] for i in indices], axis=0)
    return bands


# ── Bandwidth Selection ────────────────────────────────────────────────────

def silverman_bandwidth(x):
    """Silverman's plug-in bandwidth for Gaussian kernel."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = sp_stats.iqr(x)
    return 1.06 * min(s, iqr / 1.34) * n ** (-1 / 5)


def quantile_adjusted_bandwidth(h_base, tau):
    """
    Yu & Jones (1998) quantile-specific bandwidth adjustment.

    h(tau) = h_base * ((tau*(1-tau)) / phi(Phi^{-1}(tau))^2)^{1/5}
    """
    phi_val = sp_stats.norm.pdf(sp_stats.norm.ppf(tau))
    return h_base * ((tau * (1 - tau) / (phi_val ** 2)) ** 0.2)


# ── Kernel Helpers ─────────────────────────────────────────────────────────

def gaussian_kernel(u):
    """Standard Gaussian kernel K(u) = phi(u)."""
    return sp_stats.norm.pdf(u)


# ── Embed (Time-Series Lag Matrix) ─────────────────────────────────────────

def embed(x, dim=2):
    """
    Mirrors R's ``embed(x, dim)``.
    Returns an (n - dim + 1) × dim matrix where column 0 = x[t],
    column 1 = x[t-1], etc.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < dim:
        raise ValueError("Series too short for embedding dimension.")
    return np.column_stack([x[dim - 1 - i: n - i] for i in range(dim)])


# ── Check (Rho) Function for Pseudo R² ─────────────────────────────────────

def check_function(u, tau):
    """Quantile regression check (rho) function: u * (tau - I(u<0))."""
    return u * (tau - (u < 0).astype(float))


# ── R-style wavelet name mapping ───────────────────────────────────────────

R_WAVELET_MAP = {
    "la8": "sym4",
    "la16": "sym8",
    "la20": "sym10",
    "d4": "db2",
    "d6": "db3",
    "d8": "db4",
    "haar": "haar",
    "coif1": "coif1",
}

def map_wavelet(r_name):
    """Map R wavelet family name to pywt name."""
    return R_WAVELET_MAP.get(r_name, r_name)


# ── Safe Print (Windows encoding workaround) ──────────────────────────────

import sys as _sys

def safe_print(text):
    """Print text, replacing unencodable characters on Windows cp1252."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = _sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))
