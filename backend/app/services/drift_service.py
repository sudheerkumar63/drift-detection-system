import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel


def compute_ks_drift(reference_features: np.ndarray, current_features: np.ndarray) -> dict:
    ref = np.asarray(reference_features, dtype=float).ravel()
    cur = np.asarray(current_features, dtype=float).ravel()
    ks_stat, p_value = stats.ks_2samp(ref, cur)
    return {
        "statistic": float(ks_stat),
        "p_value": float(p_value),
        "status": "drift_detected" if p_value < 0.05 else "no_significant_drift",
        "method": "Kolmogorov-Smirnov (two-sample)"
    }


def compute_mmd(reference_features: np.ndarray, current_features: np.ndarray, gamma: float = 0.5) -> dict:
    ref = np.asarray(reference_features, dtype=float)
    cur = np.asarray(current_features, dtype=float)
    ref = ref.reshape(1, -1) if ref.ndim == 1 else ref
    cur = cur.reshape(1, -1) if cur.ndim == 1 else cur
    ref_kernel = rbf_kernel(ref, ref, gamma=gamma)
    cur_kernel = rbf_kernel(cur, cur, gamma=gamma)
    cross = rbf_kernel(ref, cur, gamma=gamma)
    raw_mmd = float(np.mean(ref_kernel) + np.mean(cur_kernel) - 2 * np.mean(cross))
    score = float(np.clip(raw_mmd, 0.0, 2.0) / 2.0)
    return {
        "score": score,
        "raw_score": raw_mmd,
        "method": "Maximum Mean Discrepancy"
    }


def run_drift_detection(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    threshold: float = 0.2,
    method: str = "combined",
) -> dict:
    reference = np.asarray(reference_features, dtype=float).ravel()
    current = np.asarray(current_features, dtype=float).ravel()
    if reference.size < 2 or current.size < 2:
        raise ValueError("Reference and current distributions must each contain at least two values.")
    if reference.size != current.size:
        raise ValueError("Reference and current distributions must have the same number of values.")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Reference and current distributions must contain only finite values.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    if method not in {"ks", "mmd", "combined"}:
        raise ValueError("Method must be 'ks', 'mmd', or 'combined'.")

    ks = compute_ks_drift(reference, current)
    mmd = compute_mmd(reference, current)
    if method == "ks":
        drift_score = ks["statistic"]
    elif method == "mmd":
        drift_score = mmd["score"]
    else:
        drift_score = max(ks["statistic"], mmd["score"])
    status = "warning" if drift_score >= threshold else "stable"
    return {
        "method": method,
        "reference_samples": int(reference.size),
        "current_samples": int(current.size),
        "ks_test": ks,
        "mmd": mmd,
        "drift_score": float(drift_score),
        "threshold": float(threshold),
        "status": status,
        "note": "The comparison uses per-class prediction probabilities. KS compares their marginal values; MMD compares the full probability vectors. This is a prediction-output drift signal, not a comparison of internal CNN embeddings or proof of operational safety.",
    }
