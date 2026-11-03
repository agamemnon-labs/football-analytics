"""L2-regularised logistic regression, used to fit the emission models on external games only."""

import numpy as np
from scipy.optimize import minimize


def logistic(X, y, l2=1.0, weights=None):
    """Coefficients minimising mean log-loss + l2/2 |w|^2 (intercept, column 0, unpenalised).

    Deterministic: L-BFGS from a zero start on float64 data.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    penalty = np.full(X.shape[1], l2 / len(y) * 0 + l2)
    penalty[0] = 0.0

    def loss(beta):
        z = X @ beta
        # log(1 + exp(z)) - y z, computed stably
        lse = np.logaddexp(0.0, z)
        value = np.sum(w * (lse - y * z)) + 0.5 * np.sum(penalty * beta ** 2) / len(y)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (w * (p - y)) + penalty * beta / len(y)
        return value, grad

    result = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "gtol": 1e-8})
    return result.x
