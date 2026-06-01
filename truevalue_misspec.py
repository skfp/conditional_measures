import argparse

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm


# ---------------------------------------------------------------------------
# True conditional quantile functions
# ---------------------------------------------------------------------------

def true_Q_3a(p, alpha, beta, c, c2, x):
    """Q_Y(p|x) for the quadratic DGP."""
    return np.exp(alpha + beta * (np.log(p / (1 - p)) + c * p * x + c2 * p * x**2))


def true_Q_3b(p, mu0, sigma0, sigma1, x):
    """Q_Y(p|x) for the log-normal heteroscedastic DGP."""
    return np.exp(mu0 + (sigma0 + sigma1 * x) * norm.ppf(p))


def true_Q_3c(p, c_bell, x):
    """Q_Y(p|x) for the non-monotone-beta1 DGP."""
    return np.exp(np.log(p / (1 - p)) + c_bell * 4 * p * (1 - p) * x)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def compute_true_indices(Q_func):
    """Numerically integrate qZI and qDI for a given x-fixed quantile function Q_func(p)."""
    def qZ(p):
        return 1 - Q_func(p / 2) / Q_func(p / 2 + 0.5)

    def qD(p):
        return 1 - Q_func(p / 2) / Q_func(1 - p / 2)

    qZI, _ = quad(qZ, 0, 1, limit=200)
    qDI, _ = quad(qD, 0, 1, limit=200)
    return qZI, qDI


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", choices=['3a', '3b', '3c'], required=True)
    # 3a
    parser.add_argument("--alpha",  type=float, default=0.5)
    parser.add_argument("--beta",   type=float, default=0.2)
    parser.add_argument("--c",      type=float, default=0.5)
    parser.add_argument("--c2",     type=float, default=0.005)
    # 3b
    parser.add_argument("--mu0",    type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument("--sigma1", type=float, default=0.02)
    # 3c
    parser.add_argument("--c_bell", type=float, default=0.05)
    # common
    parser.add_argument("--xmax",   type=float, default=30)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    xs = np.arange(1, 31, 1)

    qZI_list, qDI_list = [], []
    for x_val in xs:
        if args.dgp == '3a':
            Q = lambda p, x=x_val: true_Q_3a(p, args.alpha, args.beta,
                                              args.c, args.c2, x)
        elif args.dgp == '3b':
            Q = lambda p, x=x_val: true_Q_3b(p, args.mu0, args.sigma0,
                                              args.sigma1, x)
        elif args.dgp == '3c':
            Q = lambda p, x=x_val: true_Q_3c(p, args.c_bell, x)

        qZI, qDI = compute_true_indices(Q)
        qZI_list.append(qZI)
        qDI_list.append(qDI)

    if args.dgp == '3a':
        param_str = f"a={args.alpha}b={args.beta}_c={args.c}_c2={args.c2}"
    elif args.dgp == '3b':
        param_str = f"mu0={args.mu0}_s0={args.sigma0}_s1={args.sigma1}"
    else:
        param_str = f"cbell={args.c_bell}"

    outfile = f"{args.output}_dgp={args.dgp}_{param_str}_xmax={args.xmax}.csv"
    pd.DataFrame({'xs': xs, 'qZI': qZI_list, 'qDI': qDI_list}).to_csv(outfile, index=False)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
