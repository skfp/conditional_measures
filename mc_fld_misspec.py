import argparse
import pickle

import numpy as np
import pandas as pd
import mc_fld_no_r as base

np.random.seed(100)


# ---------------------------------------------------------------------------
# DGP functions
# ---------------------------------------------------------------------------

def gen_sample_3a(alpha, beta, c, c2, n, xmax):
    """3a: quadratic-in-x conditional quantile function.
    log(Q_Y(p|x)) = alpha + beta*(logit(p) + c*p*x + c2*p*x^2)
    Extends the EFLD log-linear model with a quadratic x term.
    """
    x = np.random.uniform(0, xmax, n)
    u = np.random.uniform(0, 1, n)
    y = alpha + beta * (np.log(u / (1 - u)) + c * u * x + c2 * u * x**2)
    return pd.DataFrame({"x": x, "y": y})


def gen_sample_3b(mu0, sigma0, sigma1, n, xmax):
    """3b: log-normal with heteroscedastic scale.
    log(Y)|X=x ~ Normal(mu0, (sigma0 + sigma1*x)^2)
    Non-EFLD marginal shape; sigma(x) increases with x.
    """
    x = np.random.uniform(0, xmax, n)
    sigma_x = sigma0 + sigma1 * x
    y = mu0 + sigma_x * np.random.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y})


def gen_sample_3c(c_bell, n, xmax):
    """3c: non-monotone beta1(p).
    log(Q_Y(p|x)) = logit(p) + c_bell*4*p*(1-p)*x
    beta1(p) = c_bell*4*p*(1-p) is bell-shaped (max at p=0.5) -- not monotone.
    Validity constraint: c_bell <= 1/(u(1-u)) / (4*|1-2u|*xmax) ~ 0.089 for xmax=30.
    """
    x = np.random.uniform(0, xmax, n)
    u = np.random.uniform(0, 1, n)
    y = np.log(u / (1 - u)) + c_bell * 4 * u * (1 - u) * x
    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# compute_indices  (same estimation pipeline for all three DGPs)
# ---------------------------------------------------------------------------

def compute_indices(dgp_type, dgp_params, n, xmax):
    if dgp_type == '3a':
        data = gen_sample_3a(**dgp_params, n=n, xmax=xmax)
    elif dgp_type == '3b':
        data = gen_sample_3b(**dgp_params, n=n, xmax=xmax)
    elif dgp_type == '3c':
        data = gen_sample_3c(**dgp_params, n=n, xmax=xmax)
    else:
        raise ValueError(f"Unknown dgp_type: {dgp_type}")

    beta25 = [i[0] for i in base.estimate_betas_iso_oqr(data, [0.25])]
    beta75 = [i[0] for i in base.estimate_betas_iso_oqr(data, [0.75])]
    x_mean = np.mean(data['x'])
    tau_iqr = ((beta75[0] + beta75[1] * x_mean) -
               (beta25[0] + beta25[1] * x_mean)) / np.sqrt(n)

    iso_qr   = base.estimate_indices_iso_oqr(data, base.qs, base.xlist)
    iso_af   = base.estimate_indices_iso_approx(data, base.qs, base.xlist, tau_iqr)
    kb82     = base.estimate_indices_KB82(data, base.qs, base.xlist)
    b10      = base.estimate_indices_b10(data, base.qs, base.xlist)
    wl       = base.estimate_indices_wl(data, base.qs, base.xlist)
    qrr_linear = base.estimate_indices_qrr_linear(data, base.qs, base.xlist)
    strat    = base.estimate_indices_strat(data, base.qs, base.xlist)
    iqrr     = base.estimate_indices_iqrr(data, base.qs, base.xlist)

    results  = [iso_qr["qZI"],   iso_qr["qDI"]]
    results += [iso_af["qZI"],   iso_af["qDI"]]
    results += [kb82["qZI"],     kb82["qDI"]]
    results += [b10["qZI"],      b10["qDI"]]
    results += [wl["qZI_wl1"],   wl["qDI_wl1"]]
    results += [qrr_linear["qZI"], qrr_linear["qDI"]]
    results += [strat["qZI"],    strat["qDI"]]
    results += [iqrr["qZI"],     iqrr["qDI"], base.xlist]
    return results


# ---------------------------------------------------------------------------
# run / main
# ---------------------------------------------------------------------------

METHODS = ["iso_qr", "iso_tau_IQR", "KB82", "b10", "WL1", "qrr_linear", "strat_K5", "iqrr"]


def _dgp_params_from_args(args):
    if args.dgp == '3a':
        return dict(alpha=args.alpha, beta=args.beta, c=args.c, c2=args.c2)
    if args.dgp == '3b':
        return dict(mu0=args.mu0, sigma0=args.sigma0, sigma1=args.sigma1)
    if args.dgp == '3c':
        return dict(c_bell=args.c_bell)


def _param_str(args):
    if args.dgp == '3a':
        return f"a={args.alpha}b={args.beta}_c={args.c}_c2={args.c2}"
    if args.dgp == '3b':
        return f"mu0={args.mu0}_s0={args.sigma0}_s1={args.sigma1}"
    if args.dgp == '3c':
        return f"cbell={args.c_bell}"


def run(args):
    dgp_params = _dgp_params_from_args(args)
    param_str = _param_str(args)
    outputfilename = (f"{args.output}_dgp={args.dgp}_n={args.n}_"
                      f"{param_str}_xmax={args.xmax}.csv")

    ncol = 2 * len(METHODS) + 1
    column_names = [f"{m}_{idx}" for m in METHODS for idx in ["qZI", "qDI"]]
    column_names.append("xs")

    outputs = []
    for _ in range(args.mc):
        outputs.append(compute_indices(args.dgp, dgp_params, args.n, args.xmax))

    outputfilenamepickle = outputfilename.replace('.csv', '.pickle')
    with open(outputfilenamepickle, 'wb') as fh:
        pickle.dump(outputs, fh)

    outputs2 = [[0] for _ in range(ncol)]
    for i in range(len(outputs)):
        outputs2 = [outputs2[j] + outputs[i][j] for j in range(ncol)]
    outputs2 = [op[1:] for op in outputs2]
    outputs_dict = {column_names[i]: outputs2[i] for i in range(ncol)}
    pd.DataFrame(outputs_dict).to_csv(outputfilename, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", choices=['3a', '3b', '3c'], required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--mc", type=int, default=5)
    parser.add_argument("--xmax", type=float, default=30)
    parser.add_argument("--output", type=str, required=True)
    # 3a params
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--c2", type=float, default=0.005)
    # 3b params
    parser.add_argument("--mu0", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=0.2)
    parser.add_argument("--sigma1", type=float, default=0.02)
    # 3c params
    parser.add_argument("--c_bell", type=float, default=0.05)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
