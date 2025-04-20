import argparse
import pickle

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.special as sp
import cvxpy as cp
from numba import jit
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from scipy.integrate import simpson
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
from scipy.interpolate import interp1d

# set seed first
np.random.seed(100)

MINIMIZE_ALGORITHM = 'Nelder-Mead'  # Nelder-Mead, other option is SLSQP
MINIMIZE_ALGORITHM_BOUNDS = 'Nelder-Mead'  # Nelder-Mead, other option is SLSQP
MINIMIZE_TOL = 1e-5  # default is 1e-4
B10_NUMBER_OF_QUANTILES = 100

# final3
qs = np.arange(0.01, 1, 0.01)
qs2 = np.arange(0.02, 0.99, 0.01)
qs2_01 = np.arange(0.02, 0.99, 0.01)
qs2_01 = np.insert(qs2_01, 0, 0)
qs2_01 = np.insert(qs2_01, len(qs2_01), 1)
qshalf = np.arange(0.01, 0.5, 0.01)

epsilon = 0.000001
xlist = [1, 5, 10, 15, 20, 25, 30]

solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"

# rpy2 initialization
quantreg = importr('quantreg')
stats = importr('stats')

def gen_sample(alpha, beta, c, n, xmax):
    x = np.random.uniform(0, xmax, n)
    kappa = c * x
    u = np.random.uniform(0, 1, n)
    y = alpha + beta * (np.log(u / (1 - u)) + kappa * u)
    d = pd.DataFrame({"x": x, "y": y})
    return d

# approximations of |x|
@jit(forceobj=True)
def approx_g(tau, theta_tau):
    y = theta_tau / tau
    return tau * (-sp.log_expit(-y) - sp.log_expit(y))

@jit(nopython=True)
def approx_h(tau, theta_tau):
    return theta_tau * np.tanh(theta_tau / tau)

def approx_f(tau, theta_tau):
    return (approx_g(tau, theta_tau) + approx_h(tau, theta_tau)) / 2.0

def find_starting_point(p, x, y):
    x_min = np.min(x)
    x_max = np.max(x)
    data_subset_x_min = y[x == x_min]
    data_subset_x_max = y[x == x_max]
    sp0 = np.quantile(data_subset_x_min, p)
    x_min_q = sp0
    x_max_q = np.quantile(data_subset_x_max, p)
    sp1 = (x_max_q - x_min_q) / (x_max - x_min)
    return [sp0, sp1]

def Q_regr_bounds(p, x, y, bounds, starting_point=None):
    #     sp=find_starting_point(p, x, y)
    x = np.squeeze(x)
    if not starting_point:
        starting_point = [bounds[0][0], bounds[1][0]]
    def f_(beta):
        b0, b1 = beta
        theta_tau = b0 + b1*x
        return np.mean(np.abs(y - theta_tau) + (2*p-1)*(y-theta_tau))

    theta_tau_est = minimize(f_, x0=starting_point, method=MINIMIZE_ALGORITHM_BOUNDS, bounds=bounds, tol=MINIMIZE_TOL).x
    return list(theta_tau_est)

def A_regr(p, x, y, tau, starting_point=[0, 0], approx_fun=approx_f):
    x = np.squeeze(x)
    def f_(beta):
        b0, b1 = beta
        theta_tau = b0 + b1*x
        return np.mean(approx_fun(tau, y-theta_tau) + (2*p-1)*(y-theta_tau))
    #theta_tau_est = minimize(f_, x0=starting_point, method="SLSQP").x
    theta_tau_est = minimize(f_, x0=starting_point, method=MINIMIZE_ALGORITHM, tol=MINIMIZE_TOL).x
    return list(theta_tau_est)

def qZ(x, qs, qs2, beta0, beta1):
    qs = np.around(qs, decimals=2)
    beta0_interp = interp1d(qs, beta0)
    beta1_interp = interp1d(qs, beta1)
    qz_results = 1 - np.exp(beta0_interp(qs2/2) + beta1_interp(qs2/2) * x) / np.exp(beta0_interp(0.5 + qs2/2) + beta1_interp(0.5 + qs2/2) * x)
    qz_results = np.insert(qz_results, 0, 1)
    qz_results = np.insert(qz_results, len(qz_results), 1)
    return qz_results

def qD(x, qs, qs2, beta0, beta1):
    qs = np.around(qs, decimals=2)
    beta0_interp = interp1d(qs, beta0)
    beta1_interp = interp1d(qs, beta1)
    qd_results = 1 - np.exp(beta0_interp(qs2/2) + beta1_interp(qs2/2) * x) / np.exp(beta0_interp(1 - qs2/2) + beta1_interp(1 - qs2/2) * x)
    qd_results = np.insert(qd_results, 0, 1)
    qd_results = np.insert(qd_results, len(qd_results), 0)
    return qd_results

def qZ_on_Qs(x, qs, qs2, beta0, beta1, iso_flag):
    qs = np.around(qs, decimals=2)
    Qs = np.exp(beta0 + beta1 * x)
    if iso_flag == True:
        Qs_iso = IsotonicRegression().fit_transform(qs, Qs)
        Qs_interp = interp1d(qs, Qs_iso)
    else:
        Qs_interp = interp1d(qs, Qs)
    qz_results = 1 - Qs_interp(qs2/2)/Qs_interp(qs2/2+0.5)
    qz_results = np.insert(qz_results, 0, 1)
    qz_results = np.insert(qz_results, len(qz_results), 1)
    return qz_results

def qD_on_Qs(x, qs, qs2, beta0, beta1, iso_flag):
    qs = np.around(qs, decimals=2)
    Qs = np.exp(beta0 + beta1 * x)
    if iso_flag == True:
        Qs_iso = IsotonicRegression().fit_transform(qs, Qs)
        Qs_interp = interp1d(qs, Qs_iso)
    else:
        Qs_interp = interp1d(qs, Qs)
    qd_results = 1 - Qs_interp(qs2/2)/Qs_interp(1-qs2/2)
    qd_results = np.insert(qd_results, 0, 1)
    qd_results = np.insert(qd_results, len(qd_results), 1)
    return qd_results

def estimate_betas_iso_oqr(data, qs):
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    beta0s = []
    beta1s = []
    for q in qs:
        qr = QuantileRegressor(quantile=q, alpha=0, solver=solver).fit(x, y)
        beta0s.append(qr.intercept_)
        beta1s.append(qr.coef_[0])
    return([beta0s, beta1s])

def estimate_indices_iso_oqr(data, qs, xs):
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    beta0s = []
    beta1s = []
    for q in qs:
        qr = QuantileRegressor(quantile=q, alpha=0, solver=solver).fit(x, y)
        beta0s.append(qr.intercept_)
        beta1s.append(qr.coef_[0])
    beta0s_iso = IsotonicRegression().fit_transform(qs, beta0s)
    beta1s_iso = IsotonicRegression().fit_transform(qs, beta1s)
    qZ_preds = [qZ(x, qs, qs2, beta0s_iso, beta1s_iso) for x in xs]
    qD_preds = [qD(x, qs, qs2, beta0s_iso, beta1s_iso) for x in xs]
    # use Simpson rule instead of trapezoidal due to higher accuracy
    qZI_pred = [simpson(qZ_pred, x=qs2_01) for qZ_pred in qZ_preds]
    qDI_pred = [simpson(qD_pred, x=qs2_01) for qD_pred in qD_preds]

    return {"qZI": qZI_pred, "qDI": qDI_pred}

def quantile_loss(u, tau):
    # Piecewise linear convex form of the check function
    return cp.maximum(tau * u, (tau - 1) * u)

def quantile_regression_single_tau(x, y, tau, beta_init=None, direction="up"):
    # Perform quantile regression for a single tau, optionally using initial beta from previous step
    n, p = len(y), x.shape[1]
    z = np.column_stack([np.ones(n), x])  # Add a column of ones for the intercept
    # Define beta variable
    beta = cp.Variable(p + 1)
    # Define objective function (minimizing quantile loss)
    residuals = y - z @ beta
    objective = cp.sum(quantile_loss(residuals, tau))
    # Define constraints for non-crossing (depending on direction)
    constraints = []
    if beta_init is not None:
        if direction == "up":
            constraints.append(z @ beta >= z @ beta_init)  # Ensure non-crossing (upward)
        else:
            constraints.append(z @ beta <= z @ beta_init)  # Ensure non-crossing (downward)
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return beta.value

def scheme_1(x, y, taus):
    # Step 1: Estimate coefficients for the median (tau = 0.5)
    tau_median = 0.5
#     print(f"Estimating coefficients for tau = {tau_median}")
    beta_median = quantile_regression_single_tau(x, y, tau_median)
    # Prepare to estimate coefficients for other taus using SU and SD
    beta_hat = {tau_median: beta_median}
    # Step 2: Simplified Up (SU) for taus greater than 0.5
    for tau in sorted([t for t in taus if t > tau_median]):
        tau = np.round(tau, 2)
#         print(f"Estimating coefficients for tau = {tau} using Simplified Up")
        beta_hat[tau] = quantile_regression_single_tau(x, y, tau, beta_init=beta_hat[tau_median], direction="up")
        tau_median = tau  # Move the base point for the next estimation
    # Step 3: Simplified Down (SD) for taus less than 0.5
    tau_median = 0.5  # Reset to median
    for tau in sorted([t for t in taus if t < tau_median], reverse=True):
        tau = np.round(tau, 2)
#         print(f"Estimating coefficients for tau = {tau} using Simplified Down")
        beta_hat[tau] = quantile_regression_single_tau(x, y, tau, beta_init=beta_hat[tau_median], direction="down")
        tau_median = tau  # Move the base point for the next estimation
    return beta_hat

def scheme_2_u(x, y, taus, beta_hat):
    # Start from the smallest tau and apply Simplified Up (SU) again to refine the estimates
    taus = np.array([np.round(tau, 2) for tau in taus])
    beta_refined = {}
    # Use the initial coefficients from Scheme 1
    beta_refined[taus[0]] = beta_hat[taus[0]]
    for i in range(1, len(taus)):
        tau = taus[i]
        # Use the previous tau's coefficients to refine the current tau using Simplified Up (SU)
        beta_refined[tau] = quantile_regression_single_tau(x, y, tau, beta_init=beta_refined[taus[i - 1]], direction="up")
    return beta_refined

def scheme_2_d(x, y, taus, beta_hat):
    # Start from the largest tau and apply Simplified Down (SD) to refine the estimates
    taus = np.array([np.round(tau, 2) for tau in taus][::-1])
    beta_refined = {}
    # Use the initial coefficients from Scheme 1
    beta_refined[taus[0]] = beta_hat[taus[0]]
    for i in range(1, len(taus)):
        tau = taus[i]
#         print(f"Refining coefficients for tau = {tau} using Scheme 2(D)")
        # Use the previous tau's coefficients to refine the current tau using Simplified Down (SD)
        beta_refined[tau] = quantile_regression_single_tau(x, y, tau, beta_init=beta_refined[taus[i - 1]], direction="down")
    return beta_refined

def estimate_betas_b10(x, y, m):
    # Define the number of observations and quantiles
    n = len(y)
    p = np.linspace(1/m, (m-1)/m, m-1)  # quantile levels, e.g., [1/m, 2/m, ..., (m-1)/m]
    # Create z matrix (1, x)^T
    z = np.column_stack([np.ones(n), x])
    # Create a list of beta vectors (one for each quantile, excluding 0 and 1)
    beta = cp.Variable((z.shape[1], m-1))  # Shape (2, m-1)
    # Objective function
    objective = 0
    for t in range(m-1):
        tau = p[t]
        residuals = y - z @ beta[:, t]
        # Quantile loss: sum of piecewise linear terms
        objective += cp.sum(quantile_loss(residuals, tau))
    # Constraints: z^T beta_{tau_t} >= z^T beta_{tau_{t-1}} for each t
    constraints = [z @ beta[:, t] >= z @ beta[:, t-1] for t in range(1, m-1)]
    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()
    return beta.value

def estimate_indices_b10(data, qs, xs):
    x = np.array(data["x"])
    y = np.array(data["y"])
    beta_b10 = estimate_betas_b10(x, y, B10_NUMBER_OF_QUANTILES)
    beta0s = beta_b10[0]
    beta1s = beta_b10[1]
    qZ_preds = [qZ(x, qs, qs2, beta0s, beta1s) for x in xs]
    qD_preds = [qD(x, qs, qs2, beta0s, beta1s) for x in xs]
    # use Simpson rule instead of trapezoidal due to higher accuracy
    qZI_pred = [simpson(qZ_pred, x=qs2_01) for qZ_pred in qZ_preds]
    qDI_pred = [simpson(qD_pred, x=qs2_01) for qD_pred in qD_preds]
    return {"qZI": qZI_pred, "qDI": qDI_pred}

def estimate_indices_wl(data, qs, xs):
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    beta_wl1 = scheme_1(x, y, qs)
    beta0_wl1 = [beta_wl1[np.round(q, 2)][0] for q in qs]
    beta1_wl1 = [beta_wl1[np.round(q, 2)][1] for q in qs]
    qZ_preds_wl1 = [qZ(x, qs, qs2, beta0_wl1, beta1_wl1) for x in xs]
    qD_preds_wl1 = [qD(x, qs, qs2, beta0_wl1, beta1_wl1) for x in xs]
    # use Simpson rule instead of trapezoidal due to higher accuracy
    qZI_pred_wl1 = [simpson(qZ_pred_wl1, x=qs2_01) for qZ_pred_wl1 in qZ_preds_wl1]
    qDI_pred_wl1 = [simpson(qD_pred_wl1, x=qs2_01) for qD_pred_wl1 in qD_preds_wl1]
    return {"qZI_wl1": qZI_pred_wl1, "qDI_wl1": qDI_pred_wl1}

def estimate_indices_KB82(data, qs, xs):
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    beta0s = []
    beta1s = []
    for q in qs:
        qr = QuantileRegressor(quantile=q, alpha=0, solver=solver).fit(x, y)
        beta0s.append(qr.intercept_)
        beta1s.append(qr.coef_[0])
    beta0s = np.array(beta0s)
    beta1s = np.array(beta1s)
    qZ_preds = [qZ_on_Qs(x, qs, qs2, beta0s, beta1s, False) for x in xs]
    qD_preds = [qD_on_Qs(x, qs, qs2, beta0s, beta1s, False) for x in xs]
    # use Simpson rule instead of trapezoidal due to higher accuracy
    qZI_pred = [simpson(qZ_pred, x=qs2_01) for qZ_pred in qZ_preds]
    qDI_pred = [simpson(qD_pred, x=qs2_01) for qD_pred in qD_preds]
    return {"qZI": qZI_pred, "qDI": qDI_pred}

def estimate_indices_iso_approx(data, qs, xs, tau, approx_fun=approx_f):
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"])
    # compute with poor starting point
    betas = [A_regr(p=qs[0], x=x, y=y, tau=tau, approx_fun=approx_fun, starting_point=[0, 0])]
    for q in qs[1:]:
        # since qs are ordered use last estimate as starting point
        beta_regr = A_regr(p=q, x=x, y=y, tau=tau, approx_fun=approx_fun, starting_point=betas[-1])
        betas.append(beta_regr)
    betas_array = np.array(betas)
    beta0s = betas_array[:, 0]
    beta1s = betas_array[:, 1]
    beta0s_iso = IsotonicRegression().fit_transform(qs, beta0s)
    beta1s_iso = IsotonicRegression().fit_transform(qs, beta1s)
    qZ_preds = [qZ(x, qs, qs2, beta0s_iso, beta1s_iso) for x in xs]
    qD_preds = [qD(x, qs, qs2, beta0s_iso, beta1s_iso) for x in xs]
    # use Simpson rule instead of trapezoidal due to higher accuracy
    qZI_pred = [simpson(qZ_pred, x=qs2_01) for qZ_pred in qZ_preds]
    qDI_pred = [simpson(qD_pred, x=qs2_01) for qD_pred in qD_preds]
    return {"qZI": qZI_pred, "qDI": qDI_pred}


def estimate_beta_qrfnc_R(data, qshalf, epsilon, qs, xs, starting_q):
    # print('testing_qrfnc')
    def df_to_r(df):
        with (robjects.default_converter + pandas2ri.converter).context():
            r_from_pd_df = robjects.conversion.get_conversion().py2rpy(df)
        return r_from_pd_df
    def np_to_r(arr):
        with (robjects.default_converter + numpy2ri.converter).context():
            arr_from_np = robjects.conversion.get_conversion().py2rpy(arr)
        return arr_from_np
    data_l = data.copy()
    data_u = data
    data_l['y'] = -data_l['y']
    r_data_u = df_to_r(data_u)
    r_data_l = df_to_r(data_l)
    r_R = np_to_r(np.eye(2))
    if starting_q == 0.5:
        lqr_fit50 = quantreg.rq('y ~ x', tau=0.5, data=r_data_u, method='fnc', r=np_to_r(np.array([0, 0])), R=r_R)
        betas = np.asarray(stats.coef(lqr_fit50)).reshape(1, 2)
        for q in qshalf:
            q_u_fit = quantreg.rq('y ~ x', tau=np_to_r(0.5+q), data=r_data_u, method='fnc', r=np_to_r(betas[-1, :]), R=r_R)
            coefs_u = np.asarray(stats.coef(q_u_fit)).reshape(1, 2)
            q_l_fit = quantreg.rq('y ~ x', tau=np_to_r(0.5+q), data=r_data_l, method='fnc', r=np_to_r(-betas[0, :]), R=r_R)
            coefs_l = -np.asarray(stats.coef(q_l_fit)).reshape(1, 2)
            if np.all(np.isclose(coefs_l, betas[0])):
                betas = np.vstack([coefs_l - epsilon, betas])
            else:
                betas = np.vstack([coefs_l, betas])
            if np.all(np.isclose(coefs_u, betas[-1])):
                betas = np.vstack([betas, coefs_u + epsilon])
            else:
                betas = np.vstack([betas, coefs_u])
    if starting_q == 0.01:
        lqr_fit_min = quantreg.rq('y ~ x', tau=starting_q, data=r_data_u, method='fnc', r=np_to_r(np.array([-10, -10])), R=r_R)
        betas = np.asarray(stats.coef(lqr_fit_min)).reshape(1, 2)
        for q in qs[1::]:
            q_u_fit = quantreg.rq('y ~ x', tau=np_to_r(q), data=r_data_u, method='fnc', r=np_to_r(betas[-1, :]), R=r_R)
            coefs_u = np.asarray(stats.coef(q_u_fit)).reshape(1, 2)
            if np.all(np.isclose(coefs_u, betas[-1])):
                betas = np.vstack([betas, coefs_u + epsilon])
            else:
                betas = np.vstack([betas, coefs_u])
    if starting_q == 0.99:
        lqr_fit_max = quantreg.rq('y ~ x', tau=starting_q, data=r_data_l, method='fnc', r=np_to_r(np.array([0, 0])), R=r_R)
        betas = np.asarray(stats.coef(lqr_fit_max)).reshape(1, 2)
        # for q in qs[::-1]:
        for q in qs[1::]:
            q_l_fit = quantreg.rq('y ~ x', tau=np_to_r(q), data=r_data_l, method='fnc', r=np_to_r(-betas[0, :]), R=r_R)
            coefs_l = -np.asarray(stats.coef(q_l_fit)).reshape(1, 2)
            if np.all(np.isclose(coefs_l, betas[0])):
                betas = np.vstack([coefs_l - epsilon, betas])
            else:
                betas = np.vstack([coefs_l, betas])
    qZ_preds = [qZ(x, qs, qs2, np.array(betas[:, 0]), np.array(betas[:, 1])) for x in xs]
    qD_preds = [qD(x, qs, qs2, np.array(betas[:, 0]), np.array(betas[:, 1])) for x in xs]
    qZI_pred = [simpson(qZ_pred, x=qs2_01) for qZ_pred in qZ_preds]
    qDI_pred = [simpson(qD_pred, x=qs2_01) for qD_pred in qD_preds]
    return {"qZI": qZI_pred, "qDI": qDI_pred}


def compute_indices(alpha, beta, c, n, xmax, taus, ms):
    # generate sample from FLD distribution
    data = gen_sample(alpha, beta, c, n, xmax)
    beta25 = [i[0] for i in estimate_betas_iso_oqr(data, [0.25])]
    beta75 = [i[0] for i in estimate_betas_iso_oqr(data, [0.75])]
    x_mean = np.mean(data['x'])
    tau_iqr_1 = ((beta75[0]+beta75[1]*x_mean) - (beta25[0]+beta25[1]*x_mean))/np.sqrt(n)
    taus2 = [tau_iqr_1]
    iso_qr_pred = estimate_indices_iso_oqr(data, qs, xlist)
    iso_Af_preds2 = [estimate_indices_iso_approx(data, qs, xlist, tau) for tau in taus2]
    qrfnc_pred_R_middle = estimate_beta_qrfnc_R(data, qshalf, epsilon, qs, xlist, 0.5)
    KB82_pred = estimate_indices_KB82(data, qs, xlist)
    b10_pred = estimate_indices_b10(data, qs, xlist)
    wl_pred = estimate_indices_wl(data, qs, xlist)
    results = [
        iso_qr_pred["qZI"],
        iso_qr_pred["qDI"]
    ]
    for iso_Af_pred in iso_Af_preds2:
        results = results + [iso_Af_pred["qZI"], iso_Af_pred["qDI"]]
    results = results + [KB82_pred["qZI"], KB82_pred["qDI"]] 
    results = results + [
        b10_pred["qZI"],
        b10_pred["qDI"]
    ]
    results = results + [
        wl_pred["qZI_wl1"],
        wl_pred["qDI_wl1"]
    ]
    results = results + [
        qrfnc_pred_R_middle["qZI"],
        qrfnc_pred_R_middle["qDI"],
        xlist,
    ]
    return results


def run(args):
    outputfilename = f"{args.output}_n={args.n}_a={args.alpha}b={args.beta}_c={args.c}_xmax={args.xmax}.csv"
    taus = args.taus_float_type
    ms=[]
    METHODS = ["iso_qr"] + ["iso_tau_IQR"] + ["KB82"] + ["b10"] + ["WL1"] + ["qrfnc_R"]
    ncol = 2 * len(METHODS) + 1
    column_names = []
    for method in METHODS:
        column_names += [f"{method}_{index}" for index in ["qZI", "qDI"]]
    column_names.append("xs")
    outputs = []
    for _ in range(args.mc):
        indices = compute_indices(args.alpha, args.beta, args.c, args.n, args.xmax, taus, ms)
        outputs.append(indices)

    # save results to pickle as well
    outputfilenamepickle = outputfilename.replace('.csv', '.pickle')
    with open(outputfilenamepickle, 'wb') as handle:
        pickle.dump(outputs, handle)

    # the code below is rather unclear
    outputs2 = [[0] for _ in range(ncol)]
    for i in range(len(outputs)):
        outputs2 = [outputs2[j] + outputs[i][j] for j in range(ncol)]
    outputs2 = [op[1::] for op in outputs2]
    outputs_dict = {column_names[i]: outputs2[i] for i in range(ncol)}
    outputs = pd.DataFrame(outputs_dict)
    outputs.to_csv(outputfilename, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Sample sizes")
    parser.add_argument("--mc", type=int, default=5, help="Number of repetitions")
    parser.add_argument("--alpha", type=float, default=0.5, help="Parameter alpha of FLD")
    parser.add_argument("--beta", type=float, default=0.2, help="Parameter beta of FLD")
    parser.add_argument("--c", type=float, default=0.5, help="Parameter c of FLD")
    parser.add_argument("--xmax", type=float, default=30, help="Parameter xmax of FLD")
    parser.add_argument('--taus-float-type', nargs='+', type=float, help="List of smoothness parameters tau")
    parser.add_argument("--output", type=str, required=True, help="Name of output file")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
