import argparse

import numpy as np
import pandas as pd
from scipy.integrate import quad


def qfld_exp(p, alpha, beta, c, x):
    return np.exp(alpha + beta * (np.log(p / (1 - p)) + c * x * p))

def compute_index_true_value(alpha, beta, c, xs, index_flag):
    indices = []
    def qZ(p):
        return 1 - qfld_exp(p / 2, alpha, beta, c, x) / qfld_exp(1 / 2 + p / 2, alpha, beta, c, x)
    def qD(p):
        return 1 - qfld_exp(p / 2, alpha, beta, c, x) / qfld_exp(1 - p / 2, alpha, beta, c, x)

    for x in xs:
        if index_flag == "qZI":
            i, _ = quad(qZ, 0, 1, limit=100)
        elif index_flag == "qDI":
            i, _ = quad(qD, 0, 1, limit=100)
        indices.append(i)
    return indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="Parameter alpha of FLD")
    parser.add_argument("--beta", type=float, default=0.2, help="Parameter beta of FLD")
    parser.add_argument("--c", type=float, default=0.5, help="Parameter c of FLD")
    parser.add_argument("--xmax", type=float, default=30, help="Parameter xmax of FLD")
    parser.add_argument("--output", type=str, required=True, help="Name of output file")
    args = parser.parse_args()
    outputfilename= f"{args.output}_a={args.alpha}b={args.beta}_c={args.c}_xmax={args.xmax}.csv"

    xs = np.arange(1, 31, 1)

    qZI = compute_index_true_value(args.alpha, args.beta, args.c, xs=xs, index_flag="qZI")
    qDI = compute_index_true_value(args.alpha, args.beta, args.c, xs=xs, index_flag="qDI")

    df = pd.DataFrame(
        {'xs': xs,
         'qZI': qZI,
         'qDI': qDI
         })
    df.to_csv(outputfilename, index=False)


if __name__ == "__main__":
    main()

