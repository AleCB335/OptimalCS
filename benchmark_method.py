"""Traduccion a Python de Daniel.gms (Mayorga y Rios).

Programa lineal: minimizar la capacidad total de condensadores sincronos
    min  sum_i Q_i
    s.a. SCRb_j + sum_i DeltaSCR_ij * Q_i >= 5    para toda estacion j
         Q_i >= 0
Q_i en p.u. sobre 100 MVA. Se resuelve con HiGHS a traves de scipy.
"""
import numpy as np
from scipy.optimize import linprog

from datos_scr import BARRAS, ESTACIONES, SCRb, DELTA_SCR, SCR_MIN, S_BASE_MVA


def resolver():
    n = DELTA_SCR.shape[0]
    c = np.ones(n)
    # linprog usa A_ub x <= b_ub; la restriccion >= se multiplica por -1
    A_ub = -DELTA_SCR.T
    b_ub = SCRb - SCR_MIN
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, res.fun


def main():
    Q, Z = resolver()
    print("Modelo Mayorga y Rios (LP)")
    print(f"Z = sum Q = {Z:.4f} p.u. = {Z * S_BASE_MVA:.1f} MVAr\n")
    print("Capacidades instaladas (Q > 0):")
    for idx, q in enumerate(Q):
        if q > 1e-6:
            print(f"  i{idx + 1:<3} Barra {BARRAS[idx]:>2}: {q:.4f} p.u. = {q * S_BASE_MVA:.1f} MVAr")
    scr = SCRb + DELTA_SCR.T @ Q
    print("\nSCR resultante (aproximacion lineal):")
    for est, s0, s in zip(ESTACIONES, SCRb, scr):
        print(f"  {est}: {s0:6.2f} -> {s:7.3f}")


if __name__ == "__main__":
    main()
