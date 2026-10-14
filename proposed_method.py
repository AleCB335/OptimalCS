"""Modelo entero con capacidades discretas:
    min  sum_i sum_k  (c1*Q_k) * X_ik
    s.a. SCRb_j + sum_i sum_k DeltaSCR_ij * Q_k * X_ik >= 5   para toda j
         sum_k X_ik = 1                                       para toda i
         X_ik binaria
Como Q_k es un parametro, el modelo es lineal en X (MILP), aunque en GAMS
se resolvia con "minlp". Se resuelve con HiGHS a traves de scipy.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from datos_scr import BARRAS, ESTACIONES, SCRb, DELTA_SCR, SCR_MIN, S_BASE_MVA

# Capacidades discretas: 0 a 6.0 p.u. en pasos de 0.25 (k1..k25)
Q = np.arange(25) * 0.25

# Coeficientes de costo, tal como estan en Lorena.gms
C1 = 4268800
C0 = 0


def resolver(c0=C0, c1=C1):
    n, m = DELTA_SCR.shape
    q = len(Q)
    # Variable X aplanada: indice i*q + k
    costo_k =  (c1 * Q )
    c = np.tile(costo_k, n)

    # Restricciones de SCR: sum_i sum_k DeltaSCR_ij Q_k X_ik >= SCR_MIN - SCRb_j
    A_scr = np.zeros((m, n * q))
    for i in range(n):
        A_scr[:, i * q:(i + 1) * q] = np.outer(DELTA_SCR[i, :], Q)
    restr_scr = LinearConstraint(A_scr, lb=SCR_MIN - SCRb, ub=np.inf)

    # Una sola capacidad por barra: sum_k X_ik = 1
    A_uno = np.kron(np.eye(n), np.ones(q))
    restr_uno = LinearConstraint(A_uno, lb=1, ub=1)

    res = milp(c, constraints=[restr_scr, restr_uno],
               integrality=np.ones(n * q), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(res.message)
    X = np.round(res.x).reshape(n, q)
    return X, res.fun


def main():
    X, Z = resolver()
    Q_barra = X @ Q
    print("Modelo Castro, De Oliveira-De Jesus y Rios (ILP)")
    print(f"c0 = {C0}, c1 = {C1}")
    print(f"Z (costo) = {Z:.4f}")
    print(f"Capacidad total = {Q_barra.sum():.2f} p.u. = {Q_barra.sum() * S_BASE_MVA:.0f} MVAr\n")
    print("Capacidades instaladas (Q > 0):")
    for idx, qi in enumerate(Q_barra):
        if qi > 1e-6:
            k = int(np.argmax(X[idx])) + 1
            print(f"  i{idx + 1:<3} Barra {BARRAS[idx]:>2}: k{k:<3} {qi:.2f} p.u. = {qi * S_BASE_MVA:.0f} MVAr")
    scr = SCRb + DELTA_SCR.T @ Q_barra
    print("\nSCR resultante (aproximacion lineal):")
    for est, s0, s in zip(ESTACIONES, SCRb, scr):
        print(f"  {est}: {s0:6.2f} -> {s:7.3f}")


if __name__ == "__main__":
    main()
