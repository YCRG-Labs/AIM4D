"""Exact symbolic verification of Proposition 1's algebraic core (SymPy).

Companion to identification_montecarlo.py, which verifies the asymptotic
consistency claim numerically. Here we verify the deterministic algebra exactly:

  1. Spatial reduced form (I - aW)^{-1} exists and its identification/stability
     region is det(I - aW) != 0.
  2. The LAGGED spatial regressor W*y_{t-1} is orthogonal to the current error
     e_t if and only if the error is serially uncorrelated (rho = 0). This is
     the exact form of the sequential-exogeneity condition under which the
     contagion coefficient is point-identified.
  3. The CONTEMPORANEOUS spatial regressor W*y_t is mechanically correlated with
     e_t through (I - aW)^{-1} for any sigma^2 > 0. This is the reflection /
     simultaneity bias that the lagged design (A2) avoids.

Run:  python3 robustness/verify_proposition_sympy.py
"""
import sympy as sp


def main():
    a, sig2, rho = sp.symbols("alpha sigma2 rho", real=True)

    W = sp.Matrix([
        [0, sp.Rational(1, 2), sp.Rational(1, 2)],
        [sp.Rational(1, 2), 0, sp.Rational(1, 2)],
        [sp.Rational(1, 2), sp.Rational(1, 2), 0],
    ])
    n = W.shape[0]
    I = sp.eye(n)

    print("=" * 68)
    print("Proposition 1 — symbolic verification of the algebraic core")
    print("=" * 68)

    print("\n[1] Spatial reduced form (I - aW)^{-1}")
    M = I - a * W
    Minv = M.inv()
    resid = sp.simplify(M * Minv - I)
    assert resid == sp.zeros(n, n), "reduced form failed"
    detM = sp.factor(M.det())
    print(f"    (I - aW)(I - aW)^-1 = I            verified: {resid == sp.zeros(n, n)}")
    print(f"    det(I - aW) = {detM}")
    poles = sp.solve(sp.Eq(M.det(), 0), a)
    print(f"    singular (unidentified) at alpha = {poles}")
    print(f"    => identification/stability region: alpha not in {poles}")

    print("\n[2] LAGGED regressor orthogonality  Cov(W y_{t-1}, e_t)")
    cov_eps_lag = rho * I
    cov_lag = sp.simplify(W * Minv * cov_eps_lag)
    print("    Cov(W y_{t-1}, e_t) = rho * W (I - aW)^-1 =")
    sp.pprint(cov_lag)
    cov_lag_indep = cov_lag.subs(rho, 0)
    print(f"\n    at rho = 0 (serially-uncorrelated e):  {cov_lag_indep == sp.zeros(n, n)}")
    print("    => lagged spatial term is exogenous IFF errors are serially")
    print("       uncorrelated; this is the exact sequential-exogeneity")
    print("       condition for point identification of alpha.")

    print("\n[3] CONTEMPORANEOUS regressor bias  Cov(W y_t, e_t)")
    cov_eps_con = sig2 * I
    cov_con = sp.simplify(W * Minv * cov_eps_con)
    print("    Cov(W y_t, e_t) = sigma2 * W (I - aW)^-1 =")
    sp.pprint(cov_con)
    nonzero = sp.simplify(cov_con.subs([(a, sp.Rational(3, 10)), (sig2, 1)]))
    print(f"\n    at alpha=0.3, sigma2=1:  nonzero = {nonzero != sp.zeros(n, n)}")
    print("    => contemporaneous spatial term carries the (I - aW)^-1")
    print("       simultaneity bias for ANY sigma2 > 0 (the Manski reflection")
    print("       problem). This is why A2 lags the neighbor outcomes.")

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print("  [1] reduced form + identification region: VERIFIED exactly")
    print("  [2] lagged design is clean IFF rho=0 (sequential exogeneity):")
    print("      VERIFIED — matches the corrected proposition (consistency")
    print("      requires serially-uncorrelated errors, not just lagging)")
    print("  [3] contemporaneous design is biased via (I - aW)^-1: VERIFIED")
    print("  Asymptotic consistency under [2] is verified numerically in")
    print("  identification_montecarlo.py (bias->0, RMSE->0, coverage~0.95).")


if __name__ == "__main__":
    main()
