# -*- coding: utf-8 -*-
"""IEEE 33-bus: SOCP relaxation tightness under LAPLACE-noised microgrid injections.

This is a separate network-input sensitivity experiment. It adds Laplace draws
to fixed baseline nodal loads at buses 12/22/32, with scale equal to 5%/10%/20%
of the corresponding active and reactive baseline loads. For each setting, 10
independent draws are solved and the maximum absolute relaxation gap
|v*l - (P^2+Q^2)| is reported. These percentages are not privacy budgets.

Honest scope: this is a power-flow-level representation of noised microgrid
injections, not a step-by-step reproduction of the iterative DP-ADMM exchanges.
"""
import numpy as np
import cvxpy as cp

LINES = [
    (1,2,0.0922,0.0470,100,60),(2,3,0.4930,0.2511,90,40),(3,4,0.3660,0.1864,120,80),
    (4,5,0.3811,0.1941,60,30),(5,6,0.8190,0.7070,60,20),(6,7,0.1872,0.6188,200,100),
    (7,8,0.7114,0.2351,200,100),(8,9,1.0300,0.7400,60,20),(9,10,1.0440,0.7400,60,20),
    (10,11,0.1966,0.0650,45,30),(11,12,0.3744,0.1238,60,35),(12,13,1.4680,1.1550,60,35),
    (13,14,0.5416,0.7129,120,80),(14,15,0.5910,0.5260,60,10),(15,16,0.7463,0.5450,60,20),
    (16,17,1.2890,1.7210,60,20),(17,18,0.7320,0.5740,90,40),(2,19,0.1640,0.1565,90,40),
    (19,20,1.5042,1.3554,90,40),(20,21,0.4095,0.4784,90,40),(21,22,0.7089,0.9373,90,40),
    (3,23,0.4512,0.3083,90,50),(23,24,0.8980,0.7091,420,200),(24,25,0.8960,0.7011,420,200),
    (6,26,0.2030,0.1034,60,25),(26,27,0.2842,0.1447,60,25),(27,28,1.0590,0.9337,60,20),
    (28,29,0.8042,0.7006,120,70),(29,30,0.5075,0.2585,200,600),(30,31,0.9744,0.9630,150,70),
    (31,32,0.3105,0.3619,210,100),(32,33,0.3410,0.5302,60,40),
]

V_BASE = 12.66e3
S_BASE = 10e6
Z_BASE = V_BASE**2 / S_BASE
N = 33

# Microgrid buses in the manuscript (1-indexed): MMG1=12, MMG2=22, MMG3=32
MG_BUSES = [12, 22, 32]

children = {i: [] for i in range(1, N+1)}
edge_idx = {}
for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
    edge_idx[(fb, tb)] = k
    children[fb].append(tb)

R_pu = np.array([L[2] / Z_BASE for L in LINES])
X_pu = np.array([L[3] / Z_BASE for L in LINES])
# base load per branch (demand at the FROM bus is actually at tb; see model below:
# the balance is written at tb, so P_L[k] is the load at node tb)
P_L0 = np.array([L[4] * 1e3 / S_BASE for L in LINES])
Q_L0 = np.array([L[5] * 1e3 / S_BASE for L in LINES])

# branch index ending at each bus (load node) -- for referencing MG nodes
branch_of_bus = {}
for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
    branch_of_bus[tb] = k
mg_branches = [branch_of_bus[b] for b in MG_BUSES]  # branch indices whose to-bus is an MG node

print("IEEE 33-bus SOCP relaxation under LAPLACE-noised microgrid injections")
print("MG nodes:", MG_BUSES, "| Laplace scale b as fraction of baseline nodal load")
print("=" * 78)


def solve(extra_P, extra_Q):
    """extra_P/extra_Q: arrays added to the nodal loads (same length as branches)."""
    P = cp.Variable(len(LINES))
    Q = cp.Variable(len(LINES))
    l = cp.Variable(len(LINES), nonneg=True)
    v = cp.Variable(N)
    cons = [v[0] == 1.0, v >= 0.85**2, v <= 1.15**2]
    for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
        out_p = cp.sum([P[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0
        out_q = cp.sum([Q[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0
        cons += [P[k] - out_p - l[k]*R_pu[k] == P_L0[k] + extra_P[k]]
        cons += [Q[k] - out_q - l[k]*X_pu[k] == Q_L0[k] + extra_Q[k]]
        cons += [v[tb-1] == v[fb-1] - 2*(R_pu[k]*P[k] + X_pu[k]*Q[k]) + l[k]*(R_pu[k]**2 + X_pu[k]**2)]
        cons += [cp.SOC(v[fb-1] + l[k], cp.hstack([2*P[k], 2*Q[k], v[fb-1] - l[k]]))]
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(R_pu, l))), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None
    gaps = []
    for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
        lhs = l[k].value * v[fb-1].value
        rhs = P[k].value**2 + Q[k].value**2
        gaps.append(abs(lhs - rhs))
    return max(gaps), float(prob.value)


# baseline (no noise)
g, loss = solve(np.zeros(len(LINES)), np.zeros(len(LINES)))
print("baseline (no noise)              : max gap = %.3e p.u. | loss = %.1f kW" % (g, loss*S_BASE/1e3))

rng = np.random.default_rng(2026)
# Baseline nodal loads at the three selected buses (p.u.)
mg_P = np.array([P_L0[b] for b in mg_branches])
mg_Q = np.array([Q_L0[b] for b in mg_branches])

for frac, tag in [(0.05, "5% baseline load"), (0.10, "10% baseline load"), (0.20, "20% baseline load")]:
    gaps, losses = [], []
    for trial in range(10):
        etaP = rng.laplace(0.0, frac, size=3) * mg_P      # Lap(0, b), b = frac * |P_MG|
        etaQ = rng.laplace(0.0, frac, size=3) * mg_Q
        extra_P = np.zeros(len(LINES)); extra_Q = np.zeros(len(LINES))
        for i, b in enumerate(mg_branches):
            extra_P[b] += etaP[i]
            extra_Q[b] += etaQ[i]
        g, loss = solve(extra_P, extra_Q)
        if g is not None:
            gaps.append(g); losses.append(loss)
    gaps = np.array(gaps); losses = np.array(losses)
    print("Laplace b=%.0f%% (%s)          : max gap mean = %.3e (worst %.3e) | loss = %.1f +- %.2f kW"
          % (frac*100, tag, gaps.mean(), gaps.max(), losses.mean()*S_BASE/1e3, losses.std()*S_BASE/1e3))
