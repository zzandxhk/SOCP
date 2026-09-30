# -*- coding: utf-8 -*-
"""SOCP relaxation gap check on IEEE 33-bus (standard Baran & Wu data).
Solves the Branch Flow Model with SOC relaxation and reports absolute and
relative cone residuals. A small absolute gap alone is not a general
exactness proof, especially on branches with near-zero flow.
"""
import numpy as np
import cvxpy as cp

# (fb, tb, R_ohm, X_ohm, P_load_kW, Q_load_kVar)  -- standard IEEE 33
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

V_BASE = 12.66e3      # V
S_BASE = 10e6         # VA
Z_BASE = V_BASE**2 / S_BASE

N = 33
# children map (radial tree rooted at bus 1)
children = {i: [] for i in range(1, N+1)}
edge_idx = {}
for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
    edge_idx[(fb, tb)] = k
    children[fb].append(tb)

R_pu = np.array([L[2] / Z_BASE for L in LINES])
X_pu = np.array([L[3] / Z_BASE for L in LINES])
P_L = np.array([L[4] * 1e3 / S_BASE for L in LINES])   # kW -> W -> p.u.
Q_L = np.array([L[5] * 1e3 / S_BASE for L in LINES])

P = cp.Variable(len(LINES))   # branch active flow fb->tb
Q = cp.Variable(len(LINES))   # branch reactive flow fb->tb
l = cp.Variable(len(LINES), nonneg=True)  # |I|^2
v = cp.Variable(N)            # |V|^2 per node

constraints = [v[0] == 1.0, v >= 0.85**2, v <= 1.15**2]

for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
    # DistFlow: power balance at tb (sum inflows = demand + outflows)
    outflow_expr = cp.sum([P[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0
    constraints += [P[k] - outflow_expr - l[k]*R_pu[k] == P_L[k]]
    qout = cp.sum([Q[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0
    constraints += [Q[k] - qout - l[k]*X_pu[k] == Q_L[k]]
    # voltage drop
    constraints += [v[tb-1] == v[fb-1] - 2*(R_pu[k]*P[k] + X_pu[k]*Q[k]) + l[k]*(R_pu[k]**2 + X_pu[k]**2)]
    # SOC cone: ||[2P; 2Q; v_fb - l]||_2 <= v_fb + l
    constraints += [cp.SOC(v[fb-1] + l[k], cp.hstack([2*P[k], 2*Q[k], v[fb-1] - l[k]]))]

objective = cp.Minimize(cp.sum(cp.multiply(R_pu, l)))  # minimize losses
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CLARABEL, verbose=False)

print("IEEE 33-bus SOCP relaxation check")
print("  solver status:", prob.status)
print("  objective (loss, p.u.):", round(prob.value, 8))
if prob.status in ("optimal", "optimal_inaccurate"):
    gaps = []
    for k, (fb, tb, R, X, Pl, Ql) in enumerate(LINES):
        lhs = l[k].value * v[fb-1].value          # v_i * l_ij
        rhs = P[k].value**2 + Q[k].value**2
        gaps.append((tb, lhs - rhs, (lhs - rhs) / max(lhs, 1e-12)))
    max_abs = max(g[1] for g in gaps)
    max_rel = max(g[2] for g in gaps)
    print("  max |v*l - (P^2+Q^2)|:", f"{max_abs:.3e}")
    print("  max relative gap     :", f"{max_rel:.3e}")
    tight = sum(1 for g in gaps if g[2] < 1e-6)
    print(f"  branches with rel gap < 1e-6: {tight}/{len(LINES)}")
    print("  => SOC relaxation is", "EXACT (tight) on this radial network" if max_rel < 1e-5 else "NOT tight")
else:
    print("  solver failed:", prob.status)
