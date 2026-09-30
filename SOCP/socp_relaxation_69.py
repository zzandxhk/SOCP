# -*- coding: utf-8 -*-
"""SOCP relaxation + scalability check on IEEE 69-bus (Matpower case69, Baran & Wu).
Loads in kW, branch R/X in Ohms; base 12.66 kV, 10 MVA.
"""
import numpy as np
import cvxpy as cp
import time

# bus -> (Pd_kW, Qd_kVar)  (zero loads omitted)
LOADS = {
 6:(2.6,2.2),7:(40.4,30),8:(75,54),9:(30,22),10:(28,19),11:(145,104),12:(145,104),
 13:(8,5.5),14:(8,5.5),16:(45.5,30),17:(60,35),18:(60,35),20:(1,0.6),21:(114,81),
 22:(5.3,3.5),24:(28,20),26:(14,10),27:(14,10),28:(26,18.6),29:(26,18.6),33:(14,10),
 34:(19.5,14),35:(6,4),36:(26,18.6),37:(26,18.6),39:(24,17),40:(24,17),41:(1.2,1),
 43:(6,4.3),45:(39.2,26.3),46:(39.2,26.3),48:(79,56.4),49:(384.7,274.5),50:(384.7,274.5),
 51:(40.5,28.3),52:(3.6,2.7),53:(4.3,3.5),54:(26.4,19),55:(24,17.2),59:(100,72),
 61:(1244,888),62:(32,23),64:(227,162),65:(59,42),66:(18,13),67:(18,13),68:(28,20),69:(28,20),
}

# (fb, tb, R_ohm, X_ohm)
BRANCHES = [
 (1,2,0.0005,0.0012),(2,3,0.0005,0.0012),(3,4,0.0015,0.0036),(4,5,0.0251,0.0294),
 (5,6,0.366,0.1864),(6,7,0.381,0.1941),(7,8,0.0922,0.047),(8,9,0.0493,0.0251),
 (9,10,0.819,0.2707),(10,11,0.1872,0.0619),(11,12,0.7114,0.2351),(12,13,1.03,0.34),
 (13,14,1.044,0.34),(14,15,1.058,0.3496),(15,16,0.1966,0.065),(16,17,0.3744,0.1238),
 (17,18,0.0047,0.0016),(18,19,0.3276,0.1083),(19,20,0.2106,0.069),(20,21,0.3416,0.1129),
 (21,22,0.014,0.0046),(22,23,0.1591,0.0526),(23,24,0.3463,0.1145),(24,25,0.7488,0.2475),
 (25,26,0.3089,0.1021),(26,27,0.1732,0.0572),(3,28,0.0044,0.0108),(28,29,0.064,0.1565),
 (29,30,0.3978,0.1315),(30,31,0.0702,0.0232),(31,32,0.351,0.116),(32,33,0.839,0.2816),
 (33,34,1.708,0.5646),(34,35,1.474,0.4873),(3,36,0.0044,0.0108),(36,37,0.064,0.1565),
 (37,38,0.1053,0.123),(38,39,0.0304,0.0355),(39,40,0.0018,0.0021),(40,41,0.7283,0.8509),
 (41,42,0.31,0.3623),(42,43,0.041,0.0478),(43,44,0.0092,0.0116),(44,45,0.1089,0.1373),
 (45,46,0.0009,0.0012),(4,47,0.0034,0.0084),(47,48,0.0851,0.2083),(48,49,0.2898,0.7091),
 (49,50,0.0822,0.2011),(8,51,0.0928,0.0473),(51,52,0.3319,0.114),(9,53,0.174,0.0886),
 (53,54,0.203,0.1034),(54,55,0.2842,0.1447),(55,56,0.2813,0.1433),(56,57,1.59,0.5337),
 (57,58,0.7837,0.263),(58,59,0.3042,0.1006),(59,60,0.3861,0.1172),(60,61,0.5075,0.2585),
 (61,62,0.0974,0.0496),(62,63,0.145,0.0738),(63,64,0.7105,0.3619),(64,65,1.041,0.5302),
 (11,66,0.2012,0.0611),(66,67,0.0047,0.0014),(12,68,0.7394,0.2444),(68,69,0.0047,0.0016),
]

V_BASE, S_BASE = 12.66e3, 10e6
Z_BASE = V_BASE**2 / S_BASE
N = 69
children = {i: [] for i in range(1, N+1)}
edge_idx = {}
for k, (fb, tb, R, X) in enumerate(BRANCHES):
    edge_idx[(fb, tb)] = k
    children[fb].append(tb)

R = np.array([b[2]/Z_BASE for b in BRANCHES])
X = np.array([b[3]/Z_BASE for b in BRANCHES])
PL = np.zeros(len(BRANCHES)); QL = np.zeros(len(BRANCHES))
for k, (fb, tb, Rr, Xx) in enumerate(BRANCHES):
    if tb in LOADS:
        PL[k] = LOADS[tb][0]*1e3/S_BASE
        QL[k] = LOADS[tb][1]*1e3/S_BASE

P = cp.Variable(len(BRANCHES)); Q = cp.Variable(len(BRANCHES))
l = cp.Variable(len(BRANCHES), nonneg=True); v = cp.Variable(N)
cons = [v[0] == 1.0, v >= 0.9**2, v <= 1.1**2]
for k, (fb, tb, Rr, Xx) in enumerate(BRANCHES):
    op = cp.sum([P[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0.0
    oq = cp.sum([Q[edge_idx[(tb, c)]] for c in children[tb]]) if children[tb] else 0.0
    cons.append(P[k] - op - l[k]*R[k] == PL[k])
    cons.append(Q[k] - oq - l[k]*X[k] == QL[k])
    cons.append(v[tb-1] == v[fb-1] - 2*(R[k]*P[k] + X[k]*Q[k]) + l[k]*(R[k]**2 + X[k]**2))
    cons.append(cp.SOC(v[fb-1] + l[k], cp.hstack([2*P[k], 2*Q[k], v[fb-1] - l[k]])))

prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(R, l))), cons)
t0 = time.time()
prob.solve(solver=cp.CLARABEL, verbose=False)
dt = time.time() - t0

print("IEEE 69-bus SOCP relaxation + scalability check")
print("  solver status:", prob.status)
print("  solve time (s):", f"{dt:.2f}")
print("  loss (p.u.):", round(prob.value, 6), "=", f"{prob.value*S_BASE/1e3:.1f} kW")
if prob.status in ("optimal", "optimal_inaccurate"):
    gaps = []
    for k, (fb, tb, Rr, Xx) in enumerate(BRANCHES):
        lhs = l[k].value * v[fb-1].value
        rhs = P[k].value**2 + Q[k].value**2
        gaps.append((lhs - rhs, (lhs - rhs)/max(lhs, 1e-12)))
    print("  max |v*l-(P^2+Q^2)| :", f"{max(g[0] for g in gaps):.3e}")
    print("  max relative gap    :", f"{max(g[1] for g in gaps):.3e}")
    tight = sum(1 for g in gaps if g[1] < 1e-5)
    print(f"  branches rel gap<1e-5: {tight}/{len(BRANCHES)}")
