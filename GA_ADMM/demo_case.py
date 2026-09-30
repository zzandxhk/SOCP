"""Deterministic demonstration inputs. These are NOT the manuscript's original data."""

from __future__ import annotations

import math


# IEEE 33-bus branch data: from, to, R (ohm), X (ohm), load at to-bus (kW, kvar).
IEEE33 = [
    (1, 2, .0922, .0470, 100, 60), (2, 3, .4930, .2511, 90, 40),
    (3, 4, .3660, .1864, 120, 80), (4, 5, .3811, .1941, 60, 30),
    (5, 6, .8190, .7070, 60, 20), (6, 7, .1872, .6188, 200, 100),
    (7, 8, .7114, .2351, 200, 100), (8, 9, 1.0300, .7400, 60, 20),
    (9, 10, 1.0440, .7400, 60, 20), (10, 11, .1966, .0650, 45, 30),
    (11, 12, .3744, .1238, 60, 35), (12, 13, 1.4680, 1.1550, 60, 35),
    (13, 14, .5416, .7129, 120, 80), (14, 15, .5910, .5260, 60, 10),
    (15, 16, .7463, .5450, 60, 20), (16, 17, 1.2890, 1.7210, 60, 20),
    (17, 18, .7320, .5740, 90, 40), (2, 19, .1640, .1565, 90, 40),
    (19, 20, 1.5042, 1.3554, 90, 40), (20, 21, .4095, .4784, 90, 40),
    (21, 22, .7089, .9373, 90, 40), (3, 23, .4512, .3083, 90, 50),
    (23, 24, .8980, .7091, 420, 200), (24, 25, .8960, .7011, 420, 200),
    (6, 26, .2030, .1034, 60, 25), (26, 27, .2842, .1447, 60, 25),
    (27, 28, 1.0590, .9337, 60, 20), (28, 29, .8042, .7006, 120, 70),
    (29, 30, .5075, .2585, 200, 600), (30, 31, .9744, .9630, 150, 70),
    (31, 32, .3105, .3619, 210, 100), (32, 33, .3410, .5302, 60, 40),
]


def make_demo_case() -> dict:
    """Construct a small, explicitly synthetic but structurally complete case."""
    lines = []
    for i, (fb, tb, r, x, pd, qd) in enumerate(IEEE33):
        # Standard IEEE33 data have no investment lengths or thermal upgrade data.
        length = round(max(.3, r / .4), 4)  # illustrative only
        base_limit = 8.0 if i == 0 else (4.0 if fb in (2, 3, 6) else 2.5)
        lines.append({
            "from": fb, "to": tb, "r_ohm": r, "x_ohm": x,
            "p_kw": pd, "q_kvar": qd, "length_km": length,
            "base_limit_mva": base_limit, "upgrade_limit_mva": 1.8 * base_limit,
            "upgrade_impedance_factor": .75,
        })

    hours = list(range(24))
    price = [150.0 if 17 <= h <= 21 else (50.0 if h <= 6 else 100.0) for h in hours]
    scenarios = []
    for name, days, load_factor, sun_factor, wind_factor in [
        ("spring", 92, .95, .85, 1.00),
        ("summer", 92, 1.10, 1.10, .85),
        ("autumn", 91, 1.00, .90, 1.05),
        ("winter", 90, 1.08, .65, 1.20),
    ]:
        load = [load_factor * (1 + .12 * math.sin(2 * math.pi * (h - 8) / 24)) for h in hours]
        pv = [sun_factor * max(0, math.sin(math.pi * (h - 6) / 12)) for h in hours]
        wind = [min(1.0, wind_factor * (.45 + .17 * math.cos(2 * math.pi * h / 24))) for h in hours]
        scenarios.append({
            "name": name, "days": days,
            "load_multiplier": load, "pv_capacity_factor": pv, "wind_capacity_factor": wind,
        })

    return {
        "data_provenance": "SYNTHETIC_DEMONSTRATION_NOT_ORIGINAL_RESULTS",
        "base_mva": 10.0, "base_kv": 12.66,
        "lines": lines,
        "sop_candidates": [[12, 22], [18, 33], [25, 29]],
        "sop_size_steps_mva": [0.0, .25, .50, .75, 1.0],
        "microgrids": [
            {"bus": 12, "load_mw": .30, "pv_mw": 0.0, "wind_mw": 1.5,
             "ess_mwh": .5, "ess_power_mw": .15, "dg_mw": .20},
            {"bus": 22, "load_mw": .34, "pv_mw": 1.0, "wind_mw": 0.0,
             "ess_mwh": .4, "ess_power_mw": .12, "dg_mw": .20},
            {"bus": 32, "load_mw": .40, "pv_mw": 1.2, "wind_mw": 0.0,
             "ess_mwh": .3, "ess_power_mw": .10, "dg_mw": .20},
        ],
        "scenarios": scenarios, "price_usd_per_mwh": price,
        "power_factor": .95, "voltage_min_pu": .90, "voltage_max_pu": 1.10,
        "trade_limit_mw": .60, "grid_import_limit_mw": 3.0,
        "dg_cost_usd_per_mwh": 90.0,
        "ess_throughput_cost_usd_per_mwh": 3.0,
        "curtailment_cost_usd_per_mwh": 8.0,
        "load_shed_cost_usd_per_mwh": 3000.0,
        "line_upgrade_cost_usd_per_km": 15000.0,
        "sop_fixed_cost_usd": 20000.0,
        "sop_variable_cost_usd_per_kva": 50.0,
        "capital_recovery_factor": .10,
        "noise_scale_mw": .155,
        "admm_rho": .5,
        "admm_max_iterations": 20,
        "admm_tolerance_mw": 1e-4,
        "ga_population": 12,
        "ga_generations": 8,
        "ga_crossover_probability": .9,
        "ga_mutation_probability": .05,
        "infeasibility_penalty_usd": 10_000_000.0,
    }
