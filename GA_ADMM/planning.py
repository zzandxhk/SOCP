"""Independent GA + Laplace-perturbed ADMM + SOCP planning implementation.

This code follows the model structure in the revised manuscript. It is a new
implementation, not the missing original program that generated its tables.
The bundled input is explicitly synthetic and must not be cited as a replication
of the reported Case I--III dollar values. See README.md before using results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from demo_case import make_demo_case


def validate_case(case: dict) -> None:
    """Fail early on missing or inconsistent scenario and network inputs."""
    required = ["lines", "sop_candidates", "microgrids", "scenarios", "price_usd_per_mwh"]
    for key in required:
        if key not in case:
            raise ValueError(f"Missing required input: {key}")
    lines = case["lines"]
    if len(lines) != 32 or len(case["microgrids"]) != 3:
        raise ValueError("This implementation expects the paper's 32-branch, 3-microgrid structure")
    buses = {1}
    for line in lines:
        if line["from"] not in buses or line["to"] in buses:
            raise ValueError("Lines must be a radial tree in parent-before-child order")
        buses.add(line["to"])
        for key in ("r_ohm", "x_ohm", "length_km", "base_limit_mva", "upgrade_limit_mva"):
            if float(line[key]) <= 0:
                raise ValueError(f"Line {line['from']}-{line['to']} has nonpositive {key}")
    if buses != set(range(1, 34)):
        raise ValueError("The network must contain buses 1 through 33")
    if len(case["sop_candidates"]) != 3 or len(case["sop_size_steps_mva"]) < 2:
        raise ValueError("Expected three SOP corridors and at least two size steps")
    if case["sop_size_steps_mva"][0] != 0 or any(
        b <= a for a, b in zip(case["sop_size_steps_mva"], case["sop_size_steps_mva"][1:])
    ):
        raise ValueError("SOP size steps must start at zero and increase")
    if len(case["price_usd_per_mwh"]) != 24:
        raise ValueError("Expected 24 hourly prices")
    if sum(int(s["days"]) for s in case["scenarios"]) != 365:
        raise ValueError("Representative scenario day counts must sum to 365")
    for scenario in case["scenarios"]:
        for key in ("load_multiplier", "pv_capacity_factor", "wind_capacity_factor"):
            if len(scenario[key]) != 24:
                raise ValueError(f"Scenario {scenario['name']} must contain 24 {key} values")
    for mg in case["microgrids"]:
        if mg["bus"] not in buses:
            raise ValueError("Microgrid bus is absent from network")
        for key in ("load_mw", "pv_mw", "wind_mw", "ess_mwh", "ess_power_mw", "dg_mw"):
            if float(mg[key]) < 0:
                raise ValueError(f"Negative microgrid {key}")


def scenario_inputs(case: dict, scenario: dict, hours: int) -> list[dict[str, np.ndarray]]:
    load_shape = np.asarray(scenario["load_multiplier"][:hours], dtype=float)
    pv_shape = np.asarray(scenario["pv_capacity_factor"][:hours], dtype=float)
    wind_shape = np.asarray(scenario["wind_capacity_factor"][:hours], dtype=float)
    return [
        {
            "load": float(mg["load_mw"]) * load_shape,
            "pv": float(mg["pv_mw"]) * pv_shape,
            "wind": float(mg["wind_mw"]) * wind_shape,
        }
        for mg in case["microgrids"]
    ]


class LocalMicrogrid:
    """One ADMM local dispatch with optional exact binary ESS exclusivity."""

    def __init__(self, case: dict, mg_id: int, profile: dict, hours: int,
                 binary_ess: bool):
        self.mg_id = mg_id
        self.neighbors = [n for n in range(3) if n != mg_id]
        self.hours = hours
        self.profile = profile
        self.binary_ess = binary_ess
        mg = case["microgrids"][mg_id]
        h = hours

        self.trade = cp.Variable((2, h))
        self.grid = cp.Variable(h, nonneg=True)
        self.dg = cp.Variable(h, nonneg=True)
        self.pv = cp.Variable(h, nonneg=True)
        self.wind = cp.Variable(h, nonneg=True)
        self.ch = cp.Variable(h, nonneg=True)
        self.dis = cp.Variable(h, nonneg=True)
        self.energy = cp.Variable(h + 1)
        self.shed = cp.Variable(h, nonneg=True)
        self.received = cp.Parameter((2, h), value=np.zeros((2, h)))
        self.dual = cp.Parameter((2, h), value=np.zeros((2, h)))

        if binary_ess:
            mode_ch = cp.Variable(h, boolean=True)
            mode_dis = cp.Variable(h, boolean=True)
            mode_constraints = [mode_ch + mode_dis <= 1]
        else:
            # Convex relaxation used only when a MIQP solver is unavailable.
            mode_ch = cp.Variable(h)
            mode_dis = cp.Variable(h)
            mode_constraints = [mode_ch >= 0, mode_dis >= 0,
                                mode_ch + mode_dis <= 1]

        e_max = float(mg["ess_mwh"])
        p_max = float(mg["ess_power_mw"])
        constraints = mode_constraints + [
            self.dg <= float(mg["dg_mw"]),
            self.pv <= profile["pv"],
            self.wind <= profile["wind"],
            self.shed <= profile["load"],
            self.grid <= float(case["grid_import_limit_mw"]),
            self.trade <= float(case["trade_limit_mw"]),
            self.trade >= -float(case["trade_limit_mw"]),
            self.ch <= p_max * mode_ch,
            self.dis <= p_max * mode_dis,
            self.energy[0] == .5 * e_max,
            self.energy >= 0,
            self.energy <= e_max,
            self.energy[h] >= self.energy[0],
            self.dg + self.pv + self.wind + self.dis - self.ch + self.grid
            == profile["load"] - self.shed + cp.sum(self.trade, axis=0),
        ]
        for t in range(h):
            constraints.append(
                self.energy[t + 1] == self.energy[t] + .95 * self.ch[t] - self.dis[t] / .95
            )
        price = np.asarray(case["price_usd_per_mwh"][:h], dtype=float)
        self.physical_cost = (
            cp.sum(cp.multiply(price, self.grid))
            + float(case["dg_cost_usd_per_mwh"]) * cp.sum(self.dg)
            + float(case["ess_throughput_cost_usd_per_mwh"]) * cp.sum(self.ch + self.dis)
            + float(case["curtailment_cost_usd_per_mwh"])
            * (float(np.sum(profile["pv"] + profile["wind"])) - cp.sum(self.pv + self.wind))
            + float(case["load_shed_cost_usd_per_mwh"]) * cp.sum(self.shed)
        )
        rho = float(case["admm_rho"])
        # Dollars are scaled by 1000 to make the penalty numerically comparable.
        objective = cp.Minimize(
            self.physical_cost / 1000.0
            + .5 * rho * cp.sum_squares(self.trade + self.received + self.dual)
        )
        self.problem = cp.Problem(objective, constraints)

    def solve(self, received: np.ndarray, dual: np.ndarray) -> dict:
        self.received.value = received
        self.dual.value = dual
        solver = "GUROBI" if self.binary_ess else "CLARABEL"
        try:
            self.problem.solve(solver=solver, warm_start=True, verbose=False)
        except cp.SolverError as exc:
            raise RuntimeError(f"Microgrid {self.mg_id + 1} solve failed with {solver}: {exc}") from exc
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Microgrid {self.mg_id + 1} status: {self.problem.status}")
        trade = np.asarray(self.trade.value, dtype=float)
        net_demand = np.asarray(self.profile["load"], dtype=float) - (
            np.asarray(self.dg.value) + np.asarray(self.pv.value)
            + np.asarray(self.wind.value) + np.asarray(self.dis.value)
            - np.asarray(self.ch.value) + np.asarray(self.shed.value)
        )
        balance_error = np.max(np.abs(
            np.asarray(self.dg.value) + np.asarray(self.pv.value)
            + np.asarray(self.wind.value) + np.asarray(self.dis.value)
            - np.asarray(self.ch.value) + np.asarray(self.grid.value)
            - np.asarray(self.profile["load"]) + np.asarray(self.shed.value)
            - np.sum(trade, axis=0)
        ))
        return {
            "trade": trade,
            "net_demand_mw": net_demand,
            "grid_import_mw": np.asarray(self.grid.value, dtype=float),
            "operating_cost_usd": float(self.physical_cost.value),
            "max_local_balance_error_mw": float(balance_error),
        }


def run_admm(case: dict, scenario: dict, hours: int, seed: int,
             binary_ess: bool, max_iter: int) -> dict:
    """Consensus ADMM with noisy directed trade messages."""
    agents = [LocalMicrogrid(case, n, p, hours, binary_ess)
              for n, p in enumerate(scenario_inputs(case, scenario, hours))]
    directed = [(n, m) for n in range(3) for m in range(3) if n != m]
    undirected = [(0, 1), (0, 2), (1, 2)]
    consensus = {edge: np.zeros(hours) for edge in undirected}
    dual = {edge: np.zeros(hours) for edge in directed}
    rng = np.random.default_rng(seed)
    rho = float(case["admm_rho"])
    b = float(case["noise_scale_mw"])
    tol = float(case["admm_tolerance_mw"])
    last = None
    primal = dual_residual = math.inf
    for iteration in range(1, max_iter + 1):
        local = []
        for n, agent in enumerate(agents):
            received = np.vstack([
                -consensus[(min(n, m), max(n, m))] * (1 if n < m else -1)
                for m in agent.neighbors
            ])
            u = np.vstack([dual[(n, m)] for m in agent.neighbors])
            local.append(agent.solve(received, u))
        messages = {}
        raw = {}
        for n, agent in enumerate(agents):
            for row, m in enumerate(agent.neighbors):
                raw[(n, m)] = local[n]["trade"][row].copy()
                messages[(n, m)] = raw[(n, m)] + rng.laplace(0.0, b, hours)
        previous = {edge: value.copy() for edge, value in consensus.items()}
        for i, j in undirected:
            consensus[(i, j)] = .5 * (
                messages[(i, j)] + dual[(i, j)]
                - messages[(j, i)] - dual[(j, i)]
            )
            dual[(i, j)] += raw[(i, j)] - consensus[(i, j)]
            dual[(j, i)] += raw[(j, i)] + consensus[(i, j)]
        primal = max(float(np.max(np.abs(raw[(i, j)] + raw[(j, i)])))
                     for i, j in undirected)
        dual_residual = rho * max(float(np.max(np.abs(consensus[edge] - previous[edge])))
                                  for edge in undirected)
        last = local
        if primal <= tol and dual_residual <= tol:
            break
    assert last is not None
    return {
        "scenario": scenario["name"], "iterations": iteration,
        "converged_to_tolerance": primal <= tol and dual_residual <= tol,
        "primal_residual_mw": primal, "dual_residual": dual_residual,
        "net_demand_mw": np.vstack([x["net_demand_mw"] for x in last]),
        "local_grid_import_mw": np.vstack([x["grid_import_mw"] for x in last]),
        "daily_local_operating_cost_usd": sum(x["operating_cost_usd"] for x in last),
        "max_local_balance_error_mw": max(x["max_local_balance_error_mw"] for x in last),
    }


@dataclass(frozen=True)
class Plan:
    line_upgrades: tuple[int, ...]
    sop_steps: tuple[int, ...]


class NetworkSOCP:
    """Single-hour branch-flow SOCP with invested lines and SOP terminals."""

    def __init__(self, case: dict):
        self.case = case
        self.lines = case["lines"]
        self.n = len(self.lines)
        self.children = {bus: [] for bus in range(1, 34)}
        self.to_index = {}
        for k, line in enumerate(self.lines):
            self.children[line["from"]].append(k)
            self.to_index[line["to"]] = k
        self.zbase_ohm = (float(case["base_kv"]) * 1000) ** 2 / (
            float(case["base_mva"]) * 1e6
        )
        self.base_p = np.array([line["p_kw"] / 1000 for line in self.lines])
        self.base_q = np.array([line["q_kvar"] / 1000 for line in self.lines])

    def evaluate_hour(self, plan: Plan, net_demand_mw: np.ndarray,
                      price_usd_per_mwh: float) -> dict | None:
        case = self.case
        base = float(case["base_mva"])
        p_load = self.base_p.copy()
        q_load = self.base_q.copy()
        tan_phi = math.tan(math.acos(float(case["power_factor"])))
        for mg, net in zip(case["microgrids"], net_demand_mw):
            k = self.to_index[mg["bus"]]
            p_load[k] = float(net)
            q_load[k] = float(net) * tan_phi
        p_load /= base
        q_load /= base

        r = np.array([
            line["r_ohm"] / self.zbase_ohm
            * (line["upgrade_impedance_factor"] if plan.line_upgrades[k] else 1.0)
            for k, line in enumerate(self.lines)
        ])
        x = np.array([
            line["x_ohm"] / self.zbase_ohm
            * (line["upgrade_impedance_factor"] if plan.line_upgrades[k] else 1.0)
            for k, line in enumerate(self.lines)
        ])
        limits = np.array([
            line["upgrade_limit_mva"] if plan.line_upgrades[k] else line["base_limit_mva"]
            for k, line in enumerate(self.lines)
        ]) / base

        P = cp.Variable(self.n)
        Q = cp.Variable(self.n)
        ell = cp.Variable(self.n, nonneg=True)
        v = cp.Variable(33)
        sop_p = cp.Variable(3)
        sop_qa = cp.Variable(3)
        sop_qb = cp.Variable(3)
        active = {bus: 0 for bus in range(1, 34)}
        reactive = {bus: 0 for bus in range(1, 34)}
        constraints = [
            v[0] == 1.0,
            v >= float(case["voltage_min_pu"]) ** 2,
            v <= float(case["voltage_max_pu"]) ** 2,
            ell <= limits ** 2,
        ]
        for j, (a, b) in enumerate(case["sop_candidates"]):
            size_pu = case["sop_size_steps_mva"][plan.sop_steps[j]] / base
            active[a] = active[a] + sop_p[j]
            active[b] = active[b] - sop_p[j]
            reactive[a] = reactive[a] + sop_qa[j]
            reactive[b] = reactive[b] + sop_qb[j]
            constraints.extend([
                cp.SOC(size_pu, cp.hstack([sop_p[j], sop_qa[j]])),
                cp.SOC(size_pu, cp.hstack([sop_p[j], sop_qb[j]])),
            ])

        for k, line in enumerate(self.lines):
            fb, tb = line["from"], line["to"]
            outgoing = self.children[tb]
            out_p = cp.sum(P[outgoing]) if outgoing else 0
            out_q = cp.sum(Q[outgoing]) if outgoing else 0
            constraints.extend([
                P[k] - out_p - r[k] * ell[k] == p_load[k] - active[tb],
                Q[k] - out_q - x[k] * ell[k] == q_load[k] - reactive[tb],
                v[tb - 1] == v[fb - 1] - 2 * (r[k] * P[k] + x[k] * Q[k])
                + (r[k] ** 2 + x[k] ** 2) * ell[k],
                cp.SOC(v[fb - 1] + ell[k],
                       cp.hstack([2 * P[k], 2 * Q[k], v[fb - 1] - ell[k]])),
            ])
        root_flow = cp.sum(P[self.children[1]])
        constraints.append(root_flow >= 0)
        objective = cp.Minimize(
            cp.sum(cp.multiply(r, ell))
            + 1e-5 * cp.sum_squares(sop_p)
            + 1e-5 * cp.sum_squares(sop_qa)
            + 1e-5 * cp.sum_squares(sop_qb)
        )
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver="CLARABEL", verbose=False)
        except cp.SolverError:
            return None
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        grid_mw = float(root_flow.value) * base
        max_loading = max(
            math.hypot(float(P.value[k]), float(Q.value[k])) / float(limits[k])
            for k in range(self.n)
        )
        max_gap = max(
            abs(float(v.value[line["from"] - 1] * ell.value[k]
                      - P.value[k] ** 2 - Q.value[k] ** 2))
            for k, line in enumerate(self.lines)
        )
        return {
            "grid_import_mw": grid_mw,
            "grid_purchase_cost_usd": grid_mw * float(price_usd_per_mwh),
            "network_loss_mw": float(np.dot(r, ell.value)) * base,
            "min_voltage_pu": math.sqrt(float(np.min(v.value))),
            "max_line_loading_fraction": max_loading,
            "max_absolute_soc_gap_pu": max_gap,
        }


def capital_cost(case: dict, plan: Plan) -> tuple[float, float, float]:
    line_capital = sum(
        float(case["line_upgrade_cost_usd_per_km"]) * float(line["length_km"])
        for bit, line in zip(plan.line_upgrades, case["lines"]) if bit
    )
    sop_capital = sum(
        float(case["sop_fixed_cost_usd"])
        + float(case["sop_variable_cost_usd_per_kva"])
        * 1000 * float(case["sop_size_steps_mva"][step])
        for step in plan.sop_steps if step
    )
    annualized = float(case["capital_recovery_factor"]) * (line_capital + sop_capital)
    return line_capital, sop_capital, annualized


def evaluate_plan(case: dict, plan: Plan, scenarios: list[dict],
                  admm: list[dict], hours: int, network: NetworkSOCP) -> dict:
    line_capital, sop_capital, annual_capital = capital_cost(case, plan)
    annual_grid = 0.0
    annual_local = 0.0
    max_loading = 0.0
    min_voltage = math.inf
    max_gap = 0.0
    failed = []
    for scenario, operation in zip(scenarios, admm):
        days = int(scenario["days"])
        annual_local += days * float(operation["daily_local_operating_cost_usd"])
        for t in range(hours):
            result = network.evaluate_hour(
                plan, operation["net_demand_mw"][:, t],
                case["price_usd_per_mwh"][t]
            )
            if result is None:
                failed.append(f"{scenario['name']}:hour{t + 1}")
                continue
            annual_grid += days * result["grid_purchase_cost_usd"]
            max_loading = max(max_loading, result["max_line_loading_fraction"])
            min_voltage = min(min_voltage, result["min_voltage_pu"])
            max_gap = max(max_gap, result["max_absolute_soc_gap_pu"])
    penalty = float(case["infeasibility_penalty_usd"]) * len(failed)
    # This follows F_DSO = annualized investment + upstream-grid purchase cost.
    objective = annual_capital + annual_grid + penalty
    return {
        "objective_usd_per_year": objective,
        "annualized_investment_usd": annual_capital,
        "line_capital_usd": line_capital,
        "sop_capital_usd": sop_capital,
        "annual_grid_purchase_usd": annual_grid,
        "annual_local_operating_usd": annual_local,
        "infeasibility_penalty_usd": penalty,
        "failed_network_hours": failed,
        "max_line_loading_fraction": max_loading if min_voltage < math.inf else None,
        "min_voltage_pu": min_voltage if min_voltage < math.inf else None,
        "max_absolute_soc_gap_pu": max_gap if min_voltage < math.inf else None,
        "line_upgrades": [list((line["from"], line["to"]))
                          for bit, line in zip(plan.line_upgrades, case["lines"]) if bit],
        "sop_sizes_mva": [
            {"corridor": pair, "size_mva": case["sop_size_steps_mva"][step]}
            for pair, step in zip(case["sop_candidates"], plan.sop_steps) if step
        ],
    }


def random_plan(case: dict, rng: np.random.Generator) -> Plan:
    lines = tuple(int(x) for x in (rng.random(len(case["lines"])) < .12))
    sop = tuple(int(rng.integers(0, len(case["sop_size_steps_mva"])))
                if rng.random() < .45 else 0 for _ in case["sop_candidates"])
    return Plan(lines, sop)


def genetic_search(case: dict, scenarios: list[dict], admm: list[dict],
                   hours: int, seed: int, population: int, generations: int) -> tuple[Plan, dict, list]:
    rng = np.random.default_rng(seed)
    network = NetworkSOCP(case)
    n_lines = len(case["lines"])
    zero = Plan((0,) * n_lines, (0,) * 3)
    all_lines = Plan((1,) * n_lines, (0,) * 3)
    individuals = [zero, all_lines] + [random_plan(case, rng)
                                      for _ in range(max(0, population - 2))]
    cache: dict[Plan, dict] = {}
    history = []

    def score(plan: Plan) -> dict:
        if plan not in cache:
            cache[plan] = evaluate_plan(case, plan, scenarios, admm, hours, network)
        return cache[plan]

    def tournament() -> Plan:
        choices = [individuals[int(rng.integers(len(individuals)))] for _ in range(3)]
        return min(choices, key=lambda p: score(p)["objective_usd_per_year"])

    for gen in range(generations):
        ranked = sorted(individuals, key=lambda p: score(p)["objective_usd_per_year"])
        best = ranked[0]
        history.append({"generation": gen + 1,
                        "best_usd_per_year": score(best)["objective_usd_per_year"],
                        "evaluated_plans": len(cache)})
        next_generation = ranked[:max(1, min(2, population // 4))]  # elitism
        while len(next_generation) < population:
            a, b = tournament(), tournament()
            if rng.random() < float(case["ga_crossover_probability"]):
                line_bits = [a.line_upgrades[k] if rng.random() < .5 else b.line_upgrades[k]
                             for k in range(n_lines)]
                sop_steps = [a.sop_steps[k] if rng.random() < .5 else b.sop_steps[k]
                             for k in range(3)]
            else:
                line_bits, sop_steps = list(a.line_upgrades), list(a.sop_steps)
            mutation = float(case["ga_mutation_probability"])
            for k in range(n_lines):
                if rng.random() < mutation:
                    line_bits[k] = 1 - line_bits[k]
            for k in range(3):
                if rng.random() < mutation:
                    sop_steps[k] = int(rng.integers(len(case["sop_size_steps_mva"])))
            next_generation.append(Plan(tuple(line_bits), tuple(sop_steps)))
        individuals = next_generation
    final_best = min(individuals, key=lambda p: score(p)["objective_usd_per_year"])
    return final_best, score(final_best), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use explicitly synthetic demonstration inputs")
    source.add_argument("--input", type=Path, help="JSON with actual network, scenarios and costs")
    parser.add_argument("--quick", action="store_true",
                        help="Two-hour, one-scenario pipeline smoke test; not an annual result")
    parser.add_argument("--binary-ess", action="store_true",
                        help="Enforce exact charge/discharge binaries; requires Gurobi")
    parser.add_argument("--noise-scale-mw", type=float, default=None,
                        help="Override fixed Laplace scale, e.g. 0 for a matched no-noise run")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("planning_result.json"))
    parser.add_argument("--write-demo-input", type=Path,
                        help="Also write the synthetic JSON so inputs can be inspected")
    args = parser.parse_args()

    if args.binary_ess and "GUROBI" not in cp.installed_solvers():
        parser.error("--binary-ess requires an installed/licensed Gurobi CVXPY solver")
    if args.demo:
        case = make_demo_case()
        if args.write_demo_input:
            args.write_demo_input.parent.mkdir(parents=True, exist_ok=True)
            args.write_demo_input.write_text(json.dumps(case, indent=2), encoding="utf-8")
    else:
        case = json.loads(args.input.read_text(encoding="utf-8"))
        case.setdefault("data_provenance", "EXTERNAL_INPUT_NOT_INDEPENDENTLY_VERIFIED")
    validate_case(case)
    if args.noise_scale_mw is not None:
        if args.noise_scale_mw < 0:
            parser.error("Noise scale must be nonnegative")
        case["noise_scale_mw"] = args.noise_scale_mw

    scenarios = case["scenarios"][:1] if args.quick else case["scenarios"]
    hours = 2 if args.quick else 24
    population = 4 if args.quick else int(case["ga_population"])
    generations = 2 if args.quick else int(case["ga_generations"])
    max_iter = 3 if args.quick else int(case["admm_max_iterations"])
    source_hash = hashlib.sha256(
        json.dumps(case, sort_keys=True).encode("utf-8")
    ).hexdigest()

    print(f"Input: {case['data_provenance']}; hours={hours}; scenarios={len(scenarios)}")
    print(f"ESS mode: {'binary (Gurobi)' if args.binary_ess else 'continuous relaxation'}")
    print(f"Fixed Laplace scale: {case['noise_scale_mw']} MW; ADMM iterations <= {max_iter}")
    admm = []
    for k, scenario in enumerate(scenarios):
        output = run_admm(case, scenario, hours, args.seed + 1000 * k,
                          args.binary_ess, max_iter)
        print(f"ADMM {scenario['name']}: iterations={output['iterations']}, "
              f"primal={output['primal_residual_mw']:.3g} MW, "
              f"converged={output['converged_to_tolerance']}")
        admm.append(output)

    best_plan, best, history = genetic_search(
        case, scenarios, admm, hours, args.seed + 99, population, generations
    )
    report: dict[str, Any] = {
        "implementation": "independent_reimplementation_not_original_manuscript_code",
        "data_provenance": case["data_provenance"],
        "input_sha256": source_hash,
        "seed": args.seed,
        "noise_scale_mw": case["noise_scale_mw"],
        "privacy_budget_certified": False,
        "ess_mode": "binary_gurobi" if args.binary_ess else "continuous_relaxation",
        "hours_per_scenario": hours,
        "scenario_names": [s["name"] for s in scenarios],
        "full_annual_horizon": hours == 24 and sum(s["days"] for s in scenarios) == 365,
        "ga_population": population,
        "ga_generations": generations,
        "admm": [{key: value for key, value in result.items()
                  if key not in ("net_demand_mw", "local_grid_import_mw")}
                 for result in admm],
        "ga_history": history,
        "best_plan": {
            "line_upgrade_bits": list(best_plan.line_upgrades),
            "sop_size_step_indices": list(best_plan.sop_steps),
        },
        "best_evaluation": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best evaluated objective: {best['objective_usd_per_year']:.2f} USD "
          f"(partial horizon={not report['full_annual_horizon']})")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
