import unittest

from demo_case import make_demo_case
from planning import NetworkSOCP, Plan, run_admm


class PipelineTest(unittest.TestCase):
    def test_no_noise_consensus_and_network_feasibility(self):
        case = make_demo_case()
        case["noise_scale_mw"] = 0.0
        admm = run_admm(case, case["scenarios"][0], 2, 2026, False, 100)
        self.assertTrue(admm["converged_to_tolerance"])
        self.assertLess(admm["max_local_balance_error_mw"], 1e-6)
        plan = Plan((1,) * 32, (0, 0, 0))
        flow = NetworkSOCP(case).evaluate_hour(
            plan, admm["net_demand_mw"][:, 0], case["price_usd_per_mwh"][0]
        )
        self.assertIsNotNone(flow)
        self.assertGreaterEqual(flow["grid_purchase_cost_usd"], 0)
        self.assertGreaterEqual(flow["min_voltage_pu"], case["voltage_min_pu"] - 1e-5)
        self.assertLessEqual(flow["max_line_loading_fraction"], 1 + 1e-5)


if __name__ == "__main__":
    unittest.main()
