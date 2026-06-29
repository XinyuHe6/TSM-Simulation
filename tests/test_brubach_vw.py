import math
import random
import unittest

from matching_algorithms import (
    canonical_matching_name,
    prepare_matching_state,
    run_matching,
)
from matching_algorithms import brubach_vw


class BrubachVWTests(unittest.TestCase):
    def test_registry_alias(self):
        self.assertEqual(canonical_matching_name("vw"), "brubach_vw")
        self.assertEqual(canonical_matching_name("brubach"), "brubach_vw")

    def test_lp_enforces_single_edge_and_pair_caps(self):
        edge_list, solution, objective = brubach_vw._solve_benchmark_lp(
            A_size=1,
            expanded_neighbors=[[0], [0]],
        )
        self.assertEqual(edge_list, [(0, 0), (1, 0)])
        self.assertLessEqual(max(solution), brubach_vw.EDGE_CAP + 1e-8)
        self.assertAlmostEqual(sum(solution), brubach_vw.PAIR_CAP, places=7)
        self.assertAlmostEqual(objective, brubach_vw.PAIR_CAP, places=7)

    def test_dependent_rounding_preserves_vertex_degrees(self):
        edge_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        f = [0.20, 0.30, 0.30, 0.20]

        for seed in range(100):
            rounded = brubach_vw._dependent_round(
                edge_list,
                f,
                random.Random(seed),
            )
            self.assertTrue(all(value in (0, 1, 2) for value in rounded))

            for copy in (0, 1):
                indices = [i for i, edge in enumerate(edge_list) if edge[0] == copy]
                before = sum(3.0 * f[i] for i in indices)
                after = sum(rounded[i] for i in indices)
                self.assertIn(after, (math.floor(before), math.ceil(before)))

            for advertiser in (0, 1):
                indices = [i for i, edge in enumerate(edge_list) if edge[1] == advertiser]
                before = sum(3.0 * f[i] for i in indices)
                after = sum(rounded[i] for i in indices)
                self.assertIn(after, (math.floor(before), math.ceil(before)))

    def test_cycle_breaking_removes_c2_and_preserves_degrees(self):
        c2 = {
            (0, 0): 2,
            (1, 0): 1,
            (0, 1): 1,
            (1, 1): 1,
        }
        before = brubach_vw._degrees(c2)
        after = brubach_vw._break_short_cycles(c2)

        self.assertEqual(brubach_vw._degrees(after), before)
        self.assertIsNone(brubach_vw._find_four_cycle(after, "c2"))
        self.assertIsNone(brubach_vw._find_four_cycle(after, "c3"))

    def test_figure_four_balancing_rules(self):
        x1_case = {
            (0, 0): 2,
            (0, 1): 1,
            (1, 0): 1,
            (2, 1): 2,
        }
        balanced = brubach_vw._balance_h(x1_case, expanded_count=3)
        self.assertAlmostEqual(balanced[(0, 0)], 1.0 - brubach_vw.X1)
        self.assertAlmostEqual(balanced[(0, 1)], brubach_vw.X1)

        three_edge_case = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 1,
            (1, 1): 1,
            (2, 2): 2,
        }
        balanced = brubach_vw._balance_h(three_edge_case, expanded_count=3)
        self.assertAlmostEqual(balanced[(0, 0)], 0.15)
        self.assertAlmostEqual(balanced[(0, 1)], 0.20)
        self.assertAlmostEqual(balanced[(0, 2)], 0.65)

    def test_integral_rate_expansion_and_online_run(self):
        neighbors = [[0, 1], [1, 2]]
        state = prepare_matching_state(
            "brubach_vw",
            A_size=3,
            I_size=2,
            neighbors=neighbors,
            e=[2, 1],
            T=3,
            seed=11,
        )
        self.assertEqual([len(copies) for copies in state["copies_of_type"]], [2, 1])

        random.seed(12)
        result = run_matching(
            "brubach_vw",
            A_size=3,
            I_size=2,
            neighbors=neighbors,
            arrivals=[0, 1, 0],
            state=state,
        )
        self.assertGreaterEqual(result.alg, 0)
        self.assertLessEqual(result.alg, 3)


if __name__ == "__main__":
    unittest.main()
