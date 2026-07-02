import random
import unittest

from matching_algorithms import (
    MATCHING_SPECS,
    prepare_matching_states,
    run_matching,
)


class ModularAlgorithmIntegrationTests(unittest.TestCase):
    def test_every_registered_algorithm_runs_from_the_package(self):
        algorithms = list(MATCHING_SPECS)
        neighbors = [[0, 1, 2], [0, 1], [1, 2]]
        arrivals = [0, 1, 2]
        e = [1, 1, 1]

        states = prepare_matching_states(
            algorithms=algorithms,
            A_size=3,
            I_size=3,
            neighbors=neighbors,
            e=e,
            T=3,
            seed=7,
            mc_trials=3,
            corr_lp_constraint_mode="pair_approx",
        )

        for index, algorithm in enumerate(algorithms):
            with self.subTest(algorithm=algorithm):
                random.seed(100 + index)
                result = run_matching(
                    algorithm,
                    A_size=3,
                    I_size=3,
                    neighbors=neighbors,
                    arrivals=arrivals,
                    state=states[algorithm],
                )
                self.assertGreaterEqual(result.alg, 0)
                self.assertLessEqual(result.alg, 3)

    def test_every_algorithm_handles_an_edgeless_graph(self):
        algorithms = list(MATCHING_SPECS)
        neighbors = [[], []]
        e = [1, 1]
        states = prepare_matching_states(
            algorithms=algorithms,
            A_size=2,
            I_size=2,
            neighbors=neighbors,
            e=e,
            T=2,
            seed=11,
            mc_trials=2,
            corr_lp_constraint_mode="pair_approx",
        )

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                result = run_matching(
                    algorithm,
                    A_size=2,
                    I_size=2,
                    neighbors=neighbors,
                    arrivals=[0, 1],
                    state=states[algorithm],
                )
                self.assertEqual(result.alg, 0)


if __name__ == "__main__":
    unittest.main()
