# src/hermes/api.py

import json
from . import policy
from . import evaluation
import os
import sys

class HermesFinder:
    def __init__(self, search_space, X_train, y_train, X_test, y_test):
        if not search_space:
            raise ValueError("A search_space dictionary must be provided.")
        self.search_space = search_space
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = []
        self.best_result = None

    def search(self, num_trials=10, include_baseline=True):
        print(f"--- Beginning Hermes Random Search for {num_trials} Trials ---")
        self.results = []
        if include_baseline:
            print("\nCalculating baseline performance...")
            baseline_score = evaluation.evaluate_model(
                self.X_train, self.y_train, self.X_test, self.y_test
            )
            print(f"--> Baseline Score (Accuracy): {baseline_score:.4f}")
            self.results.append({'policy': 'baseline', 'score': baseline_score})
        for i in range(num_trials):
            print(f"\n--- Trial {i+1}/{num_trials} ---")
            random_policy = policy.generate_random_policy(self.search_space)
            score = evaluation.evaluate_policy(
                random_policy, self.X_train, self.y_train, self.X_test, self.y_test
            )
            self.results.append({'policy': random_policy, 'score': score})
        self.best_result = max(self.results, key=lambda x: x['score'])
        print("\n\n--- SEARCH COMPLETE ---")
        print(f"Best policy found with score: {self.best_result['score']:.4f}")
        return self.best_result

    def print_results(self):
        if not self.results:
            print("No search has been run yet.")
            return
        print("\n--- Full Search Results ---")
        for res in sorted(self.results, key=lambda x: x['score'], reverse=True):
            policy_name = res['policy']
            if isinstance(policy_name, list):
                policy_name = '+'.join([p['name'] for p in policy_name])
            print(f"  - Policy: {policy_name:<30} | Score: {res['score']:.4f}")

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from hermes import policy
    print("--- Testing the HermesFinder API ---")
    dummy_X_train = np.random.rand(50, 12, 4)
    dummy_y_train = np.random.randint(0, 2, 50)
    dummy_X_test = np.random.rand(10, 12, 4)
    dummy_y_test = np.random.randint(0, 2, 10)
    finder = HermesFinder(
        search_space=policy.SEARCH_SPACE,
        X_train=dummy_X_train,
        y_train=dummy_y_train,
        X_test=dummy_X_test,
        y_test=dummy_y_test
    )
    best_policy_info = finder.search(num_trials=3)
    finder.print_results()
    print("\n--- Best Performing Policy ---")
    print(json.dumps(best_policy_info, indent=2))
