# src/hermes/policy.py

import random
import numpy as np

SEARCH_SPACE = {
    "jittering": {
        "sigma": {"type": "continuous", "range": [0.01, 0.5]}
    },
    "time_warping": {
        "sigma": {"type": "continuous", "range": [0.1, 0.5]},
        "num_knots": {"type": "discrete", "range": [2, 8]}
    },
    "scaling": {
        "sigma": {"type": "continuous", "range": [0.1, 0.3]},
        "num_knots": {"type": "discrete", "range": [3, 8]}
    },
    "permutation": {
        "num_segments": {"type": "discrete", "range": [2, 8]}
    },
}

def generate_random_policy(search_space):
    policy = []
    num_techniques = random.randint(1, 3)
    available_techniques = list(search_space.keys())
    chosen_techniques = random.sample(available_techniques, num_techniques)
    
    for tech_name in chosen_techniques:
        technique_spec = search_space[tech_name]
        sampled_params = {}
        for param_name, param_spec in technique_spec.items():
            if param_spec["type"] == "continuous":
                low, high = param_spec["range"]
                sampled_params[param_name] = random.uniform(low, high)
            elif param_spec["type"] == "discrete":
                low, high = param_spec["range"]
                sampled_params[param_name] = random.randint(low, high)
        policy.append({
            "name": tech_name,
            "params": sampled_params
        })
    return policy
