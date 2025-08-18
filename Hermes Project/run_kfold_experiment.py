import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from src.hermes import evaluation
from scipy.stats import ttest_rel

def load_dataset(name, sequence_length=12):
    print(f"\n--- Loading and Preparing Dataset: {name.upper()} ---")
    df = pd.read_csv('energydata_complete.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    median_usage = df['Appliances'].median()
    df['target'] = (df['Appliances'] > median_usage).astype(int)
    features = df.drop(columns=['Appliances', 'lights', 'target'])
    target = df['target']
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    def create_sequences(feats, targs, seq_len):
        X, y = [], []
        for i in range(len(feats) - seq_len):
            X.append(feats[i:(i + seq_len)])
            y.append(targs.iloc[i + seq_len])
        return np.array(X), np.array(y)
    X, y = create_sequences(features_scaled, target, sequence_length)
    print("Dataset ready.")
    return X, y

if __name__ == "__main__":
    DATASET_TO_RUN = 'energy'
    NUM_SPLITS = 3 # Use 5 for a robust test
    X, y = load_dataset(DATASET_TO_RUN)

    # Define challenger policies beforehand 
    POLICIES_TO_TEST = {
        "baseline": [], # The baseline is just an empty policy list
        "jitter_only": [{"name": "jittering", "params": {"sigma": 0.05}}],
        "scaling_only": [{"name": "scaling", "params": {"sigma": 0.1, "num_knots": 4}}],
        "jitter_and_scaling": [
            {"name": "jittering", "params": {"sigma": 0.05}},
            {"name": "scaling", "params": {"sigma": 0.1, "num_knots": 4}}
        ],
        "time_warp_only": [{"name": "time_warping", "params": {"sigma": 0.2, "num_knots": 4}}]
    }

    # This dictionary will hold the list of scores for each policy
    results = {name: [] for name in POLICIES_TO_TEST.keys()}

    kfold = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)
    fold_num = 1

    print(f"\n--- Beginning {NUM_SPLITS}-Fold Statistical Showdown for {DATASET_TO_RUN.upper()} ---")

    # Main loop now iterates through folds 
    for train_index, test_index in kfold.split(X):
        print(f"\n===== FOLD {fold_num}/{NUM_SPLITS} =====")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner loop evaluates each predefined policy 
        for policy_name, policy in POLICIES_TO_TEST.items():
            print(f"-> Evaluating Policy: {policy_name}")
            if policy_name == "baseline":
                accuracy = evaluation.evaluate_model(X_train, y_train, X_test, y_test)
            else:
                accuracy = evaluation.evaluate_policy(policy, X_train, y_train, X_test, y_test)
            
            results[policy_name].append(accuracy)
        
        fold_num += 1

    # T-Test logic is now more direct 
    print("\n\n" + "="*55)
    print("--- STATISTICAL SIGNIFICANCE ANALYSIS (PAIRED T-TEST) ---")
    print("Comparing each policy against the baseline:")

    baseline_scores = results['baseline']
    
    print(f"\nBaseline Mean Accuracy: {np.mean(baseline_scores):.4f} (+/- {np.std(baseline_scores):.4f})")
    
    for policy_name, scores in results.items():
        if policy_name == "baseline":
            continue # Don't compare the baseline to itself

        t_stat, p_value = ttest_rel(baseline_scores, scores)
        
        print(f"\n  Policy: {policy_name}")
        print(f"  ├─ Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        print(f"  ├─ P-value vs Baseline: {p_value:.4f}")
        
        if p_value < 0.05:
            if np.mean(scores) > np.mean(baseline_scores):
                print("  └─ Result:  Statistically significant IMPROVEMENT.")
            else:
                print("  └─ Result:  Statistically significant DEGRADATION.")
        else:
            print("  └─ Result:  No significant difference from baseline.")
    print("="*55)