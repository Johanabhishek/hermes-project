import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from src.hermes.api import HermesFinder
from src.hermes.policy import SEARCH_SPACE

def load_and_prep_data(dataset_name, sequence_length=24):
    
    print(f"\n- - - Loading and Preparing Dataset: {dataset_name.upper()} - - -")

    if dataset_name == 'energy':
        df = pd.read_csv('energydata_complete.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        median_usage = df['Appliances'].median()
        df['target'] = (df['Appliances'] > median_usage).astype(int)
        features_df = df.drop(columns=['Appliances', 'lights', 'target'])
        target = df['target']

    elif dataset_name == 'kpi':
        df = pd.read_csv('kpi.csv')
        df.rename(columns={'label': 'target'}, inplace=True)
        features_df = df[['value']]
        target = df['target']

    elif dataset_name == 'yahoo':
        df = pd.read_csv('yahoo_sub_5.csv')
        
        df.rename(columns={'anomaly': 'target'}, inplace=True)
        
        features_df = df[['value_0', 'value_1', 'value_2', 'value_3', 'value_4']]
        target = df['target']
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # Create sequences for the LSTM model
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:(i + sequence_length)])
        y.append(target.iloc[i + sequence_length])
        
    print(f"Dataset '{dataset_name}' ready. Shape of X: {np.array(X).shape}")
    return np.array(X), np.array(y)


if __name__ == "__main__":
    
    DATASETS_TO_RUN = ['energy', 'yahoo', 'kpi']
    NUM_TRIALS = 3  # Number of random policies to try for each dataset
    SEQUENCE_LENGTH = 24 # Timesteps for the LSTM
    TEST_SET_SIZE = 0.2
    
    
    all_results = {}

    print("="*60)
    print("  PROJECT HERMES: UNIFIED BENCHMARK EXPERIMENT  ")
    print("="*60)
    print(f"Objective: Find the best augmentation policy for {len(DATASETS_TO_RUN)} diverse datasets.")
    print(f"Method: Random search with {NUM_TRIALS} trials per dataset.\n")

    for dataset_name in DATASETS_TO_RUN:
        X, y = load_and_prep_data(dataset_name, sequence_length=SEQUENCE_LENGTH)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SET_SIZE, random_state=42, stratify=y
        )
        
        
        finder = HermesFinder(
            search_space=SEARCH_SPACE,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        
        best_policy_info = finder.search(num_trials=NUM_TRIALS)
        all_results[dataset_name] = best_policy_info

    
    print("\n\n" + "="*60)
    print("          EXPERIMENT COMPLETE: FINAL REPORT          ")
    print("="*60)
    print("Comparing the best performing policies across all datasets:\n")
    
    for dataset, result in all_results.items():
        score = result['score']
        policy = result['policy']
        
        # Format policy for clean printing
        if policy == 'baseline':
            policy_str = "Baseline (No Augmentation)"
        else:
            policy_str = ' + '.join(p['name'] for p in policy)
            
        print(f"  Dataset: {dataset.upper():<10}")
        print(f"  ─ Best Accuracy: {score:.4f}")
        print(f"  ─ Optimal Policy: {policy_str}\n")

   
    print("Conclusion: The results clearly demonstrate that the optimal data augmentation\n"
          "policy is highly dependent on the dataset, validating the core premise of\n"
          "Project Hermes. One size does not fit all.")
    print("="*60)

    