import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz
from sklearn.ensemble import IsolationForest
import joblib

def main():
    print("Starting ML Pipeline Training for AutoBill Verify...", flush=True)
    start_time = time.time()
    
    # 1. Load Datasets
    cghs_path = Path("data/cghs_rate_list.csv")
    billing_path = Path("data/billing_dataset.csv")
    
    if not cghs_path.exists() or not billing_path.exists():
        raise FileNotFoundError("Dataset files missing. Please run generate_data.py first.")
        
    print(f"Loading '{cghs_path}' and '{billing_path}'...", flush=True)
    cghs_df = pd.read_csv(cghs_path)
    billing_df = pd.read_csv(billing_path)
    
    print(f"Loaded {len(cghs_df)} CGHS official rates and {len(billing_df)} billing line items.", flush=True)

    # 2. RapidFuzz Token-Sort Matching (Threshold > 80%)
    cghs_names = cghs_df['item_name'].tolist()
    unique_billed = billing_df['billed_item_name'].unique()
    print(f"Performing RapidFuzz token-sort matching for {len(unique_billed)} unique billed items...", flush=True)
    
    match_cache = {}
    matched_count = 0
    
    for billed_name in unique_billed:
        # Match using token_sort_ratio scorer with score_cutoff for speed
        best_match = process.extractOne(
            billed_name, 
            cghs_names, 
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80.0
        )
        
        if best_match:  # Satisfies threshold > 80%
            matched_idx = best_match[2]
            cghs_row = cghs_df.iloc[matched_idx]
            match_cache[billed_name] = {
                'matched_cghs_name': cghs_row['item_name'],
                'cghs_item_code': cghs_row['item_code'],
                'cghs_rate': float(cghs_row['cghs_rate']),
                'std_dev': float(cghs_row['std_dev']),
                'match_score': best_match[1]
            }
            matched_count += 1
        else:
            match_cache[billed_name] = None
            
    print(f"Matching complete: {matched_count}/{len(unique_billed)} unique items matched with token-sort ratio > 80%.", flush=True)

    # Map match results back to the full billing dataset
    cghs_rates = []
    std_devs = []
    match_scores = []
    valid_indices = []

    for idx, row in billing_df.iterrows():
        info = match_cache.get(row['billed_item_name'])
        if info is not None:
            cghs_rates.append(info['cghs_rate'])
            std_devs.append(info['std_dev'])
            match_scores.append(info['match_score'])
            valid_indices.append(idx)

    processed_df = billing_df.iloc[valid_indices].copy()
    processed_df['cghs_rate'] = cghs_rates
    processed_df['std_dev'] = std_devs
    processed_df['match_score'] = match_scores

    # 3. Compute Feature Vectors
    # Price Ratio = billed_price / cghs_rate
    # Z-Score = (billed_price - cghs_rate) / std_dev
    processed_df['price_ratio'] = processed_df['billed_price'] / processed_df['cghs_rate']
    processed_df['z_score'] = (processed_df['billed_price'] - processed_df['cghs_rate']) / processed_df['std_dev']

    X = processed_df[['price_ratio', 'z_score']].values
    print(f"Computed feature matrix X with shape {X.shape}.", flush=True)
    print(f"Sample Features [Price Ratio, Z-Score]:\n{X[:5]}", flush=True)

    # 4. Train IsolationForest Model
    print("Training IsolationForest(n_estimators=100, contamination=0.1, random_state=42)...", flush=True)
    clf = IsolationForest(
        n_estimators=100, 
        contamination=0.1, 
        random_state=42
    )
    clf.fit(X)
    
    # Evaluate predictions summary
    predictions = clf.predict(X)  # -1 for anomaly, 1 for normal
    anomaly_count = (predictions == -1).sum()
    normal_count = (predictions == 1).sum()
    print(f"Model Predictions: {normal_count} Normal (1), {anomaly_count} Anomaly (-1).", flush=True)

    # 5. Serialize and Save Model
    model_dir = Path("model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "isolation_forest.joblib"
    
    joblib.dump(clf, model_path)
    print(f"Successfully saved trained model to '{model_path}'.", flush=True)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"Pipeline completed in {elapsed} seconds.", flush=True)

if __name__ == "__main__":
    main()
