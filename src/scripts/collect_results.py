import json
import pandas as pd
import glob
import os
from sklearn.metrics import roc_auc_score

def compute_auc(csv_path, label):
    try:
        df = pd.read_csv(csv_path)
        # Aggregate by patient_id and laterality (taking max or mean, same as training script)
        if 'patient_id' in df.columns and 'laterality' in df.columns:
            df_agg = df.groupby(['patient_id', 'laterality']).mean()
            return roc_auc_score(df_agg[label], df_agg['prediction'])
        else:
            return roc_auc_score(df[label], df['prediction'])
    except Exception as e:
        return None

def main():
    base_dir = "/data/nas07_new/PersonalData/robit/Mammo-CLIP"
    
    print("Collecting Results...")
    print("="*60)
    
    # 1. Zero Shot
    print("Zero-Shot Results:")
    zs_dir = os.path.join(base_dir, "outputs", "zero_shot")
    for model in ["b2", "b5"]:
        json_path = os.path.join(zs_dir, model, "results-vindr.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            # data has shape: { "ckpt_path": { "mass": auc, "suspicious_calcification": auc } }
            for ckpt, metrics in data.items():
                print(f"Model: {model}")
                for label, auc in metrics.items():
                    print(f"  {label}: {auc:.4f}")
        else:
            print(f"Model: {model} - No zero-shot results found.")
            
    print("-" * 60)
    print("Linear Probe Results:")
    lp_dir = os.path.join(base_dir, "outputs", "linear_probe")
    for model in ["b2", "b5"]:
        for fraction in ["0.1", "0.5", "1.0"]:
            for label in ["Mass", "Suspicious_Calcification", "density"]:
                # The train script creates a subfolder zz/{model_type}/{arch}/{root} inside output_path
                # We need to glob for the output csv
                search_pattern = os.path.join(lp_dir, model, f"{label}_{fraction}", "**", "*_outputs.csv")
                csv_files = glob.glob(search_pattern, recursive=True)
                if csv_files:
                    auc = compute_auc(csv_files[0], label)
                    print(f"Model: {model} | Fraction: {fraction} | Label: {label} -> AUC: {auc:.4f}" if auc else "Failed")
                
    print("-" * 60)
    print("Fine-Tune Results:")
    ft_dir = os.path.join(base_dir, "outputs", "fine_tune")
    for model in ["b2", "b5"]:
        for label in ["Mass", "Suspicious_Calcification", "density"]:
            search_pattern = os.path.join(ft_dir, model, label, "**", "*_outputs.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            if csv_files:
                auc = compute_auc(csv_files[0], label)
                print(f"Model: {model} | Label: {label} -> AUC: {auc:.4f}" if auc else "Failed")

if __name__ == "__main__":
    main()
