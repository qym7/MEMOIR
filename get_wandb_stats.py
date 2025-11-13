import pandas as pd
import wandb
api = wandb.Api()
# --- Parameters ---
# method = "DEV"
# suffix = "table_result_sweep_loc055_noactloss_norouting"

method = "MEMOIR"
# suffix = "_EXP1_SWEEPNEW_last_mean"
# suffix = "_SWEEPNEW_last_last"
suffix = "_SWEEPNEW_mean_decentered"

# suffix = "table_result_sweep_loc06_noactloss_norouting"
datasets = ["ZsRE"]
ds_values = [1, 10, 100, 1000] # , 1000]
ds_values = [1000] # , 1000]
# models = ["Meta-Llama-3-8B-instruct"]
# models = ["Mistral-7B-v0.1"]
models = ["Meta-Llama-3-8B-instruct", "Mistral-7B-v0.1"]
for model in models:
    for dataset in datasets:
        for ds in ds_values:
            # --- Compose project name ---
            project_name = f"sibylria/lifelong_edit_{dataset}_{method}_t{ds}_{model}{suffix}"
            print(f"\n:bar_chart: Statistics for project: {project_name}")
            try:
                runs = api.runs(project_name)
            except wandb.CommError as e:
                print(f":x: Error fetching runs: {e}")
                continue
            summary_list, config_list, name_list = [], [], []
            for run in runs:
                summary_list.append(run.summary._json_dict)
                config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
                name_list.append(run.name)
            if not summary_list:
                print(":warning:  No runs found.")
                continue
            df = pd.DataFrame(summary_list)
            df["name"] = name_list
            if dataset == "ZsRE":
                metrics = [
                    "post/mean_rewrite_acc",
                    "post/mean_rephrase_acc",
                    "post/mean_neighborhood_acc",
                ]
            elif dataset == "Hallucination":
                metrics = [
                    "post/mean_rewrite_ppl",
                    "post/mean_neighborhood_acc",
                ]
            else:
                print(f":warning:  Unknown dataset: {dataset}")
                continue
            print(f"Total runs: {len(df)}")
            for metric in metrics:
                if metric in df.columns:
                    values = df[metric].dropna()
                    # import numpy as np
                    # values = np.clip(values, 0, 100)
                    if len(values) > 0:
                        print(f"{metric} = {values.mean():.6f} (n={len(values)})")
                    else:
                        print(f"{metric} = NaN (n=0)")
                else:
                    print(f"{metric} = MISSING")