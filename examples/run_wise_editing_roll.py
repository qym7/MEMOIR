import os.path
import sys
import json
import argparse

import wandb

from omegaconf import OmegaConf, open_dict, DictConfig

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    MEMOIRHyperParams,
    DeferHyperParams,
    AlphaEditHyperParams,
    BaseEditor,
    summary_metrics,
)

import subprocess


def setup_wandb(cfg, name, wandb_mode, data_type, method_name, ds_size, project_suffix=""):
    model_name = cfg.model_name.split('/')[-1]
    # Retrieve the current branch and commit hash
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    cfg_to_log = cfg.__dict__
    cfg_to_log['commit'] = commit
    cfg_to_log['branch'] = branch
    project_name = f'lifelong_edit_{data_type}_{method_name}_t{ds_size}_{model_name}'
    if project_suffix != "":
        project_name = project_name + project_suffix
    kwargs = {
        'name': f'{model_name}_{name}',
        'project': project_name,
        'config': cfg_to_log,
        'settings': wandb.Settings(_disable_stats=True),
        'reinit': True,
        'mode': wandb_mode}
    print('hyparams', cfg.__dict__)
    wandb_run = wandb.init(**kwargs)
    wandb.save('*.txt')

    return wandb_run

def resume_wandb(wandb_run):
    # Compute the mean of all logged values at the end
    import pdb; pdb.set_trace()

    # Fetch the current run using the stored details
    project_name = wandb.run.project  # Get project name dynamically
    run_id = wandb.run.id  # Get run ID dynamically
    entity = wandb.run.entity  # Get entity (username/org)

    # Ensure the run is finished before querying
    wandb.finish()

    # Use W&B API to retrieve data
    api = wandb.Api()

    # Load run history
    run = api.run(f"{entity}/{project_name}/{run_id}")

    # Convert to a pandas DataFrame
    history = run.history(pandas=True)

    # Print the data
    print(history)

    # Save the log to a CSV file if needed
    history.to_csv("wandb_logs.csv", index=False)

    # Identify all "pre/" and "post/" metrics
    metrics = [col for col in history.columns if col.startswith(("pre/", "post/"))]

    for metric in metrics:
        mean_value = history[metric].dropna().mean()  # Ignore missing values
        wandb.summary[f"{metric}_mean"] = mean_value  # Store in summary

    wandb.finish()  # Finalize run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--name', required=False, type=str, default='dev')
    parser.add_argument('--wandb_mode', required=False, type=str, default='disabled')   # online | offline | disabled
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination', 'counterfact'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--start_batch', default=0, type=int,
                    help='Batch index to start from (0 loads 0:K, 1 loads K:2K, etc.)')
    parser.add_argument('--sequential_edit', action="store_true")
    parser.add_argument('--project_suffix', default="", type=str)

    # parse_known_args returns a (known_args, unknown_args) tuple
    known_args, unknown_args = parser.parse_known_args()

    clean_overrites = [o.lstrip("-") for o in unknown_args]
    overrites_dict = OmegaConf.from_dotlist(clean_overrites)
    overrites_dict = OmegaConf.to_container(overrites_dict, resolve=True)
    
    return known_args, overrites_dict


if __name__ == "__main__":
    args, overrites_dict = parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'MEMOIR':
        editing_hparams = MEMOIRHyperParams
    # FIXME: added defer here not sure if it works
    elif args.editing_method == 'DEFER':
        editing_hparams = DeferHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size
    start_idx = args.start_batch * K
    end_idx = (args.start_batch + 1) * K

    if args.data_type == 'ZsRE':
        # edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:3000]
        # loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))
        
        # if 3000 > len(edit_data):
        #     import random
        #     random.seed(10)
        #     random.shuffle(loc_data)
        #     num_to_append = 3000 - len(edit_data)
        #     edit_data = edit_data + loc_data[-(num_to_append):]

        # edit_data = edit_data[start_idx:end_idx]
        # loc_data = loc_data[start_idx:end_idx]

        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    elif args.data_type == 'counterfact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-train.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        
        rephrase_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact.json', 'r', encoding='utf-8'))

        # when training with more than 1000 edits
        if K > len(edit_data):
            random.seed(10)
            random.shuffle(loc_data)
            num_to_append = K - len(edit_data)
            edit_data = edit_data + loc_data[-(num_to_append):]

        loc_data = loc_data[:args.ds_size]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]

        case_ids = [edit_data_['case_id'] for edit_data_ in edit_data]
        print(f"case_ids", case_ids)
        rephrase_prompts = [rephrase_data[case_id]['paraphrase_prompts'][0] for case_id in case_ids]

        # rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }


    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        # loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))
        loc_data = loc_data * 2
        loc_data = loc_data[start_idx:end_idx]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'temporal':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-edit.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-train.json', 'r', encoding='utf-8'))[start_idx:end_idx]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['ood_rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    # given that we are using the same model for all methods, we can set the model name here
    if args.editing_method == 'WISE':
        if args.ds_size > 1:
            hparams.save_freq = args.ds_size // 2
            hparams.merge_freq = args.ds_size
    

    # replacing hparams with passed arguments
    for key, value in overrites_dict.items():
        setattr(hparams, key, value)

    # Initiate wandb
    wandb_run = setup_wandb(hparams, args.name, args.wandb_mode, args.data_type, args.editing_method, args.ds_size, args.project_suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl',
        'counterfact': 'token em',
    }
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

    # resume_wandb(wandb_run)
