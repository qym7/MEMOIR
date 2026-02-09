# MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs

This is the source code for [MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs
](https://arxiv.org/abs/2506.07899).

The repo is based on [EasyEdit](https://github.com/zjunlp/EasyEdit) repository.

![](framework.jpg)

Please check more information on the [project website](https://qym7.github.io/MEMOIR-Web/).

---

## Overview

MEMOIR is a scalable framework for lifelong model editing that integrates new knowledge without retraining or overwriting existing information. It stores edits in a residual memory module using sparse, sample-dependent masks to minimize interference and reduce forgetting. At inference, MEMOIR retrieves relevant knowledge by matching activation patterns, achieving balanced reliability, generalization and locality performance after thousands of sequential edits.

- Code location: `./easyeditor/models/memoir`  
- Hyperparameters: `./hparams/MEMOIR`

---

## Environment Setup

1. Create the conda environment:
   ```bash
   conda create -n lifelong_edit python=3.9 -y && source activate lifelong_edit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the project path:
   ```bash
   export PYTHONPATH="path_of_codes:$PYTHONPATH"
   ```

4. Login to Hugging Face:
   ```bash
   huggingface-cli login --token your_token
   ```

---

## Dataset Prepartion

Please refer to the [EasyEdit](https://github.com/zjunlp/EasyEdit) repository for dataset prepartion.

---

## Running MEMOIR

### Question Answering with 100 Edits on ZsRE dataset

```bash
python examples/run_wise_editing.py \
  --editing_method=MEMOIR \
  --hparams_dir=./hparams/MEMOIR/llama3-8b.yaml \
  --data_dir=data_dir \
  --ds_size=100 \
  --data_type=ZsRE \
  --sequential_edit \
  --top_k=4096 \
  --irr_threshold=0.4
```

### Hallucination Correction with 100 Edits on SelfcheckGPT dataset

```bash
python examples/run_wise_editing.py \
  --editing_method=MEMOIR \
  --hparams_dir=./hparams/MEMOIR/llama3-8b.yaml \
  --data_dir=data_dir \
  --ds_size=100 \
  --data_type=hallucination \
  --sequential_edit \
  --top_k=4096 \
  --irr_threshold=0.9
```

---

## Other usage

### Iterating over irrelevant samples to obtain background features

```bash
python examples/run_wise_editing.py \
  --editing_method=MEMOIR \
  --hparams_dir=./hparams/MEMOIR/llama3-8b.yaml \
  --data_dir=data_dir \
  --ds_size=100 \
  --data_type=ZsRE \
  --sequential_edit \
  --RUN_SAVE_BACKGROUND_FEATURES=True
```
---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wang2025memoir,
  title={MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs},
  author={Wang, Ke and Qin, Yiming and Dimitriadis, Nikolaos and Favero, Alessandro and Frossard, Pascal},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

```

---
