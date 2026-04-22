# BPF Runbook: Exact Step-by-Step Guide (This Folder)

This guide explains exactly how to run this repository end-to-end in its current folder state, what each step runs, where outputs are written, and what to change for speed or custom experiments.

Scope: this project is a script-driven pipeline (not notebook-driven) for incremental object detection (IOD) based on maskrcnn-benchmark.

## 1. What You Need Before Running

Minimum prerequisites:
- NVIDIA GPU with CUDA available.
- Python environment (recommended: isolated venv or conda env).
- PyTorch + torchvision compatible with your CUDA runtime.
- Dependencies from [requirements.txt](requirements.txt).
- Built extension via [setup.py](setup.py) (`python setup.py build develop`).
- Pascal VOC data available under the exact paths expected by [maskrcnn_benchmark/config/paths_catalog.py](maskrcnn_benchmark/config/paths_catalog.py).

Canonical paper environment from [INSTALL.md](INSTALL.md):
- Python 3.7
- PyTorch 1.10.0
- torchvision 0.11.0
- CUDA 11.3

Practical note for this workspace: modern Python/CUDA can work, but if you deviate from canonical pins, expect compatibility adjustments.

## 2. Verify Folder-State Before Training

Important: in this repository state, some script filenames referenced in [scripts/run_first_step.sh](scripts/run_first_step.sh) and [scripts/run_finetune_step.sh](scripts/run_finetune_step.sh) use `8x` config names that are not present for all tasks. Existing tested configs are mostly `4x` for 10-10 and 15-5.

Safe existing 10-10 files:
- [configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml](configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml)
- [configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_sec_10.yaml](configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_sec_10.yaml)
- [configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_BPF_Target_model.yaml](configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_BPF_Target_model.yaml)

Safe existing 15-5 files:
- [configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml](configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml)
- [configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_sec5.yaml](configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_4x_sec5.yaml)
- [configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_8x_BPF_Target_model.yaml](configs/OD_cfg/15-5/e2e_faster_rcnn_R_50_C4_8x_BPF_Target_model.yaml)

Recommendation: use direct Python commands (below) rather than wrapper shell scripts unless you first update script config paths.

## 3. Environment Setup (Windows + Linux)

### 3.1 Windows PowerShell

```powershell
cd "D:/Desktop_informations/Hoc AI-VN/AIO_side_project/Continual learning/Specific-task-paper/BPF"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup.py build develop
```

### 3.2 Linux/macOS Bash

```bash
cd /path/to/BPF
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup.py build develop
```

### 3.3 Quick sanity checks

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import maskrcnn_benchmark; print('maskrcnn_benchmark import ok')"
```

If import fails at extension level, re-run `python setup.py build develop` and verify compiler/CUDA toolchain.

## 4. Dataset Layout (Critical)

The dataset registry in [maskrcnn_benchmark/config/paths_catalog.py](maskrcnn_benchmark/config/paths_catalog.py) expects VOC 2007 at:
- `data/voc07/VOCdevkit/VOC2007`

Required subfolders include:
- `JPEGImages`
- `Annotations`
- `ImageSets/Main`

If this path does not exist, training will fail at dataloader creation.

## 5. Full Training Pipeline (Recommended 10-10 Path)

This project is a 3-stage pipeline:
1. Source model (old classes).
2. Intermediate model (current/new classes).
3. Final target model (BPF + DwF distillation).

### 5.1 Stage 1: train source model

Runs [tools/train_first_step.py](tools/train_first_step.py) with first-stage config.

```bash
python -m torch.distributed.launch --master_port=$(python get_free_port.py) --nproc_per_node=1 \
  tools/train_first_step.py \
  -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_att1_rpn1.yaml \
  --skip-test
```

What it runs:
- model build + optimizer + scheduler
- first-task training
- checkpoint writing to folder defined by YAML `OUTPUT_DIR`

Expected artifacts (example):
- `output/10-10/LR005_BS4_BF_att1_rpn1/model_*.pth`
- `output/10-10/LR005_BS4_BF_att1_rpn1/model_final.pth`

### 5.2 Stage 1.5: trim source checkpoint

Runs [tools/trim_detectron_model.py](tools/trim_detectron_model.py).

```bash
python tools/trim_detectron_model.py --name "10-10/LR005_BS4_BF_att1_rpn1"
```

What it runs:
- loads `model_final.pth`
- keeps only model weights
- writes `model_trimmed.pth`

Expected artifact:
- `output/10-10/LR005_BS4_BF_att1_rpn1/model_trimmed.pth`

### 5.3 Stage 2: train intermediate model

Runs [tools/train_first_step.py](tools/train_first_step.py) again with `sec` config.

```bash
python -m torch.distributed.launch --master_port=$(python get_free_port.py) --nproc_per_node=1 \
  tools/train_first_step.py \
  -c configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_sec_10.yaml \
  --skip-test
```

Expected artifact:
- `output/finetune/sec_10/model_final.pth`

### 5.4 Stage 3: train final BPF target model

Runs [tools/train_incremental_finetune_all.py](tools/train_incremental_finetune_all.py).

```bash
python tools/train_incremental_finetune_all.py \
  -t 10-10 -n test \
  --cls 0.15 \
  -l 0.4 -high 0.7 \
  -lw 1.0 -hw 0.3 \
  --skip-test
```

What it runs:
- loads source model (old teacher)
- loads finetune model (new teacher)
- builds trainable target model
- applies Bridge-the-Past pseudo-label merge
- applies DwF distillation losses
- saves target checkpoints

Expected artifacts:
- `output/10-10/test/STEP1/model_*.pth` (or task/name path set by script)
- final target checkpoint and trimmed checkpoint

## 6. Evaluation

Use [tools/test_net.py](tools/test_net.py) for standalone evaluation.

```bash
python tools/test_net.py \
  --config-file configs/OD_cfg/10-10/e2e_faster_rcnn_R_50_C4_4x_BPF_Target_model.yaml \
  MODEL.WEIGHT output/10-10/test/STEP1/model_final.pth
```

Adjust `MODEL.WEIGHT` to your actual trained checkpoint path.

## 7. What Each Main File Runs

Top-level wrappers:
- [scripts/run_first_step.sh](scripts/run_first_step.sh): stage-1 + trim flow (check config names first).
- [scripts/run_finetune_step.sh](scripts/run_finetune_step.sh): stage-2 flow (check config names first).
- [scripts/run_incre_finetune.sh](scripts/run_incre_finetune.sh): stage-3 flow and BPF flags.

Core executables:
- [tools/train_first_step.py](tools/train_first_step.py): standard train loop for source/intermediate models.
- [tools/train_incremental_finetune_all.py](tools/train_incremental_finetune_all.py): BPF target training with pseudo labels + DwF.
- [tools/trim_detectron_model.py](tools/trim_detectron_model.py): strips optimizer state from final checkpoint.
- [tools/test_net.py](tools/test_net.py): test/inference entrypoint.

Method internals:
- [maskrcnn_benchmark/modeling/pseudo_labels.py](maskrcnn_benchmark/modeling/pseudo_labels.py): pseudo-label merge (Bridge the Past).
- [maskrcnn_benchmark/modeling/attention_map.py](maskrcnn_benchmark/modeling/attention_map.py): attention maps for future filtering.
- [maskrcnn_benchmark/modeling/select_unk.py](maskrcnn_benchmark/modeling/select_unk.py): unknown/future proposal selection.
- [maskrcnn_benchmark/distillation/finetune_distillation_all.py](maskrcnn_benchmark/distillation/finetune_distillation_all.py): DwF distillation losses.
- [maskrcnn_benchmark/data/datasets/voc.py](maskrcnn_benchmark/data/datasets/voc.py): class filtering and label behavior.

## 8. Where to Tune Speed, Memory, and Runtime

Main knobs in YAML configs under [configs/OD_cfg](configs/OD_cfg):
- `SOLVER.MAX_ITER`: fewer iterations = faster run.
- `SOLVER.STEPS`: move milestones earlier for short schedules.
- `SOLVER.BASE_LR`: adjust when changing batch size.
- `SOLVER.IMS_PER_BATCH`: lower if VRAM is limited.
- `SOLVER.CHECKPOINT_PERIOD`: increase interval to reduce checkpoint I/O overhead.
- `DATALOADER.NUM_WORKERS`: reduce on Windows if dataloader is unstable.

Fast debug profile (quick pipeline verification):
- set `MAX_ITER` to 200-1000
- set `IMS_PER_BATCH` to 1-2
- use `--skip-test` during training
- run one split only (10-10 first)

## 9. Which Python Files to Modify for Custom Needs

If you want method changes:
- Pseudo-label policy or thresholds: [maskrcnn_benchmark/modeling/pseudo_labels.py](maskrcnn_benchmark/modeling/pseudo_labels.py)
- Future object filtering: [maskrcnn_benchmark/modeling/attention_map.py](maskrcnn_benchmark/modeling/attention_map.py), [maskrcnn_benchmark/modeling/select_unk.py](maskrcnn_benchmark/modeling/select_unk.py)
- Distillation behavior and weighting: [maskrcnn_benchmark/distillation/finetune_distillation_all.py](maskrcnn_benchmark/distillation/finetune_distillation_all.py)
- Task-level flags and training orchestration: [tools/train_incremental_finetune_all.py](tools/train_incremental_finetune_all.py)
- Source/intermediate trainer behavior: [tools/train_first_step.py](tools/train_first_step.py)
- Dataset/category handling: [maskrcnn_benchmark/data/datasets/voc.py](maskrcnn_benchmark/data/datasets/voc.py)

If you want command defaults changed:
- update the wrapper scripts in [scripts](scripts) to point to valid local config names.

## 10. Common Failure Points and Fixes

1. Config file not found
- Cause: script references `8x` filename that is absent.
- Fix: switch to existing `4x` config path in command or edit wrapper script.

2. Dataset not found
- Cause: VOC path does not match [maskrcnn_benchmark/config/paths_catalog.py](maskrcnn_benchmark/config/paths_catalog.py).
- Fix: create expected folder structure under `data/voc07/VOCdevkit/VOC2007`.

3. Checkpoint chain fails at stage 3
- Cause: `SOURCE_WEIGHT` or `FINETUNE_WEIGHT` path mismatch in target YAML.
- Fix: verify generated stage-1/stage-2 output paths and update YAML accordingly.

4. CUDA/OOM issues
- Cause: batch size too high or incompatible torch/cuda build.
- Fix: lower `IMS_PER_BATCH`, reduce image/batch load, confirm CUDA-enabled torch.

5. Build/import extension errors
- Cause: C++/CUDA extension not built.
- Fix: run `python setup.py build develop` and verify compiler/CUDA toolkit.

## 11. Recommended Reproducible Run Order (Checklist)

1. Create environment and install dependencies.
2. Build extension with `python setup.py build develop`.
3. Verify CUDA and package import.
4. Verify VOC folder structure matches dataset catalog paths.
5. Run stage 1 training.
6. Run checkpoint trimming.
7. Run stage 2 training.
8. Run stage 3 target training.
9. Run evaluation.
10. Archive exact YAMLs and final checkpoints for reproducibility.

## 12. Final Practical Advice

- Treat this project as a staged system, not a single-command trainer.
- Validate each stage output before moving to the next stage.
- Keep commands and YAML paths explicit in experiment logs.
- If sharing with others, include your exact environment version, task split, YAML files, and checkpoint paths.

Following this runbook carefully is sufficient for other users to clone and run this project reliably on a correctly configured machine.