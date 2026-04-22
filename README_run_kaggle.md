## Run BPF on Kaggle (Notebook Guide)

This guide explains how to run the BPF project on Kaggle using:
- `kaggle_bpf.ipynb` (main runnable notebook)
- Kaggle Pascal VOC dataset at `/kaggle/input/voc0712`

The notebook is designed to be practical and defensive: it checks GPU, installs dependencies, builds extensions, maps dataset paths, patches YAML configs, runs staged training, and visualizes predictions.

## 1) What To Upload To Kaggle

Create a Kaggle Notebook and add:
1. Your GitHub repo: `https://github.com/HelcurtLordno1/Continual-Learning-Bridge-Past-and-Future`
2. Dataset: `voc0712` with this structure:
	 - `/kaggle/input/voc0712/VOCdevkit/VOC2007`
	 - `/kaggle/input/voc0712/VOCdevkit/VOC2012`

Recommended Notebook settings:
- Accelerator: GPU
- Internet: ON (for `git clone` and package install)

## 2) What `kaggle_bpf.ipynb` Does

The notebook executes these blocks in order:
1. Environment bootstrap and GPU check.
2. Clone/pull repository into `/kaggle/working/Continual-Learning-Bridge-Past-and-Future`.
3. Install dependencies and run `python setup.py build develop`.
4. Create symlinks from read-only Kaggle dataset to repository paths.
5. Build PascalVOCSearchDataset compatibility class (from your sample notebook logic).
6. Create Kaggle runtime YAML configs under `configs/kaggle_runtime/`.
7. Run stage-1 -> trim -> stage-2 -> stage-3 training.
8. Run evaluation and prediction visualization.

## 3) Dataset Mapping Strategy

Because `/kaggle/input` is read-only, the notebook creates symlinks instead of copying data.

It links:
- `/kaggle/input/voc0712/VOCdevkit` -> `datasets/VOCdevkit`
- `/kaggle/input/voc0712/VOCdevkit` -> `data/voc07/VOCdevkit`

This keeps compatibility with the codebase paths in `maskrcnn_benchmark/config/paths_catalog.py`.

## 4) Stage Commands Used

The notebook runs these command families:

### Stage 1 (source model)
`python tools/train_first_step.py -c <generated stage1 yaml> --skip-test`

### Trim checkpoint
`python tools/trim_detectron_model.py --name '10-10/stage1'` (best effort)

If trim utility path assumptions mismatch, notebook falls back by copying `model_final.pth` to `model_trimmed.pth`.

### Stage 2 (intermediate model)
`python tools/train_first_step.py -c <generated stage2 yaml> --skip-test`

### Stage 3 (target BPF model)
`python tools/train_incremental_finetune_all.py -t 10-10 -n kaggle --cls 0.15 -l 0.4 -high 0.7 -lw 1.0 -hw 0.3 --skip-test`

Important: `train_incremental_finetune_all.py` reads a fixed target YAML path. The notebook temporarily replaces that YAML with the generated Kaggle runtime YAML, runs stage-3, then restores the original file.

## 5) Where Outputs Are Saved

All writable artifacts go under `/kaggle/working/output`:
- `/kaggle/working/output/10-10/stage1`
- `/kaggle/working/output/10-10/stage2`
- `/kaggle/working/output/10-10/stage3`
- TensorBoard logs in `/kaggle/working/output/tb/...`

This ensures outputs survive in Kaggle working storage for download.

## 6) Parameter Tuning (Speed vs Quality)

Main tuning location in notebook: YAML patching cell.

Fast debugging (recommended first run):
- `QUICK_RUN = True`
- `MAX_ITER = 500`
- `IMS_PER_BATCH = 2`
- `CHECKPOINT_PERIOD = 200`

Higher quality training:
- `QUICK_RUN = False`
- Use original config values from `configs/OD_cfg/...`
- Increase runtime budget on Kaggle accordingly.

Other knobs:
- Stage flags: `RUN_STAGE1`, `RUN_STAGE2`, `RUN_STAGE3`
- Distillation and pseudo-label weights in stage-3 command:
	- `--cls`
	- `-l` / `-high`
	- `-lw` / `-hw`

## 7) Files You Can Modify For Custom Experiments

Core training behavior:
- `tools/train_first_step.py`
- `tools/train_incremental_finetune_all.py`

Pseudo-label and distillation logic:
- `maskrcnn_benchmark/modeling/pseudo_labels.py`
- `maskrcnn_benchmark/distillation/finetune_distillation_all.py`
- `maskrcnn_benchmark/modeling/attention_map.py`
- `maskrcnn_benchmark/modeling/select_unk.py`

Dataset path registration:
- `maskrcnn_benchmark/config/paths_catalog.py`

## 8) Troubleshooting

1. Dataset not found
- Confirm `/kaggle/input/voc0712/VOCdevkit/VOC2007` exists.
- Re-run symlink cell.

2. Build/import extension failure
- Re-run `python setup.py build develop` cell.
- Ensure all dependencies installed.

3. Stage-3 cannot find source/finetune weights
- Verify stage-1/stage-2 finished and wrote expected files in `/kaggle/working/output/10-10/...`.
- Confirm generated stage-3 YAML contains correct `MODEL.WEIGHT`, `SOURCE_WEIGHT`, `FINETUNE_WEIGHT`.

4. Out-of-memory
- Lower `IMS_PER_BATCH`.
- Keep `QUICK_RUN = True` for validation.

## 9) Recommended Execution Order

Run notebook cells sequentially from top to bottom:
1. Imports and GPU checks
2. Clone repo and install
3. Dataset symlink setup
4. YAML generation
5. Main training
6. Evaluation
7. Visualization

If any stage fails, fix that stage before continuing to the next.

