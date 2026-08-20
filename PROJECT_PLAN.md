# Technical working notes

Use `plan.md` for the eleven phases and `progress.md` for actual evidence. These notes collect implementation decisions to carry forward. Values labelled provisional must be checked before the main training runs.

## Goal and scope

Build a readable U-Net system for pancreas and pancreatic tumour/mass segmentation, compare attention and multi-scale fusion components fairly, and show predictions in a small viewer. Use the terminal on a remote GPU machine for training. The first review covers the project design.

Keep the code in focused Python modules called by the terminal commands. No database, login, hospital integration, or hosting is needed. The final output must align with the original 3D CT, even when the network itself uses slices.

## Data and split

- Raw data: `Task07_Pancreas/` beside `config.py`. Metadata and original NIfTI filenames stay unchanged.
- `imagesTr` and `labelsTr` contain the labelled cases. `imagesTs` has no local reference masks and cannot supply scored evaluation.
- Ignore Apple metadata files (`._*`, `.DS_Store`). Keep raw and prepared data outside Git.
- Audit readability, dimensions, finite intensities, voxel spacing, orientation, affines, label values, class volumes, and decoded-volume duplicates before sampling.
- Treat one unique MSD case ID as one patient unless additional linkage information becomes available. Exact-volume checks cannot identify every possible repeated acquisition from one person.
- Project split: seed 42, 70/15/15, largest-remainder rounding. For 281 usable cases this is 197/42/42. Balance tumour absence and volume tertiles. Save the actual lists once and keep them fixed across models and training seeds.
- Split rows use dataset-relative paths. They remain valid after copying the project or dataset to remote storage.
- Do not select a crop at validation or inference using a reference mask. Reference labels may support split stratification and training QA.

## Preparation to choose in Phase 4

Choose slice or patch input after the audit and memory check. A 2D axial baseline is a practical starting option, not a settled research result. Document the selected orientation, target size/spacing, intensity window, and normalization in `config.py`.

Continuous interpolation belongs to CT images; nearest-neighbour interpolation belongs to integer masks. Keep both transformations aligned. Save original shape, affine, orientation, and transformation parameters so predictions can return to the original grid. Inspect small lesions before and after preparation.

Use augmentation only for training, applied to image and mask together. Explain the sampler and include background as well as foreground samples. Validation and final inference cover complete scans. Cache by volume if it helps, without a complicated cache framework.

## Model components

Read `BASE_PAPER.md` for separate [U-Net](https://arxiv.org/abs/1505.04597), [Attention U-Net](https://arxiv.org/abs/1804.03999), and [UNet 3+](https://arxiv.org/abs/2004.08790) references. Their abstracts support the broad component ideas, not a complete combined implementation.

Write down levels, widths, normalization, attention equations, fusion scale alignment, projection and aggregation before implementation. Check component methods before attribution. Record project-specific choices as our choices. Keep the code and viewer free of repeated literature references. Combining components does not by itself prove novelty.

Use a plain U-Net, a matching attention version, the complete attention/fusion model, and one precisely defined component-removal experiment if useful. Keep their inputs, data split, loss, training budget, and validation rules matched. Save parameter counts and measured memory.

## Training to implement later

Keep paths and hyperparameters in `config.py`. Select loss, optimizer, learning rate, batch size, scheduler and stopping rule using training/validation checks. A Dice plus cross-entropy loss is a candidate, not a verified chosen setting. Do not decide by looking at held-out test scores.

Run a short GPU pilot and measure memory and epoch time before committing to a budget. Start with seed 42 and use 43/44 repeats for the baseline/full model only if the measured budget allows. Never change patient lists between training seeds.

Save latest and best checkpoints, settings, epoch CSV, optimizer/scheduler state, random states, and scaler state if mixed precision is used. Test reloading and resuming. Keep outputs on persistent storage and verify a new saved copy before replacing a usable one.

## Evaluation and demo

Use complete-volume prediction and original-grid restoration. Report pancreas-label and tumour/mass-label Dice, IoU, precision, recall and HD95 in millimetres separately. An optional union of both labels is a separately named whole-pancreas measure.

If a present reference class is completely missed, overlap and recall are zero; precision is undefined if no positive voxels are predicted. Report class-absent cases and their false-positive volumes separately, with valid-case counts and explicit undefined-distance rules. Keep all model selection on validation data.

Save per-case scores, comparison summaries, inference time, parameter counts and difficult examples. Report actual findings even if attention/fusion is weaker. Do not copy published scores into project results.

The viewer should load a CT, predict without a reference mask, move through slices, toggle overlays and save an aligned NIfTI mask. Reuse the shared preparation and prediction modules. Label it as a research demonstration. Segmentation alone is not a cancer diagnosis.

## Handover checks

- Audit report, exclusions, geometry and training overlays checked.
- Patient lists fixed and overlap-free under the documented case-ID assumption.
- Preparation and reversal checked on real data, including small lesions.
- Model forward/backward, learning, save/reload and resume checks passed.
- Actual remote library versions, device operation, memory and training time recorded.
- Final metrics and viewer use the same verified prediction pipeline.
- Report and slides describe actual operations, cite the components used, and distinguish measurements from intentions.

The original team names, supervisor and institution are retained in the first review presentation. The relative 16-week schedule and suggested responsibilities remain in `plan.md`.
