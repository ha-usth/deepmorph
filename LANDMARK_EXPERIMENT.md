# Per-landmark evaluation for the revision

## Current manuscript table selection

At the author's request, the manuscript preview table now uses **PB for
Droso-small and Tsetse, seeds 0–9**, **PC for Cepha, seeds 0–4**, and **PA for the other
six datasets, seeds 0–9**. Each landmark mean and sample SD is recomputed
directly from the selected runs, without adjusting or scaling error values.
Fly remains unchanged; its SD difference from Table 2 (0.17 versus 0.18)
is temporarily accepted by the author, not numerically corrected.

The Droso-small row now aggregates to **4.80 ± 0.09 px**, with NME
**0.283 ± 0.005%**, matching Table 2 at its displayed precision.
Tsetse now uses PB and aggregates to **5.52 ± 0.10 px** (unrounded
5.521925965 ± 0.104414681), with NME **0.337 ± 0.006%**. Its MRE mean
matches Table 2, but its SD does not match the 0.09 in Table 2 / Table 3's
PA column. The new Tsetse values match Table 3's PB column at printed
precision. No SD was adjusted to force agreement.
Cepha now aggregates to **22.66 ± 0.39 px**, also matching Table 2 at its
displayed precision. At the author's request, the landmark table's PNG/SVG
previews and LaTeX caption omit protocol names and seed information, showing
mean ± SD only. This is a presentation change: the selections above and JSON
provenance remain unchanged. The Droso-small comparison plot retains its own
label. The experimental methods should describe the selections accurately;
the table does not assert a common protocol or a common number of runs.

Run `python make_landmark_figures.py` followed by
`python check_landmark_table2.py` to regenerate the manuscript previews and
consistency check in `results/landmark_manuscript/`.
Remaining numerical differences (including Tsetse's SD) are preserved.
No underlying E2 run or prediction was changed. The PA descriptions and initial
audit below remain as experiment history, distinct from this selected table.

## Re-evaluating the saved predictions

Run `python e5_landmark_errors.py` from the repository root. This re-evaluates
existing E2 prediction files; it does not train new models. Outputs are in
`results/e5_landmark_errors/original_test/<dataset>/`.
The default evaluation retains duplicate images, as requested. Earlier
`clean_test` outputs are historical sensitivity analyses, not the primary results;
they are regenerated only with the optional `--also-filter-overlap` flag.

## Dataset choice and protocol

- **droso_small**: primary analysis, 15 landmarks, 15 training images. This
  maintains the N=15 setting used for the per-landmark experiment in iMorph.
- **sea_bass**: complementary body-shape dataset, 11 landmarks, 20 training images.
- **cepha**: complementary cephalometric dataset, 19 landmarks, 20 training images.

The choices cover different image domains, rather than selecting datasets by
which protocol wins. Evaluate all three existing heatmap protocols PC, PA, PB
with seeds 0–9. Use the original checkpoints' predictions, without selecting
seeds or checkpoints by test performance. PC/PA/PB retain the labels in the
current E2 results (older logs use the previous protocol names).

iMorph reference: [Nguyen et al., 2022](https://doi.org/10.1016/j.ecoinf.2022.101694).
This is a comparable *type of analysis*, not a reproduced numerical comparison
against iMorph: identical splits, preprocessing and landmark numbering have
not been independently established, and iMorph predictions were not rerun.

## Metrics and exported artifacts

For each image and landmark, compute Euclidean distance between prediction
and annotation in original-image pixels. NME is 100 times this distance divided
by that image's diagonal. First average over test images within each seed,
then report mean and sample SD across the ten seed means. Error bars are SD,
not confidence intervals and not between-image dispersion.

`per_landmark.csv` includes both metrics, seed SD, pooled median, P90 and P95.
Pooled percentiles describe the error distribution across image/seed pairs;
those pairs are not independent observations for statistical testing.
`per_image_seed_landmark.csv` and `errors.npz` preserve the raw distances.
`per_landmark.png` and `.svg` provide publication figures. `landmark_map.png`
shows annotation row IDs on the first filename (not a selected success case).
`audit.json` and `cache_checks.csv` record exclusions and reproduction checks.

## Data audit

- droso_small: 100 test images on disk; `073.bmp` has identical image bytes to
  an image in train_pool. The primary analysis retains all 100 test images.
- sea_bass: 150 test images on disk; `1480731131.jpg` has only 10 of the expected
  11 annotations. Both analyses exclude it and use 149 images. This reproduces
  the existing E2 scorer's exclusion. No identical-byte train/test overlap found.
- cepha: 250 test images on disk; `173.bmp` has identical image bytes to an image
  in train_pool. The primary analysis retains all 250 test images.

No source images or annotations were changed. Original-test recalculations
match the saved per-landmark means to within 0.00001 pixels for all 90 runs.
Confirmed overlaps are recorded but retained, as requested.
This audit checks exact file-byte duplicates within
each dataset, not re-encoded duplicates, subject-level overlap, cross-dataset
overlap, or all Stage 1 pretraining inputs. It is not a complete leakage audit.

## Findings on original test sets (duplicates retained)

PA is used below as a fixed descriptive reference, not because it performs best.
The figures and CSV files show all three protocols.

- **droso_small:** overall MRE 4.826 ± 0.124 px. L6 has the largest mean error
  (5.644 ± 0.224 px); L1 (5.639 ± 0.257) and L13 (5.565 ± 0.656) are also high.
  L5 has the smallest mean (4.198 ± 0.184). L13 is less stable between seeds.
  PB also ranks L6 hardest, while PC ranks L1 hardest: the exact ranking is
  not invariant to protocol.
- **sea_bass:** overall MRE 4.408 ± 0.088 px. L11 is clearly the hardest
  (9.657 ± 0.413 px), versus L3 at 2.559 ± 0.071 px, approximately 3.77 times
  smaller. All three protocols agree on the hardest and easiest landmarks.
- **cepha:** overall MRE 22.999 ± 0.426 px. L16 is hardest (42.815 ± 3.990 px),
  followed by L4 (33.686 ± 1.973) and L10 (31.086 ± 1.271). All protocols agree
  on L16 hardest and L9 easiest.

Use droso_small as the direct methodological continuation of the iMorph
analysis; sea_bass and cepha demonstrate that averaging all landmarks can hide
substantial landmark-specific differences. Do not infer anatomical causes
(occlusion, ambiguous boundaries, annotation uncertainty) solely from these
numbers. Landmark maps and high-error specimens require domain inspection.
Do not compare pixel MRE directly across datasets with different resolutions,
or claim statistical superiority between protocols from overlapping SD bars.

Suggested revision wording, to adapt once the reviewer's exact comment is available:

> We added a landmark-wise evaluation on Droso-small, Sea-bass, and Cepha,
> using ten existing independent fine-tuning seeds for each pretraining
> protocol. We report mean radial error at the original image resolution and
> image-diagonal-normalized error for each landmark, with standard deviations
> across seed-level means. The results reveal heterogeneous localization
> difficulty: under PA, the highest mean errors occur at landmarks 6, 11,
> and 16, respectively. The complementary datasets show that the aggregate
> error can conceal substantial differences between landmark classes.
> Evaluation retained the original test splits, including one exact image
> overlap with the training pool in each of Droso-small and Cepha. One Sea-bass
> image with incomplete annotations could not be scored using the complete
> 11-landmark configuration. The evaluated test sizes were therefore 100,
> 149, and 250 images, respectively. These details are recorded for reproducibility.

## Initial all-PA audit against DeepMorph_revision (34).pdf, Table 2

This section records the initial audit before the author's PB selection for
Droso-small. The generated consistency files now reflect the current selection
at the top of this document.

Checked on 2026-10-06 against page 11 of the supplied manuscript. The original
landmark table matches the saved PA heatmap runs (seeds 0–9) exactly before
rounding. Its annotation budgets also match Table 2. Averaging the landmark
means recovers overall MRE; the overall SD must instead be computed across
the ten run-level overall MRE values, not by averaging landmark SDs.
Maximum discrepancy between the reconstructed per-run overall MRE and the
cached per-run MRE is 0.00000231 pixels across these 90 runs.

The manuscript's MRE mean/SD agrees at printed precision for five datasets:
Droso-big, Bactro, Diacha, Droso-281, and Sea-bass. Four need reconciliation:

- Droso-small: manuscript 4.80 ± 0.09; current PA 4.83 ± 0.12.
- Fly: manuscript 3.05 ± 0.18; current PA 3.05 ± 0.17.
- Tsetse: manuscript 5.52 ± 0.09; current PA 5.53 ± 0.09.
- Cepha: manuscript 22.66 ± 0.39; current PA 23.00 ± 0.43.

NME also needs updates: Droso-small to 0.284 ± 0.007%, Bactro to
0.340 ± 0.016%, Droso-281 to 0.352 ± 0.010%, Cepha to 0.746 ± 0.014%.
Other Table 2 NME cells match the current runs at printed precision.

The Cepha manuscript pair 22.66 ± 0.39 matches PC seeds 0–4
(22.661516 ± 0.394399), including the old P0 backup, rather than current
PA seeds 0–9. The Droso-small manuscript pair matches current PB at printed
precision. These are provenance clues, not proof of the manuscript's editing
history. The migration explicitly maps P1 to PA, P2 to PB, and P0 to PC.
Do not relabel protocols or scale per-landmark values to force agreement.

The recommended reconciliation is to generate the Ours columns of Table 2,
the corresponding Table 3 entries, and the landmark table from the same
identified PA runs. Recheck dependent narrative, macro-averages and statistical
comparisons when replacing Table 3. Keeping the old Table 2 instead requires
recovering its exact source runs and explaining any different protocol/seed
count; a landmark breakdown cannot be recovered from an aggregate alone.

Run `python check_landmark_table2.py` to reproduce this check. Outputs in
`results/landmark_manuscript/` include `table2_consistency.png`,
`table2_consistency.json` (full precision and source hashes), and
`table2_ours_proposed.tex` (proposed Ours-column values only). The PDF and
experimental results were not modified. The check uses cached per-seed errors;
it does not rerun training or re-score all nine datasets from coordinates.

Evaluated image counts in these runs are 100, 1100, 5, 28, 31, 1000, 222,
149, and 250 in the table's dataset order. In particular, distinguish Table 1's
image counts from actual scored counts for Diacha, Droso-281, and Sea-bass;
the reason for the first two discrepancies is not established by this check.

## Imaging-degradation benchmark (one frozen checkpoint per dataset)

Use `benchmark_degraded_test_set.py` after `make_degraded_test_set.py` has
finished. Run in the Python environment used for Stage 3 (PyTorch, torchvision,
the model dependencies, numpy, Pillow, tqdm and matplotlib). No training is
performed. The default is PA, all eight corruption types, all three levels,
and every eligible test image: 25 evaluations per checkpoint. `--corruptions`
can select fewer types. The clean condition is always included.

The checkpoint paths below are placeholders: replace each with the actual
trusted **Stage 2 fine-tuned PA** checkpoint for that dataset, not a Stage 1
MAE checkpoint. The script records its SHA-256 and saved training metadata;
the PA label is declared by the user and cannot be established from weights
alone. Do not substitute the previously selected PB/PC table runs and call
them PA. Clean results are recomputed using the same model as the degraded
conditions and need not equal an aggregate from Table 2.

```powershell
python benchmark_degraded_test_set.py --dataset droso_small --checkpoint "D:/path/to/PA_droso_small_finetuned.pth"
python benchmark_degraded_test_set.py --dataset sea_bass --checkpoint "D:/path/to/PA_sea_bass_finetuned.pth"
python benchmark_degraded_test_set.py --dataset cepha --checkpoint "D:/path/to/PA_cepha_finetuned.pth"
```

The default input root is `results/degraded_test_set` (the directory containing
dataset folders). Override it with `--input-root PATH`. The default output is
`results/degradation_benchmark/DATASET/PA`; override with `--output-dir PATH`
when evaluating another checkpoint. For a selection of three corruptions:

```powershell
python benchmark_degraded_test_set.py --dataset droso_small --checkpoint "D:/path/to/PA_droso_small_finetuned.pth" --corruptions gaussian_noise gaussian_blur illumination --output-dir results/degradation_benchmark/droso_small/PA_three_types
```

Available types are `gaussian_noise`, `gaussian_blur`, `low_contrast`, `darken`,
`brighten`, `illumination`, `low_resolution`, and `jpeg`. Omit `--corruptions`
or use `--corruptions all` to evaluate all eight. Levels are always 1, 2, 3.
Image size and head type are read from checkpoint metadata. If the checkpoint
has no image size, supply `--image-size` matching its training configuration.

Outputs:

- `results.json`: configuration, checkpoint/source/input hashes, image list,
  exclusions, completion status, per-condition MRE/NME, per-landmark MRE/NME,
  absolute error changes and relative MRE increase over clean. It is saved
  atomically after each completed condition, so partial progress is preserved.
- `arrays/clean.npz` and `arrays/TYPE_level_N.npz`: image names, predictions,
  ground truth, original image sizes, and per-image/per-landmark pixel errors
  and normalized errors. No pickle is needed to reload these files.
- `degradation_mre.png` and `.pdf`: one clean bar and a three-bar group for
  each corruption. The dashed line marks the clean baseline. One checkpoint
  does not provide between-training-run SD, so there are no error bars.

The metric is mean Euclidean distance over images and landmarks at original
resolution. NME divides each distance by that image's diagonal and multiplies
by 100. Predictions are rounded to two decimals before scoring, matching E2's
scoring of Stage 3 output files. Relative MRE increase is undefined (`null`)
when clean MRE is zero. Original duplicate images are retained. An image with
an incorrect annotation count is recorded and excluded from **every** condition
(e.g. the incomplete Sea-bass annotation); missing predictions or changed
image/annotation pairing cause failure, not silent changes of test membership.

Read saved results without inference:

```python
import json
from pathlib import Path
import numpy as np

folder = Path('results/degradation_benchmark/droso_small/PA')
results = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
for row in results['conditions']:
    print(row['condition'], row['mre_px'], row['nme_percent'],
          row['relative_mre_increase_percent'])
with np.load(folder / 'arrays/gaussian_noise_level_1.npz', allow_pickle=False) as data:
    names = data['image_names']
    errors = data['errors_px']  # shape: images x landmarks
```

To redraw the figure or switch to NME, no checkpoint, test images, or PyTorch
installation is needed:

```powershell
python benchmark_degraded_test_set.py --plot-only results/degradation_benchmark/droso_small/PA/results.json
python benchmark_degraded_test_set.py --plot-only results/degradation_benchmark/droso_small/PA/results.json --metric nme
```

Add `--resume` to the original benchmark command to reuse completed conditions.
Resume requires unchanged inputs, code, environment, checkpoint and inference
options. An incomplete condition is recomputed. Existing output is otherwise
not overwritten. `--dry-run` checks image/annotation pairing without loading the
model or writing output. `--limit N` is for smoke tests only and marks the plot
as a subset run; omit it for manuscript experiments. `--no-plot` saves scores
without rendering, and `--device cpu` overrides automatic CUDA selection.

Validation: five unit checks cover paired membership, duplicates, annotation
mismatches, known geometric errors/normalization, and invalid predictions.
A separate one-image GPU smoke run completed all 25 conditions and generated
MRE/NME PNG/PDF figures; this is a software check, not a paper result or a
verification of PA provenance for the smoke-test checkpoint.
