# Fleet-sweep fixes: `prithvi_eo_feature_extraction_colab.ipynb` (2026-10-05)

A targeted fix of the 2026-10-05 fleet sweep findings for this notebook. There is no full Notebook Review Framework v1 report;
each flag was first confirmed in the cell source at `main` `9380af2`. All changes are made in the generator
(`tools/build_notebook.py`, `tools/notebook_template.py`); the notebook is regenerated. Status and release labels are unchanged.
The workshop notebook (`DIMER_AI_for_Earth_Observation_and_Climate_Applications_Workshop.ipynb`, its own generator, PR #12/#13
fixes) is untouched; `tools/eo_workshop/build_workshop.py --check` passes.

**Readiness: Verification pending** (until a hosted Run all of the regenerated notebook is recorded).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed: Section 1 ran `pip install` of the pins into the kernel and raised "Restart the runtime" on stale modules. The generator is now `build_notebook.py/2.2` (the fleet's shared isolated-runtime version) and the template opts in: one kernel cell downloads the size/SHA-256-pinned `uv` 0.12.15 wheel, builds a managed CPython 3.12.12 environment from the carried hash lock `tutorials/requirements-colab.lock.txt` (131 packages, `--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused when complete; re-running Section 1 keeps the live worker and its variables; the worker forces `MPLBACKEND=Agg` and drops `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. Section 8 imports `importlib.metadata` itself (the old install cell had provided it). | Section 1 (kernel cell + runtime record), Section 8; `tools/build_notebook.py`, `tools/notebook_template.py`, `tools/validate_release_assets.py` (bootstrap check), new lock | `test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested`, `test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin`, `test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker` |
| SWP-G (guided layer) | Fixed | Confirmed: the notebook declared GUIDED with 1 of 9 guided markers. Added an opening cell (audience, Input → Model → Output table, How to use this notebook, roadmap), Predict prompts before Sections 4–8's results, What to notice + Check your reasoning after each with answers quoting the recorded Kaggle T4 run of 2026-09-25 (probe burn-scar IoU 0.8149 / F1 0.8980 vs baseline 0 at accuracy 0.7865; reconstruction MSE 0.0773 vs 0.1294; validation loss 0.6931 → 0.1996, F1 0.7577), Troubleshooting, Glossary and a Conclusion template; Sections 1–3 are labelled Infrastructure and collapsed. A literal `{{id, frames}}` / `{{id, image, label}}` in the Prerequisites (not format-processed) now renders as `{id, …}`. | opening, Sections 4–8 markdown, closing, Prerequisites | `test_swp_g_guided_layer_is_present_and_infrastructure_is_collapsed`, `test_swp_g_checkpoint_answers_quote_the_recorded_run` |
| SWP-A (quality asserts) | Fixed | Confirmed: Section 7 asserted `probe burn-scar IoU > baseline`, which aborts a BYOD run before export. It is now a recorded verdict (`improved` / `no improvement` / `worse`) printed, written to the evaluation report and `result.json`. The two procedural checks (kept epoch's validation loss ≤ zero head's; re-scoring reproduces the kept epoch) remain hard stops as explicit `RuntimeError`s; the reload-parity assert is unchanged. | Section 7, Section 8 (`result.json`) | `test_swp_a_no_model_quality_assert_remains`, `test_swp_a_section_7_records_the_verdict_and_writes_the_report[0.0-no improvement / 0.42-improved]`, `test_swp_a_contract_checks_still_stop_the_run`, `test_swp_a_result_json_carries_the_verdict` |
| SWP-F (frozen re-run) | Fixed | Confirmed: re-running Section 5 after Section 6 scored the trained probe under the label `frozen_zero_head`. Section 5 now sets the attached probe aside for its two evaluations and restores it in a `finally`. (`adapt` already restarts from a zero head on every call, so Section 6 re-runs were not affected.) | Section 5 | `test_swp_f_frozen_evaluation_sets_the_trained_probe_aside_and_restores_it` |
| SWP-B (BYOD upload only) | Fixed | Confirmed: BYOD used only `google.colab.files.upload()`. Added a `BYOD_PATH` form field (a .zip or folder in the runtime, so Kaggle/Jupyter work); the upload is the fallback and is guarded: off Colab, a cancelled or multi-file upload, or a non-.zip each stop with a message naming the file or the rule. | Section 4 | `test_swp_b_byod_path_reads_a_folder_without_colab`, `test_swp_b_missing_path_and_off_colab_upload_give_clear_messages`, `test_swp_b_cancelled_or_wrong_upload_is_refused` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run, several minutes) or reuses `dimer_isolated_env_<lock digest>/` and runs every later cell there. Linux x86_64 runtimes only (Colab, Kaggle, Linux Jupyter).
- New `BYOD_PATH` form field in Section 4.
- Section 7 prints a `verdict` line and no longer raises when the probe does not beat the majority baseline; `evaluation_report.json` and `result.json` carry `comparison.verdict` / `verdict`.
- Guided-layer cells added; infrastructure cells collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages can run here (the Hugging Face Hub is unreachable). Synthetic / stand-in: the Section 1 cell is executed for real against a stand-in environment (a symlink to the test interpreter) to show reuse and idempotence; the Section 4 BYOD block, the Section 5 frozen-evaluation block and the Section 7 cell are executed with stand-in pipelines. That is plumbing evidence, not pretrained-inference evidence.
- The lock was compiled with uv 0.12.15 from the unchanged `pyproject.toml` pins for `x86_64-manylinux_2_28` (it resolves the same pre-release `hydra-core` 1.4.0.dev10 the workshop lock carries); `check_lock` confirms every pin is locked at its version with hashes.
- `python tools/build_notebook.py --check`: up to date. `python tools/eo_workshop/build_workshop.py --check`: OK (workshop unchanged).
- `python tools/validate_release_assets.py`: PASS.
- `ruff check src tests tools`: all checks passed.
- `pytest` with CI's dependencies only (pytest, ruff, numpy, tifffile; torch absent): 62 passed, 2 skipped before → 78 passed, 2 skipped after.
- Sweep re-check on the regenerated notebook: install `ok(isolated)`, guided markers 9/9, quality asserts 0.

## Remaining gates

- A hosted **Run all in one pass** of the regenerated notebook in a fresh Colab T4 runtime (no restart expected), then a re-run of the Section 8 export cell.
- The REL12 BYOD run (`USE_BYOD = True` with a `BYOD_PATH` zip or folder).
- A full Notebook Review Framework v1 review has not been done.
