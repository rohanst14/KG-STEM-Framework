# Reproducibility guide

- Provide TEP training NPZ and causal adjacency NPY via the training CLI. Consult the training loader for required array keys and shapes.
- Evaluation defaults refer to `dataverse_files/TEP_FaultFree_Testing.npz` and `dataverse_files/TEP_Faulty_Testing.npz`.
- Supply the training artifacts in `kg_stgm_outputs/`, including configuration, scalers, adjacency and checkpoint (default `best_model.pt`).
- Faulty evaluation imports `FAULT_TABLE_SOURCE` from `local_risk`, which is absent. Obtain the original module from the author rather than fabricating fault metadata.
- Some scripts use a platform-specific Matplotlib configuration path. Set MPLCONFIGDIR to a writable directory if necessary.
- Risk analysis may require normal statistics, severity models and safe operating limits through CLI options. Record their provenance and calibration.
- Record environment versions, seeds, dataset versions, graph construction and evaluation parameters. Only load trusted serialized artifacts.

The training file now matches the existing `from train import ...` statements. All original code contents remain unchanged. Runtime and numerical validation are outstanding.
