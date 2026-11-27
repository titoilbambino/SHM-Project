# Research experiments

- Write concise research notes in `experiment_notebooks/NNN.md`. Record the question, comparison, split, metrics, results and their source.
- For a numbered note `NNN.md`, name its script `experiments/<description>_NNN.py` and its config `configs/NNN.yaml`. Link both from the note and include the command to run from the repository root.
- Reuse the `bridge_shm/` package for data loading, windowing, preprocessing, models, fitting and metrics. Training scripts must support MPS. Keep machine-specific data paths outside committed configs.
- Preserve historical notebooks and their saved outputs as experiment references. Identify scores from saved artifacts, reruns and new implementations separately.
- Get explicit approval before replacing or materially changing an existing method, comparator or evaluation protocol. Describe the existing setup, the proposed change and reason, the affected results and the limits on comparability before implementing it or running dependent experiments.
- Label results under a changed evaluation protocol as a new comparison. Verify the original protocol and reported results before claiming a reproduction.
