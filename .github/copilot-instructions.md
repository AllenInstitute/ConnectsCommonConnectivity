# Repository instructions

- Never edit `src/connects_common_connectivity/models.py` manually. It is a generated artifact.
- Make model changes in the source LinkML schemas under `schemas/`.
- Regenerate `models.py` by running `bash scripts/generate_models.sh` from the repository root.

## Test documentation

- Every new or modified test function must have a concise one-line docstring describing the behavior or contract its assertions verify.
- State the relevant condition and expected outcome; do not merely restate the test name or leave readers to infer intent from assertion failures or error messages. Keep docstrings accurate when tests change.