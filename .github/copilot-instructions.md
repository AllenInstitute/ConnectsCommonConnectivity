# Repository instructions

- Never edit `src/connects_common_connectivity/models.py` manually. It is a generated artifact.
- Make model changes in the source LinkML schemas under `schemas/`.
- Regenerate `models.py` by running `bash scripts/generate_models.sh` from the repository root.