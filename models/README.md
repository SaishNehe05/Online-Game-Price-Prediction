# Model artifacts

Run `python src/train_model.py` after downloading the dataset to generate:

- `best_model.joblib`: fitted preprocessing + estimator pipeline
- `model_metadata.json`: UI dropdown options and measured evaluation metrics

These generated model files are not included by default because the dataset has not been downloaded/trained in this build environment. Do not present synthetic/test metrics as project results.
