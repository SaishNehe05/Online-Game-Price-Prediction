# Publish this project to GitHub

The connected GitHub actions can add files to an existing repository, but this environment does not expose an action for creating a brand-new repository. The project package is ready to publish once an empty repository exists.

## Recommended repository settings

- Repository name: `Online-Game-Price-Prediction`
- Visibility: Public if your instructor needs to open it; Private if your course requires private submission
- When creating it, do not initialise it with a README, because this package already includes one

## Upload using Git on Windows

Extract the project ZIP, open PowerShell in the extracted `Online-Game-Price-Prediction` folder, and run:

```powershell
git init -b main
git add .
git commit -m "Add online game price prediction PBL project"
git remote add origin https://github.com/YOUR_USERNAME/Online-Game-Price-Prediction.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and use the exact URL of your new repository. Git must be installed and you must be signed in to GitHub.

## Before publishing the trained artifacts

1. Install requirements: `pip install -r requirements.txt`.
2. Download data: `python scripts/download_data.py`.
3. Train and evaluate: `python src/train_model.py`.
4. Review the actual metrics in `reports/test_metrics.json` and model comparison in `reports/model_comparison.csv`.
5. Test the app: `streamlit run app.py`.
6. Commit the generated model files and evaluation reports if you want the deployed app to load the already-trained model. The raw downloaded dataset remains excluded by `.gitignore`; the README links to the source.

Never report the sample-data integration test as your project's real evaluation result. Run the workflow on the actual Zenodo dataset first.
