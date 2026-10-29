# Project workflow

1. Keep dataset attribution and license information with any reused or redistributed data.
2. Do not commit local environments, secrets, raw download files, or generated large binary artifacts unless the team explicitly decides to version them.
3. Run `python -m unittest discover -s tests -v` before committing changes.
4. Re-run training after changes to data cleaning, features, preprocessing, or models; record actual CV and test metrics.
5. Never select the final model by looking repeatedly at held-out test performance.
