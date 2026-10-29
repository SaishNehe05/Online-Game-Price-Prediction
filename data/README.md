# Dataset instructions

The raw CSV is downloaded locally, not committed to Git, because it can be reproduced from the public source and the project repository stays small.

1. From the repository root, run `python scripts/download_data.py`.
2. The dataset will be saved to `data/raw/steam-games.csv`.
3. Run `python src/train_model.py` to clean the data and train/evaluate the models.

**Source:** Carlos Díez García and Pedro Vega Navarro, *Steam Games - UOC Project*, Zenodo (published 1 April 2025). DOI: [10.5281/zenodo.15121027](https://doi.org/10.5281/zenodo.15121027). The record describes 3,537 game records gathered in March 2025 from the Spanish-language Steam web store. License: CC BY-SA 4.0; cite the data source and preserve its license/attribution requirements.
