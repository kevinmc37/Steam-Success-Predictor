#  Steam Success Predictor

**Analyzing the impact of studio type (AAA, AA, Indie) on Steam market performance**

 *[Leer en español](README.md)*

Final project — **Samsung Innovation Campus** AI course, Group 2

**Authors:** Kevin Morales Cano · Adrián Contreras González

---

##  Overview

A machine learning system that predicts the commercial performance of Steam games, classifying them into three success tiers based on price, platform, language support, release timing, and — most notably — the historical success record of the developer and publisher.

The project combines a full data engineering pipeline (cleaning, entity normalization, Bayesian smoothing) with the training and comparison of several supervised classification algorithms.

##  Target variable

`exito_target` is defined as a **multiclass classification** problem based on `Estimated owners`:

| Class | Description | Threshold | % of dataset |
|---|---|---|---|
| 0 | Commercial failure | < 20,000 owners | ~80% |
| 1 | Moderate success | 20,000 – 200,000 owners | ~16% |
| 2 | Major hit | > 200,000 owners | ~4% |

##  Repository structure

```
├── 01_data_cleaning_and_features.ipynb   # Cleaning, feature engineering, Bayesian smoothing
├── 02_model_training.ipynb               # Model training, evaluation, and prediction
├── X_entrenamiento_limpio.csv            # Processed training features
├── y_entrenamiento_limpio.csv            # Training target
├── X_futuro_limpio.csv                   # Upcoming releases features
├── nombres_proximos_juegos.csv           # Upcoming release names (for traceability)
├── modelo_lightgbm_steam_success.pkl     # Final trained model
├── scaler_steam_success.pkl              # StandardScaler fitted on training data
└── README.md
```

>  **About `games.csv`:** the raw source dataset (~500 MB, sourced from Kaggle) is not included in this repository, as it exceeds GitHub's file size limit. `01_data_cleaning_and_features.ipynb` requires this file to run from scratch — if you need to reproduce the full pipeline, request it from the authors or download it from the original Kaggle source. It is not needed to work directly with the models (`02_model_training.ipynb`): the already-processed datasets (`X_entrenamiento_limpio.csv`, `y_entrenamiento_limpio.csv`, `X_futuro_limpio.csv`, `nombres_proximos_juegos.csv`) are included.

##  Data pipeline

```
[Raw Steam data]
        ↓
[Text cleaning & Developer/Publisher normalization]
        ↓
[Feature engineering (Bayesian smoothing)]
        ↓
[df_bueno (historical)]     [df_proximos (upcoming releases)]
        ↓                           ↓
        └───────────┬───────────────┘
                     ↓
        [Standardization & scaling (StandardScaler)]
                     ↓
        [Predictive modeling & validation]
```

**Key preprocessing steps:**
- Repaired a structural misalignment in `games.csv` (a corrupted header causing a column shift starting at `Price`).
- Normalized developer/publisher names via text cleaning, fuzzy matching (`SequenceMatcher`, 0.82 threshold), and manual brand consolidation — reducing 72,859 to 70,723 unique entities.
- **Bayesian Target Encoding** for `dev_exito_promedio` / `pub_exito_promedio`, preventing overfitting for studios with little history (`m = 5`).
- Extracted ~1,000 upcoming releases via the **Steam Web API**, manually adding ~34 high-profile titles not correctly indexed (e.g. *Grand Theft Auto VI*).
- Strict data leakage prevention: the scaler is fit only on the training set, and no historical statistic leaks into the prospective set.

##  Modeling

Four multiclass classification algorithms were trained and compared:

| Model | Accuracy | F1 macro | ROC-AUC (ovr, macro) |
|---|---|---|---|
| **LightGBM**  | 0.911 | **0.838** | **0.981** |
| XGBoost | 0.906 | 0.832 | 0.979 |
| Random Forest | 0.903 | 0.825 | 0.976 |
| Logistic Regression | 0.865 | 0.755 | 0.940 |

**Selected model: LightGBM**, for its best balance across the three classes and its performance on Class 2 (Major hit: precision 0.84, recall 0.74).

**Top influential features:** `Price`, `dev_exito_promedio`, `año_lanzamiento`, `pub_exito_promedio`, `mes_lanzamiento` — consistent with the project's core hypothesis that price and the developer's success history are the dominant predictors of commercial performance.

##  Known limitations

- **Feature collision in the prospective set:** ~44% of upcoming releases share an identical feature vector (studios with no history that also match on price, platforms, and release date), receiving the same predicted probability despite being different games.
- The model does not incorporate early-interest signals (wishlists, followers) or fine-grained genre information — flagged as future work.

##  Interface

`02_model_training.ipynb` includes a lightweight interactive interface built with `ipywidgets`, allowing a user to enter a hypothetical game's characteristics and get a real-time success prediction without touching the code.

##  How to run

```bash
conda create -n steam-ml python=3.11 -y
conda activate steam-ml
conda install pandas numpy matplotlib seaborn scikit-learn jupyter ipywidgets -y
pip install xgboost lightgbm requests

jupyter notebook
```

1. Run `01_data_cleaning_and_features.ipynb` end to end to regenerate the clean datasets (or use the CSVs already included in the repo).
2. Run `02_model_training.ipynb` to train the models, view the metrics, and generate predictions.

>  **Note on the CSV files:** avoid opening or re-uploading them through Google Drive/Sheets — it can corrupt long-decimal numeric columns (converting them into thousands-separated text). If you need to share them via Drive, do so inside a `.zip`.

##  Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `ipywidgets` · `Steam Web API`

##  Future improvements

- Periodic web scraping to keep the dataset updated in real time.
- Full web interface (Streamlit/Flask) beyond the notebook.
- Feature enrichment to reduce identical-vector collisions.
- Hyperparameter tuning (`GridSearchCV`/`Optuna`) focused on improving Class 2 recall.

##  Authors

| | Role |
|---|---|
| **Kevin Morales Cano** | Data acquisition, cleaning, entity normalization, and feature engineering |
| **Adrián Contreras González** | Algorithm development, modeling, validation, and data leakage prevention |

Developed as part of the **Samsung Innovation Campus — AI Course**, co-financed by the European Union, the European Social Fund Plus, the Escuela de Organización Industrial (EOI), and the Junta de Andalucía.
