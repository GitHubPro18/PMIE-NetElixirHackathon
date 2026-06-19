# PMIE Streamlit Dashboard

This folder contains the Streamlit dashboard for the Probabilistic Marketing Intelligence Engine.

Run it from the repository root after installing dependencies and training the local model:

```bash
streamlit run frontend/dashboard.py
```

The dashboard expects:

- Input CSV files in `data/`
- A trained model at `pickle/model.pkl`
- Optional Gemini credentials in `.env`

To rebuild the model from a fresh clone:

```bash
python src/train_revenue.py
python src/conformal.py
python src/train_roas.py
```
