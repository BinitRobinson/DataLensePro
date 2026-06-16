# 🔬 DataLens Pro — Automated Data Analysis Engine

Single-file Streamlit dashboard with 6 analysis modules.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

## Modules (tabs)

| Tab | Module |
|-----|--------|
| 🧹 | Data Cleaning — missing values, duplicates, export clean CSV |
| 📈 | Summary Statistics — descriptive stats, distributions, insights |
| 🔗 | Correlation Analysis — heatmap, scatter plots, ranked pairs |
| 📉 | Trend Detection — regression, rolling stats, CI bands |
| 🚨 | Outlier Detection — Z-score + IQR, visual flagging, export |
| 📄 | EDA Report — full report + downloadable .txt |

## Files

```
datalens_simple/
├── app.py            ← entire application (single file)
└── requirements.txt
```

## Usage

- Upload any CSV via the sidebar, or use the built-in sample dataset
- All 6 tabs update automatically
