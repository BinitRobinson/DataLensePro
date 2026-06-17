# 🔬 DataLens Pro — Automated Data Analysis Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Live-brightgreen" />
</p>

<p align="center">
  A professional-grade automated data analysis dashboard built with Streamlit.
  Upload any CSV and instantly get summary statistics, correlation analysis, trend detection, outlier detection, and a full EDA report — all in one app.
</p>

---

## 🚀 Live Demo

👉 **[Try it on Streamlit Cloud](https://your-app-url.streamlit.app)** *(replace with your deployed URL)*

---

## 📸 Screenshots

> Add screenshots of your deployed app here.

---

## ✨ Features

| Tab | Module | What it does |
|-----|--------|-------------|
| 🧹 | **Data Cleaning** | Handle missing values, remove duplicates, standardize column names, download clean CSV |
| 📈 | **Summary Statistics** | Descriptive stats table, histograms, box plots, skewness & kurtosis, auto-insights |
| 🔗 | **Correlation Analysis** | Pearson/Spearman/Kendall heatmap, scatter plots with trendlines, ranked pair table |
| 📉 | **Trend Detection** | Linear regression per column, R², p-value significance, rolling mean/std, 95% CI bands |
| 🚨 | **Outlier Detection** | Z-score & IQR methods, visual scatter highlighting, box plots, export flagged rows |
| 📄 | **EDA Report** | One-click full exploratory analysis with executive summary, downloadable `.txt` report |

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web app framework
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation
- **[NumPy](https://numpy.org/)** — Numerical computing
- **[Plotly](https://plotly.com/python/)** — Interactive charts
- **[SciPy](https://scipy.org/)** — Statistical analysis (linregress, zscore, t-distribution)

---

## 📂 Project Structure

```
DataLensPro/
├── app.py            ← Entire application (single file)
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/DataLensPro.git
cd DataLensPro
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

---

## 📊 Usage

1. **Upload your CSV** using the sidebar uploader, or use the built-in sample dataset
2. **Navigate tabs** to explore each analysis module
3. **Adjust settings** — correlation method, outlier threshold, rolling window, etc.
4. **Download outputs** — cleaned CSV, outlier report, or full EDA report

### CSV Requirements
- UTF-8 encoding
- First row = column headers
- At least one numeric column

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Binit** — Data Analyst Portfolio Project

[![GitHub](https://img.shields.io/badge/GitHub-your--username-181717?logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/your-profile)
