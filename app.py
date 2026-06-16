import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataLens Pro", page_icon="🔬", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.insight  { background:#eff6ff; border-left:4px solid #3b82f6; padding:.6rem 1rem;
            border-radius:0 8px 8px 0; margin:.35rem 0; font-size:.88rem; color:#1e40af; }
.warn     { background:#fffbeb; border-left:4px solid #f59e0b; padding:.6rem 1rem;
            border-radius:0 8px 8px 0; margin:.35rem 0; font-size:.88rem; color:#92400e; }
.ok       { background:#f0fdf4; border-left:4px solid #22c55e; padding:.6rem 1rem;
            border-radius:0 8px 8px 0; margin:.35rem 0; font-size:.88rem; color:#15803d; }
.danger   { background:#fef2f2; border-left:4px solid #ef4444; padding:.6rem 1rem;
            border-radius:0 8px 8px 0; margin:.35rem 0; font-size:.88rem; color:#991b1b; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PALETTE = ["#3b82f6","#8b5cf6","#22c55e","#f59e0b","#ef4444","#06b6d4","#ec4899"]
THEME   = dict(template="plotly_white", margin=dict(t=50,b=40,l=50,r=30),
               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

def num_cols(df):   return df.select_dtypes(include=np.number).columns.tolist()
def date_col(df):   return next((c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])), None)

def box(cls, msg):
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

def sample_data():
    np.random.seed(42)
    n = 60
    rev  = 100 + np.arange(n)*1.8 + np.random.randn(n)*10
    unit = rev*20 + np.random.randn(n)*50
    cost = rev*0.6 + np.random.randn(n)*5
    df = pd.DataFrame({
        "Date":       pd.date_range("2021-01-01", periods=n, freq="MS"),
        "Revenue_K":  rev.round(2),
        "Units_Sold": unit.round(0).astype(int),
        "Avg_Price":  (rev*1000/unit).round(2),
        "Op_Cost_K":  cost.round(2),
        "Margin_Pct": ((rev-cost)/rev*100 + np.random.randn(n)*1.5).round(2),
    })
    df.loc[[5,20,45], "Revenue_K"]  = np.nan   # missing
    df.loc[[10,35],   "Units_Sold"] = -999      # bad values
    df.loc[55,        "Avg_Price"]  = 9999      # outlier
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate

# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 DataLens Pro")
    st.markdown("---")
    uploaded   = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample dataset", value=(uploaded is None))
    st.markdown("---")
    st.caption("Streamlit · Plotly · SciPy")

if uploaded and not use_sample:
    df_raw = pd.read_csv(uploaded)
    for c in df_raw.columns:
        if "date" in c.lower():
            try: df_raw[c] = pd.to_datetime(df_raw[c])
            except: pass
    st.sidebar.success(f"✅ {df_raw.shape[0]:,} rows · {df_raw.shape[1]} cols")
else:
    df_raw = sample_data()
    st.sidebar.info("ℹ️ Sample dataset")

nc = num_cols(df_raw)
dc = date_col(df_raw)
x_vals = df_raw[dc] if dc else df_raw.index

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔬 DataLens Pro — Automated Data Analysis Engine")
st.caption(f"{df_raw.shape[0]:,} rows · {df_raw.shape[1]} columns · {len(nc)} numeric")

with st.expander("Preview data"):
    st.dataframe(df_raw.head(30), use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🧹 Data Cleaning",
    "📈 Summary Stats",
    "🔗 Correlation",
    "📉 Trend Detection",
    "🚨 Outlier Detection",
    "📄 EDA Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("🧹 Data Cleaning")

    miss  = df_raw.isnull().sum()
    dupes = int(df_raw.duplicated().sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",           f"{df_raw.shape[0]:,}")
    c2.metric("Missing cells",  int(miss.sum()))
    c3.metric("Duplicate rows", dupes)
    c4.metric("Numeric cols",   len(nc))

    st.markdown("**Column quality**")
    quality = pd.DataFrame({
        "Column":   df_raw.columns,
        "Dtype":    df_raw.dtypes.astype(str).values,
        "Missing":  miss.values,
        "Missing%": (miss/len(df_raw)*100).round(1).values,
        "Unique":   df_raw.nunique().values,
    }).reset_index(drop=True)
    st.dataframe(quality, use_container_width=True)

    st.markdown("---")
    st.markdown("**Apply cleaning**")
    ca, cb = st.columns(2)
    with ca:
        miss_strat  = st.selectbox("Missing values", [
            "Leave as-is", "Drop rows", "Fill mean", "Fill median", "Forward fill"])
        drop_dupes  = st.checkbox("Remove duplicates", value=True)
    with cb:
        std_cols    = st.checkbox("Standardize column names", value=True)
        neg_cols    = st.multiselect("Remove rows with negatives in:", nc)

    if st.button("Apply cleaning", type="primary"):
        df_c = df_raw.copy()
        if std_cols:
            df_c.columns = [c.strip().lower().replace(" ","_") for c in df_c.columns]
        if miss_strat == "Drop rows":         df_c = df_c.dropna()
        elif miss_strat == "Fill mean":
            for c in num_cols(df_c):          df_c[c] = df_c[c].fillna(df_c[c].mean())
        elif miss_strat == "Fill median":
            for c in num_cols(df_c):          df_c[c] = df_c[c].fillna(df_c[c].median())
        elif miss_strat == "Forward fill":    df_c = df_c.ffill()
        if drop_dupes:                         df_c = df_c.drop_duplicates()
        for c in neg_cols:
            if c in df_c.columns:             df_c = df_c[df_c[c] >= 0]

        st.session_state["df_clean"] = df_c
        st.success(f"Done — {df_c.shape[0]:,} rows remain (removed {df_raw.shape[0]-df_c.shape[0]})")

    if "df_clean" in st.session_state:
        df_c = st.session_state["df_clean"]
        r1,r2,r3 = st.columns(3)
        r1.metric("Rows after",    f"{df_c.shape[0]:,}")
        r2.metric("Missing after", int(df_c.isnull().sum().sum()))
        r3.metric("Dupes after",   int(df_c.duplicated().sum()))
        with st.expander("Preview cleaned data"):
            st.dataframe(df_c.head(30), use_container_width=True)
        st.download_button("⬇️ Download cleaned CSV",
                           df_c.to_csv(index=False).encode(),
                           "cleaned_data.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Insights**")
    for col in df_raw.columns:
        pct = miss[col]/len(df_raw)*100
        if pct > 20: box("danger", f"<b>{col}</b>: {pct:.1f}% missing — consider dropping.")
        elif pct > 5: box("warn",  f"<b>{col}</b>: {pct:.1f}% missing — impute before analysis.")
        elif pct > 0: box("ok",    f"<b>{col}</b>: {pct:.1f}% missing — minor issue.")
    if dupes > 0: box("warn", f"{dupes} duplicate rows found.")
    else:         box("ok",   "No duplicate rows.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("📈 Summary Statistics")

    if not nc:
        st.error("No numeric columns."); st.stop()

    desc = df_raw[nc].describe(percentiles=[.05,.25,.5,.75,.95]).T.round(3)
    desc["skewness"] = df_raw[nc].skew().round(3)
    desc["kurtosis"] = df_raw[nc].kurtosis().round(3)
    desc["cv%"]      = (desc["std"]/desc["mean"].abs()*100).round(2)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",          f"{df_raw.shape[0]:,}")
    c2.metric("Numeric cols",  len(nc))
    c3.metric("Missing",       int(df_raw[nc].isnull().sum().sum()))
    c4.metric("Skewed (>1)",   sum(abs(df_raw[c].skew())>1 for c in nc))

    st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("**Distributions**")
    cols_row = st.selectbox("Columns per row", [1,2,3], index=1, key="dist_cols")
    groups = [nc[i:i+cols_row] for i in range(0, len(nc), cols_row)]
    for grp in groups:
        row = st.columns(cols_row)
        for j, col in enumerate(grp):
            with row[j]:
                data = df_raw[col].dropna()
                fig  = make_subplots(1,2, column_widths=[.65,.35],
                                     subplot_titles=["Histogram","Box"])
                fig.add_trace(go.Histogram(x=data, marker_color=PALETTE[0],
                                           opacity=.75, showlegend=False), 1,1)
                fig.add_trace(go.Box(y=data, marker_color=PALETTE[0],
                                     boxpoints="outliers", showlegend=False,
                                     marker=dict(size=4,outliercolor=PALETTE[4])), 1,2)
                fig.update_layout(**THEME, title_text=col, title_font_size=13, height=270)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Insights**")
    for col in nc:
        sk  = df_raw[col].skew()
        mis = df_raw[col].isnull().mean()*100
        cv  = df_raw[col].std()/df_raw[col].mean()*100 if df_raw[col].mean()!=0 else 0
        if mis > 10:        box("danger", f"<b>{col}</b>: {mis:.1f}% missing.")
        if abs(sk) > 1:     box("warn",   f"<b>{col}</b>: skew={sk:.2f} — non-normal, consider transform.")
        if abs(cv) > 50:    box("insight",f"<b>{col}</b>: high variability (CV={cv:.1f}%).")
        if abs(sk)<=.5 and mis<5: box("ok", f"<b>{col}</b>: clean distribution (skew={sk:.2f}).")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🔗 Correlation Analysis")

    if len(nc) < 2:
        st.warning("Need ≥ 2 numeric columns."); st.stop()

    c1,c2 = st.columns(2)
    method  = c1.selectbox("Method", ["pearson","spearman","kendall"])
    min_r   = c2.slider("Highlight |r| ≥", 0.0, 1.0, 0.5, 0.05)

    corr_mat = df_raw[nc].corr(method=method).round(3)

    fig = px.imshow(corr_mat, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"{method.title()} Correlation Matrix", aspect="auto")
    fig.update_layout(**THEME, height=460)
    st.plotly_chart(fig, use_container_width=True)

    # Ranked pairs
    pairs = []
    for i in range(len(nc)):
        for j in range(i+1, len(nc)):
            v = corr_mat.iloc[i,j]
            pairs.append({"Column A":nc[i],"Column B":nc[j],"r":v,"|r|":abs(v),
                          "Strength":"Strong" if abs(v)>=.7 else ("Moderate" if abs(v)>=.4 else "Weak")})
    pairs_df = pd.DataFrame(pairs).sort_values("|r|",ascending=False)
    st.dataframe(pairs_df.drop(columns=["|r|"]).reset_index(drop=True)
                         .style.format({"r":"{:.3f}"}), use_container_width=True)

    # Scatter plots for strong pairs
    strong = pairs_df[pairs_df["|r|"] >= min_r]
    if not strong.empty:
        st.markdown(f"**Scatter plots (|r| ≥ {min_r})**")
        grps = [strong.iloc[i:i+2] for i in range(0, len(strong), 2)]
        for grp in grps:
            row = st.columns(min(2, len(grp)))
            for j,(_,p) in enumerate(grp.iterrows()):
                with row[j]:
                    xa = df_raw[p["Column A"]].dropna()
                    ya = df_raw[p["Column B"]].reindex(xa.index).dropna()
                    xa = xa.reindex(ya.index)
                    m, b = np.polyfit(xa, ya, 1)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=xa, y=ya, mode="markers",
                                              marker=dict(color=PALETTE[0], size=6, opacity=.6),
                                              name="Data"))
                    x_line = np.linspace(xa.min(), xa.max(), 100)
                    fig2.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode="lines",
                                              line=dict(color=PALETTE[4], width=2, dash="dash"),
                                              name="Trend"))
                    fig2.update_layout(**THEME, height=300,
                                       title=f'{p["Column A"]} vs {p["Column B"]}  r={p["r"]:.2f}')
                    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Insights**")
    for _,p in pairs_df[pairs_df["|r|"]>=.5].iterrows():
        r = p["r"]
        if abs(r)>=.8:   box("warn",    f'<b>{p["Column A"]} ↔ {p["Column B"]}</b>: very strong (r={r:.2f}) — possible multicollinearity.')
        elif abs(r)>=.6: box("insight", f'<b>{p["Column A"]} ↔ {p["Column B"]}</b>: strong correlation (r={r:.2f}).')
        else:            box("ok",      f'<b>{p["Column A"]} ↔ {p["Column B"]}</b>: moderate (r={r:.2f}).')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TREND DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("📉 Trend Detection")

    c1,c2,c3 = st.columns(3)
    selected = c1.multiselect("Columns", nc, default=nc[:4])
    show_ci  = c2.checkbox("95% CI bands", value=True)
    roll_win = c3.slider("Rolling window", 2, min(30, max(3,len(df_raw)//5)), 5)

    if not selected:
        st.info("Select at least one column."); st.stop()

    # Trend table
    rows = []
    for col in selected:
        s = df_raw[col].dropna()
        slope,intercept,r,p,_ = stats.linregress(np.arange(len(s)), s.values)
        r2  = r**2
        pct = slope*len(s)/s.mean()*100 if s.mean()!=0 else 0
        rows.append({"Column":col,
                     "Direction":"↑ Up" if slope>0 else "↓ Down",
                     "Change%":round(pct,1), "R²":round(r2,4),
                     "p-value":round(p,4),
                     "Significant":"✅ Yes" if p<.05 else "⚠️ No"})
    tr_df = pd.DataFrame(rows)
    st.dataframe(tr_df, use_container_width=True)

    # Individual charts
    for col in selected:
        s = df_raw[col].dropna()
        x = np.arange(len(s))
        slope,intercept,r,p,_ = stats.linregress(x, s.values)
        trend = slope*x + intercept
        pct   = slope*len(s)/s.mean()*100 if s.mean()!=0 else 0
        r2    = r**2

        label = f"{'↑' if slope>0 else '↓'} {pct:+.1f}%  |  R²={r2:.3f}  |  {'p<0.05 ✅' if p<.05 else f'p={p:.3f} ⚠️'}"
        with st.expander(f"**{col}** — {label}", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals[:len(s)], y=s.values,
                                     name=col, line=dict(color=PALETTE[0],width=1.5),
                                     marker=dict(size=4), mode="lines+markers", opacity=.8))
            fig.add_trace(go.Scatter(x=x_vals[:len(s)], y=trend,
                                     name="Trend", line=dict(color=PALETTE[4],width=2.5,dash="dash")))
            if show_ci:
                n  = len(s)
                ss = np.sqrt(np.sum((s.values-trend)**2)/(n-2))
                se = ss*np.sqrt(1/n+(x-x.mean())**2/np.sum((x-x.mean())**2))
                t  = stats.t.ppf(.975, df=n-2)
                up, lo = trend+t*se, trend-t*se
                fig.add_trace(go.Scatter(
                    x=list(x_vals[:n])+list(x_vals[:n])[::-1],
                    y=list(up)+list(lo)[::-1],
                    fill="toself", fillcolor="rgba(239,68,68,0.1)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
            fig.update_layout(**THEME, height=300, legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig, use_container_width=True)

    # Rolling stats
    st.markdown("---")
    st.markdown("**Rolling statistics**")
    roll_col = st.selectbox("Column", selected, key="rc")
    fig_r = go.Figure()
    rd = df_raw[roll_col]
    fig_r.add_trace(go.Scatter(x=x_vals, y=rd, name="Raw",
                                opacity=.35, line=dict(color="#94a3b8",width=1)))
    fig_r.add_trace(go.Scatter(x=x_vals, y=rd.rolling(roll_win,center=True).mean(),
                                name=f"Mean({roll_win})", line=dict(color=PALETTE[0],width=2.5)))
    fig_r.add_trace(go.Scatter(x=x_vals, y=rd.rolling(roll_win,center=True).std(),
                                name=f"Std({roll_win})", line=dict(color=PALETTE[3],width=1.8,dash="dot"),
                                yaxis="y2"))
    fig_r.update_layout(**THEME, height=340,
                         yaxis2=dict(overlaying="y",side="right",showgrid=False),
                         legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("**Insights**")
    for r in rows:
        pct = r["Change%"]
        if "Yes" in r["Significant"] and abs(pct)>20:
            box("warn",    f'<b>{r["Column"]}</b>: strong significant trend ({pct:+.1f}%).')
        elif "Yes" in r["Significant"]:
            box("ok",      f'<b>{r["Column"]}</b>: significant trend ({pct:+.1f}%, R²={r["R²"]}).')
        else:
            box("insight", f'<b>{r["Column"]}</b>: no significant trend (p={r["p-value"]}).')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("🚨 Outlier Detection")

    c1,c2,c3 = st.columns(3)
    out_method = c1.radio("Method", ["Z-Score","IQR","Both"], horizontal=True)
    z_thresh   = c2.slider("Z threshold", 1.5, 4.0, 2.5, 0.1)
    out_cols   = c3.multiselect("Columns", nc, default=nc)

    if not out_cols: st.info("Select columns."); st.stop()

    # Z-score outliers
    z_rows = []
    for col in out_cols:
        s = df_raw[col].dropna()
        zs = np.abs(stats.zscore(s))
        for li,(oi,v) in enumerate(s.items()):
            if zs[li] > z_thresh:
                z_rows.append({"Column":col,"Row":oi,"Value":round(v,4),"Z-Score":round(zs[li],3)})
    z_out = pd.DataFrame(z_rows)

    # IQR outliers
    iq_rows = []
    for col in out_cols:
        q1,q3 = df_raw[col].quantile(.25), df_raw[col].quantile(.75)
        iqr = q3-q1; lo,hi = q1-1.5*iqr, q3+1.5*iqr
        mask = (df_raw[col]<lo)|(df_raw[col]>hi)
        for i in df_raw[mask].index:
            iq_rows.append({"Column":col,"Row":i,"Value":round(df_raw[col][i],4),
                            "Lower":round(lo,3),"Upper":round(hi,3)})
    iq_out = pd.DataFrame(iq_rows)

    c1,c2,c3 = st.columns(3)
    c1.metric("Z-Score outliers",  len(z_out))
    c2.metric("IQR outliers",      len(iq_out))
    c3.metric("Columns affected",  z_out["Column"].nunique() if not z_out.empty else 0)

    # Box plots
    st.markdown("**Box plots**")
    grps = [out_cols[i:i+3] for i in range(0,len(out_cols),3)]
    for grp in grps:
        row = st.columns(len(grp))
        for j,col in enumerate(grp):
            with row[j]:
                fig = go.Figure(go.Box(y=df_raw[col], name=col,
                                       boxpoints="outliers",
                                       marker=dict(color=PALETTE[0],outliercolor=PALETTE[4],size=6),
                                       fillcolor="rgba(59,130,246,.15)"))
                fig.update_layout(**THEME, title=col, title_font_size=13,
                                  height=270, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    if out_method in ["Z-Score","Both"] and not z_out.empty:
        st.markdown(f"**Z-Score flagged rows (|z| > {z_thresh})**")
        st.dataframe(z_out.reset_index(drop=True), use_container_width=True)
        for col in z_out["Column"].unique():
            co = z_out[z_out["Column"]==col]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[col], mode="markers",
                                     name="Normal", marker=dict(color=PALETTE[0],size=5,opacity=.6)))
            fig.add_trace(go.Scatter(x=co["Row"], y=co["Value"], mode="markers",
                                     name="Outlier",
                                     marker=dict(color=PALETTE[4],size=11,symbol="x-open",line_width=2)))
            fig.update_layout(**THEME, title=f"{col} — outliers highlighted", height=290)
            st.plotly_chart(fig, use_container_width=True)

    if out_method in ["IQR","Both"] and not iq_out.empty:
        st.markdown("**IQR flagged rows**")
        st.dataframe(iq_out.reset_index(drop=True), use_container_width=True)

    if not z_out.empty:
        st.download_button("⬇️ Download outlier report",
                           z_out.to_csv(index=False).encode(),
                           "outliers.csv","text/csv")

    st.markdown("**Insights**")
    for col in out_cols:
        n_z = len(z_out[z_out["Column"]==col]) if not z_out.empty else 0
        pct = n_z/len(df_raw)*100
        if pct>5:   box("danger",  f"<b>{col}</b>: {n_z} outliers ({pct:.1f}%) — check data quality.")
        elif n_z>0: box("warn",    f"<b>{col}</b>: {n_z} outlier(s) detected — review flagged rows.")
        else:       box("ok",      f"<b>{col}</b>: no outliers found.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EDA REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with t6:
    st.subheader("📄 EDA Report Generator")

    c1,c2 = st.columns(2)
    analyst = c1.text_input("Analyst name", "Data Analyst")
    project = c2.text_input("Project name", "Business Dataset")

    if not st.button("🚀 Generate Report", type="primary"):
        st.info("Click **Generate Report** to run all analyses and produce a summary.")
        st.stop()

    from datetime import datetime
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    # Run analyses
    desc2   = df_raw[nc].describe(percentiles=[.05,.25,.5,.75,.95]).T.round(3)
    desc2["skewness"] = df_raw[nc].skew().round(3)
    corr2   = df_raw[nc].corr().round(3)
    z2_rows = []
    for col in nc:
        s  = df_raw[col].dropna()
        zs = np.abs(stats.zscore(s))
        for li,(oi,v) in enumerate(s.items()):
            if zs[li]>2.5: z2_rows.append({"Column":col,"Row":oi,"Value":round(v,4),"Z":round(zs[li],3)})
    z2_out = pd.DataFrame(z2_rows)

    tr2 = []
    for col in nc:
        s = df_raw[col].dropna()
        slope,_,r,p,_ = stats.linregress(np.arange(len(s)),s.values)
        pct = slope*len(s)/s.mean()*100 if s.mean()!=0 else 0
        tr2.append({"Column":col,"Direction":"↑" if slope>0 else "↓",
                    "Change%":round(pct,1),"R²":round(r**2,4),"Significant":"Yes" if p<.05 else "No"})
    tr2_df = pd.DataFrame(tr2)

    # Header
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);color:white;
                border-radius:12px;padding:2rem;margin-bottom:1.5rem;">
      <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;">
        Exploratory Data Analysis Report
      </div>
      <h2 style="margin:.25rem 0 .3rem;font-size:1.8rem;">{project}</h2>
      <div style="color:#94a3b8;font-size:.85rem;">By <b style="color:#e2e8f0;">{analyst}</b> · {now}</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Overview
    st.markdown("### 1. Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows",          f"{df_raw.shape[0]:,}")
    c2.metric("Columns",       df_raw.shape[1])
    c3.metric("Numeric cols",  len(nc))
    c4.metric("Missing cells", int(df_raw.isnull().sum().sum()))

    # 2. Summary stats
    st.markdown("### 2. Summary Statistics")
    st.dataframe(desc2.style.format("{:.3f}"), use_container_width=True)

    # 3. Correlation
    st.markdown("### 3. Correlation Matrix")
    fig = px.imshow(corr2,text_auto=".2f",color_continuous_scale="RdBu_r",
                    zmin=-1,zmax=1,aspect="auto")
    fig.update_layout(**THEME, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Trends
    st.markdown("### 4. Trend Detection")
    st.dataframe(tr2_df, use_container_width=True)

    # 5. Outliers
    st.markdown("### 5. Outlier Detection")
    c1,c2 = st.columns(2)
    c1.metric("Z-Score outliers (|z|>2.5)", len(z2_out))
    c2.metric("Columns affected", z2_out["Column"].nunique() if not z2_out.empty else 0)
    if not z2_out.empty:
        st.dataframe(z2_out.reset_index(drop=True), use_container_width=True)

    # 6. Executive summary
    st.markdown("### 6. Executive Summary")
    sig_trends = tr2_df[tr2_df["Significant"]=="Yes"]["Column"].tolist()
    strong_corr = [(nc[i],nc[j]) for i in range(len(nc)) for j in range(i+1,len(nc))
                   if abs(corr2.iloc[i,j])>=.7]
    lines = [
        f"Dataset: **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns** ({len(nc)} numeric).",
        f"Missing data: **{int(df_raw.isnull().sum().sum())} cells** ({df_raw.isnull().mean().mean()*100:.1f}% of data).",
        f"Skewed columns (>1): **{sum(abs(df_raw[c].skew())>1 for c in nc)}**.",
        f"Outliers detected: **{len(z2_out)}** (Z-score > 2.5).",
        f"Significant trends in: **{', '.join(sig_trends) if sig_trends else 'none'}**.",
        f"Strong correlations (|r|≥0.7): **{len(strong_corr)} pairs**.",
    ]
    for l in lines: st.markdown(f"- {l}")

    # Download
    report = f"""EDA REPORT — {project}
Analyst: {analyst}  |  Date: {now}
{'='*60}

1. DATASET OVERVIEW
Rows: {df_raw.shape[0]:,}  |  Columns: {df_raw.shape[1]}  |  Numeric: {len(nc)}
Missing: {int(df_raw.isnull().sum().sum())}  |  Duplicates: {int(df_raw.duplicated().sum())}

2. SUMMARY STATISTICS
{desc2[['mean','std','min','50%','max','skewness']].to_string()}

3. TREND DETECTION
{tr2_df.to_string(index=False)}

4. OUTLIERS (Z-Score > 2.5)
{z2_out.to_string(index=False) if not z2_out.empty else 'None found.'}

5. EXECUTIVE SUMMARY
""" + "\n".join(f"- {l}" for l in lines)

    st.download_button("⬇️ Download EDA Report (.txt)", report.encode(),
                       f"eda_{project.lower().replace(' ','_')}.txt", "text/plain")
