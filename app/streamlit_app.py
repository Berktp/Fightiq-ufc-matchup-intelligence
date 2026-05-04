import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FightIQ",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.15;
    }

    .hero p {
        margin-top: 0.45rem;
        margin-bottom: 0;
        font-size: 0.98rem;
        opacity: 0.92;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-top: 1.1rem;
        margin-bottom: 0.7rem;
    }

    .fighter-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 0.9rem;
    }

    .fighter-name {
        font-size: 1.22rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .muted {
        color: #6b7280;
        font-size: 0.93rem;
    }

    .good-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 0.8rem 0.95rem;
        border-radius: 10px;
        margin-bottom: 0.55rem;
    }

    .risk-box {
        background: #fff7ed;
        border-left: 5px solid #ea580c;
        padding: 0.8rem 0.95rem;
        border-radius: 10px;
        margin-bottom: 0.55rem;
    }

    .info-box {
        background: #f8fafc;
        border-left: 5px solid #334155;
        padding: 0.8rem 0.95rem;
        border-radius: 10px;
        margin-bottom: 0.55rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.9rem 0.95rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        height: 100%;
    }

    .metric-name {
        font-weight: 700;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }

    .metric-desc {
        color: #6b7280;
        font-size: 0.84rem;
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    .advantage-pill {
        display: inline-block;
        padding: 0.24rem 0.55rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.8rem;
        margin-top: 0.45rem;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }

    .small-note {
        font-size: 0.88rem;
        color: #6b7280;
        line-height: 1.45;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 0.9rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "best_model_metadata.json"
FIGHTERS_PATH = ROOT / "data" / "interim" / "clean_fighters.csv"
MODEL_BASE_PATH = ROOT / "data" / "processed" / "model_base_table_clean_names.csv"
MODEL_COMPARISON_PATH = ROOT / "reports" / "modeling" / "model_comparison.csv"
PERM_IMPORTANCE_PATH = ROOT / "reports" / "modeling" / "best_model_permutation_importance.csv"


# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_fighters():
    df = pd.read_csv(FIGHTERS_PATH, low_memory=False)
    if "Full_Name" in df.columns:
        df["Full_Name"] = df["Full_Name"].astype(str).str.strip()
    return df

@st.cache_data
def load_model_base():
    if MODEL_BASE_PATH.exists():
        df = pd.read_csv(MODEL_BASE_PATH, low_memory=False)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    return None

@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None

@st.cache_data
def load_perm_importance():
    if PERM_IMPORTANCE_PATH.exists():
        return pd.read_csv(PERM_IMPORTANCE_PATH)
    return None


# ============================================================
# FILE CHECK
# ============================================================
required = [MODEL_PATH, META_PATH, FIGHTERS_PATH]
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.error("Missing required files:")
    for p in missing:
        st.write("-", p)
    st.stop()

model = load_model()
metadata = load_metadata()
fighters = load_fighters()
model_base = load_model_base()
comparison_df = load_model_comparison()
perm_df = load_perm_importance()


# ============================================================
# HELPERS
# ============================================================
def safe_num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def display_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return str(x)

def display_prob(x):
    if pd.isna(x):
        return "N/A"
    return f"{100 * float(x):.1f}%"

def get_fighter_row(name):
    row = fighters[fighters["Full_Name"] == name]
    if row.empty:
        return None
    return row.iloc[0]

def compute_missing_flag(value):
    return int(pd.isna(value))

def infer_weight_class(f1, f2):
    wc1 = str(f1.get("Weight_Class_Profile", "")).strip()
    wc2 = str(f2.get("Weight_Class_Profile", "")).strip()

    if wc1 and wc1 != "nan" and wc2 and wc2 != "nan":
        if wc1 == wc2:
            return wc1
        return wc1
    if wc1 and wc1 != "nan":
        return wc1
    if wc2 and wc2 != "nan":
        return wc2
    return "Unknown"

def build_matchup_row(fighter_a_name, fighter_b_name, time_format, fight_year):
    f1 = get_fighter_row(fighter_a_name)
    f2 = get_fighter_row(fighter_b_name)

    if f1 is None or f2 is None:
        return None, None, None, None

    inferred_weight_class = infer_weight_class(f1, f2)
    features = metadata["features"]
    row = {}

    if "Weight_Class" in features:
        row["Weight_Class"] = inferred_weight_class
    if "Time Format" in features:
        row["Time Format"] = time_format
    if "fight_year" in features:
        row["fight_year"] = int(fight_year)

    for feat in features:
        if feat.startswith("f1_") and feat.endswith("_missing"):
            base = feat.replace("f1_", "").replace("_missing", "")
            row[feat] = compute_missing_flag(f1.get(base, np.nan))

        elif feat.startswith("f2_") and feat.endswith("_missing"):
            base = feat.replace("f2_", "").replace("_missing", "")
            row[feat] = compute_missing_flag(f2.get(base, np.nan))

        elif feat.startswith("f1_") and not feat.endswith("_missing"):
            base = feat.replace("f1_", "")
            if feat not in row:
                row[feat] = f1.get(base, np.nan)

        elif feat.startswith("f2_") and not feat.endswith("_missing"):
            base = feat.replace("f2_", "")
            if feat not in row:
                row[feat] = f2.get(base, np.nan)

    diff_map = {
        "height_diff": ("Height", "Height"),
        "weight_diff": ("Weight", "Weight"),
        "reach_diff": ("Reach", "Reach"),
        "wins_diff": ("W", "W"),
        "losses_diff": ("L", "L"),
        "draws_diff": ("D", "D"),
        "avg_fight_time_diff": ("Avg_Fight_Time", "Avg_Fight_Time"),
        "kd_diff": ("KD", "KD"),
        "str_diff": ("STR", "STR"),
        "td_diff": ("TD", "TD"),
        "sub_diff": ("SUB", "SUB"),
        "ctrl_diff": ("Ctrl", "Ctrl"),
        "sig_str_pct_diff": ("Sig_Str_Pct", "Sig_Str_Pct"),
        "sub_att_diff": ("Sub_Att", "Sub_Att"),
        "ko_rate_diff": ("KO_Rate", "KO_Rate"),
        "sub_rate_diff": ("SUB_Rate", "SUB_Rate"),
        "dec_rate_diff": ("DEC_Rate", "DEC_Rate"),
    }

    for feat, (a_col, b_col) in diff_map.items():
        if feat in features:
            row[feat] = safe_num(f1.get(a_col, np.nan)) - safe_num(f2.get(b_col, np.nan))

    row_df = pd.DataFrame([row])

    for feat in features:
        if feat not in row_df.columns:
            if feat == "fight_year":
                row_df[feat] = int(fight_year)
            else:
                row_df[feat] = np.nan

    row_df = row_df[features]
    return row_df, f1, f2, inferred_weight_class

def fighter_card_html(name, fighter_row):
    stance = fighter_row.get("Stance", "Unknown")
    style = fighter_row.get("Fighting_Style", "Unknown")
    wc = fighter_row.get("Weight_Class_Profile", "Unknown")

    return f"""
    <div class="fighter-card">
        <div class="fighter-name">{name}</div>
        <div class="muted">{wc} • {stance} • {style}</div>
    </div>
    """

def build_advantages_and_risks(f1_name, f2_name, f1, f2):
    advantages = []
    risks = []

    if safe_num(f1.get("Reach", np.nan)) > safe_num(f2.get("Reach", np.nan)):
        advantages.append(f"{f1_name} has a reach advantage.")
    if safe_num(f1.get("TD", np.nan)) > safe_num(f2.get("TD", np.nan)):
        advantages.append(f"{f1_name} has the stronger takedown profile.")
    if safe_num(f1.get("Ctrl", np.nan)) > safe_num(f2.get("Ctrl", np.nan)):
        advantages.append(f"{f1_name} shows stronger control metrics.")
    if safe_num(f1.get("STR", np.nan)) > safe_num(f2.get("STR", np.nan)):
        advantages.append(f"{f1_name} shows higher striking output.")
    if safe_num(f1.get("W", np.nan)) > safe_num(f2.get("W", np.nan)):
        advantages.append(f"{f1_name} appears more experienced.")

    if safe_num(f2.get("TD", np.nan)) > safe_num(f1.get("TD", np.nan)):
        risks.append(f"{f2_name} may present a grappling threat.")
    if safe_num(f2.get("KO_Rate", np.nan)) > safe_num(f1.get("KO_Rate", np.nan)):
        risks.append(f"{f2_name} may carry stronger KO danger.")
    if safe_num(f2.get("SUB_Rate", np.nan)) > safe_num(f1.get("SUB_Rate", np.nan)):
        risks.append(f"{f2_name} may have more submission threat.")
    if safe_num(f2.get("STR", np.nan)) > safe_num(f1.get("STR", np.nan)):
        risks.append(f"{f2_name} may outpace in striking exchanges.")

    if not advantages:
        advantages = ["No clear single-feature edge was detected from the current profile features."]
    if not risks:
        risks = ["No major red flags were detected from the current profile features."]

    return advantages[:4], risks[:4]

def make_local_explanation(f1_name, f2_name, f1, f2):
    explanation = []
    pairs = [
        ("reach", safe_num(f1.get("Reach", np.nan)) - safe_num(f2.get("Reach", np.nan))),
        ("experience", safe_num(f1.get("W", np.nan)) - safe_num(f2.get("W", np.nan))),
        ("striking", safe_num(f1.get("STR", np.nan)) - safe_num(f2.get("STR", np.nan))),
        ("grappling", safe_num(f1.get("TD", np.nan)) - safe_num(f2.get("TD", np.nan))),
        ("KO threat", safe_num(f1.get("KO_Rate", np.nan)) - safe_num(f2.get("KO_Rate", np.nan))),
        ("submission threat", safe_num(f1.get("SUB_Rate", np.nan)) - safe_num(f2.get("SUB_Rate", np.nan))),
    ]

    for label, diff in pairs:
        if pd.isna(diff):
            continue
        if diff > 0:
            explanation.append(f"{f1_name} has an edge in {label}.")
        elif diff < 0:
            explanation.append(f"{f2_name} has an edge in {label}.")

    if not explanation:
        explanation.append("No single dominant matchup edge was detected from the available profile features.")

    return explanation[:4]

def normalized_pair(a, b):
    a = safe_num(a)
    b = safe_num(b)

    if pd.isna(a) and pd.isna(b):
        return 0.0, 0.0

    if pd.isna(a):
        return 0.0, 100.0
    if pd.isna(b):
        return 100.0, 0.0

    max_val = max(abs(a), abs(b), 1e-9)
    return (a / max_val) * 100.0, (b / max_val) * 100.0

def render_metric_cards(title, metrics, f1, f2, fighter_a, fighter_b):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    cols = st.columns(len(metrics))

    for col, m in zip(cols, metrics):
        a_val = safe_num(f1.get(m["key"], np.nan))
        b_val = safe_num(f2.get(m["key"], np.nan))

        if pd.isna(a_val) and pd.isna(b_val):
            advantage = "No data"
        elif pd.isna(a_val):
            advantage = fighter_b
        elif pd.isna(b_val):
            advantage = fighter_a
        elif a_val > b_val:
            advantage = fighter_a
        elif b_val > a_val:
            advantage = fighter_b
        else:
            advantage = "Even"

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-name">{m["label"]}</div>
                    <div><b>{fighter_a}</b>: {display_num(a_val)}</div>
                    <div><b>{fighter_b}</b>: {display_num(b_val)}</div>
                    <div class="metric-desc">{m["desc"]}</div>
                    <div class="advantage-pill">Edge: {advantage}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def render_normalized_comparison_chart(title, metrics, f1, f2, fighter_a, fighter_b):
    labels = []
    a_vals = []
    b_vals = []

    for m in metrics:
        a_norm, b_norm = normalized_pair(f1.get(m["key"], np.nan), f2.get(m["key"], np.nan))
        labels.append(m["label"])
        a_vals.append(a_norm)
        b_vals.append(b_norm)

    idx = np.arange(len(labels))
    bar_h = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.barh(idx - bar_h / 2, a_vals, height=bar_h, label=fighter_a)
    ax.barh(idx + bar_h / 2, b_vals, height=bar_h, label=fighter_b)

    ax.set_yticks(idx)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative scale (0–100 within each metric)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)

def render_metric_reference_table(metrics, f1, f2, fighter_a, fighter_b):
    table_df = pd.DataFrame({
        "Metric": [m["label"] for m in metrics],
        "Meaning": [m["desc"] for m in metrics],
        fighter_a: [display_num(f1.get(m["key"], np.nan)) for m in metrics],
        fighter_b: [display_num(f2.get(m["key"], np.nan)) for m in metrics],
    })
    st.dataframe(table_df, use_container_width=True)


# ============================================================
# METRIC DEFINITIONS
# ============================================================
STRIKING_METRICS = [
    {
        "key": "STR",
        "label": "STR",
        "desc": "Overall striking output. Higher values suggest more striking activity."
    },
    {
        "key": "KD",
        "label": "KD",
        "desc": "Knockdown metric. Higher values suggest more knockdown potential."
    },
    {
        "key": "KO_Rate",
        "label": "KO Rate",
        "desc": "Knockout finishing profile. Higher values suggest stronger KO threat."
    },
    {
        "key": "Sig_Str_Pct",
        "label": "Sig Str %",
        "desc": "Significant strike accuracy. Higher values suggest cleaner striking efficiency."
    },
]

GRAPPLING_METRICS = [
    {
        "key": "TD",
        "label": "TD",
        "desc": "Takedown-related metric. Higher values suggest stronger takedown ability."
    },
    {
        "key": "SUB",
        "label": "SUB",
        "desc": "Submission-related metric. Higher values suggest more submission activity."
    },
    {
        "key": "Ctrl",
        "label": "Ctrl",
        "desc": "Control metric. Higher values suggest stronger positional control."
    },
    {
        "key": "Sub_Att",
        "label": "Sub Att",
        "desc": "Submission attempts. Higher values suggest more active submission threat."
    },
]


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🥊 FightIQ</h1>
    <p>Explainable UFC Matchup Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<span class="pill">Win probability</span>'
    '<span class="pill">Striking vs grappling breakdown</span>'
    '<span class="pill">Readable fighter comparison</span>'
    '<span class="pill">Project insights</span>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Matchup Predictor", "Fighter Explorer", "Project Insights"])


# ============================================================
# TAB 1 — MATCHUP PREDICTOR
# ============================================================
with tab1:
    fighter_names = sorted(fighters["Full_Name"].dropna().unique().tolist())

    st.markdown('<div class="section-title">Matchup setup</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 0.8])

    with c1:
        fighter_a = st.selectbox("Fighter A", fighter_names, index=0)

    with c2:
        fighter_b = st.selectbox("Fighter B", fighter_names, index=1 if len(fighter_names) > 1 else 0)

    with c3:
        default_year = int(metadata.get("test_years", [2025])[-1]) if metadata.get("test_years") else 2025
        fight_year = st.number_input("Fight Year", min_value=2000, max_value=2035, value=default_year)

    time_format = st.radio(
        "Time Format",
        ["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"],
        horizontal=True
    )

    if st.button("Run Prediction", use_container_width=True):
        if fighter_a == fighter_b:
            st.warning("Please choose two different fighters.")
        else:
            X_row, f1, f2, inferred_weight_class = build_matchup_row(fighter_a, fighter_b, time_format, fight_year)

            if X_row is None:
                st.error("Could not construct the matchup row.")
            else:
                prob_a = float(model.predict_proba(X_row)[0, 1])
                prob_b = 1 - prob_a

                st.markdown('<div class="section-title">Fighter cards</div>', unsafe_allow_html=True)
                lc, rc = st.columns(2)
                with lc:
                    st.markdown(fighter_card_html(fighter_a, f1), unsafe_allow_html=True)
                with rc:
                    st.markdown(fighter_card_html(fighter_b, f2), unsafe_allow_html=True)

                st.markdown('<div class="info-box"><b>Context:</b> '
                            f'Weight class: <b>{inferred_weight_class}</b> &nbsp;&nbsp;|&nbsp;&nbsp; '
                            f'Time format: <b>{time_format}</b> &nbsp;&nbsp;|&nbsp;&nbsp; '
                            f'Fight year: <b>{fight_year}</b></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">Win probability</div>', unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(fighter_a, display_prob(prob_a))
                with m2:
                    st.metric(fighter_b, display_prob(prob_b))

                st.progress(min(max(prob_a, 0.0), 1.0))

                advantages, risks = build_advantages_and_risks(fighter_a, fighter_b, f1, f2)
                ac, rc = st.columns(2)

                with ac:
                    st.markdown('<div class="section-title">Key advantages</div>', unsafe_allow_html=True)
                    for item in advantages:
                        st.markdown(f'<div class="good-box">✓ {item}</div>', unsafe_allow_html=True)

                with rc:
                    st.markdown('<div class="section-title">Risk flags</div>', unsafe_allow_html=True)
                    for item in risks:
                        st.markdown(f'<div class="risk-box">⚠ {item}</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-title">Model explanation</div>', unsafe_allow_html=True)
                for item in make_local_explanation(fighter_a, fighter_b, f1, f2):
                    st.markdown(f'<div class="info-box">• {item}</div>', unsafe_allow_html=True)

                render_metric_cards("Striking snapshot", STRIKING_METRICS, f1, f2, fighter_a, fighter_b)
                render_normalized_comparison_chart(
                    "Striking comparison (relative per metric)",
                    STRIKING_METRICS,
                    f1,
                    f2,
                    fighter_a,
                    fighter_b
                )
                render_metric_reference_table(STRIKING_METRICS, f1, f2, fighter_a, fighter_b)

                render_metric_cards("Grappling snapshot", GRAPPLING_METRICS, f1, f2, fighter_a, fighter_b)
                render_normalized_comparison_chart(
                    "Grappling comparison (relative per metric)",
                    GRAPPLING_METRICS,
                    f1,
                    f2,
                    fighter_a,
                    fighter_b
                )
                render_metric_reference_table(GRAPPLING_METRICS, f1, f2, fighter_a, fighter_b)

                st.markdown('<div class="section-title">Overall comparison table</div>', unsafe_allow_html=True)
                compare_df = pd.DataFrame({
                    "Feature": [
                        "Height", "Weight", "Reach", "Stance",
                        "Wins", "Losses", "KD", "STR", "TD", "SUB",
                        "Ctrl", "KO Rate", "SUB Rate", "DEC Rate"
                    ],
                    fighter_a: [
                        f1.get("Height", np.nan),
                        f1.get("Weight", np.nan),
                        f1.get("Reach", np.nan),
                        f1.get("Stance", np.nan),
                        f1.get("W", np.nan),
                        f1.get("L", np.nan),
                        f1.get("KD", np.nan),
                        f1.get("STR", np.nan),
                        f1.get("TD", np.nan),
                        f1.get("SUB", np.nan),
                        f1.get("Ctrl", np.nan),
                        f1.get("KO_Rate", np.nan),
                        f1.get("SUB_Rate", np.nan),
                        f1.get("DEC_Rate", np.nan),
                    ],
                    fighter_b: [
                        f2.get("Height", np.nan),
                        f2.get("Weight", np.nan),
                        f2.get("Reach", np.nan),
                        f2.get("Stance", np.nan),
                        f2.get("W", np.nan),
                        f2.get("L", np.nan),
                        f2.get("KD", np.nan),
                        f2.get("STR", np.nan),
                        f2.get("TD", np.nan),
                        f2.get("SUB", np.nan),
                        f2.get("Ctrl", np.nan),
                        f2.get("KO_Rate", np.nan),
                        f2.get("SUB_Rate", np.nan),
                        f2.get("DEC_Rate", np.nan),
                    ]
                })
                st.dataframe(compare_df, use_container_width=True)

                st.markdown(
                    '<div class="small-note">Charts above are normalized separately for each metric, '
                    'so small-valued features such as KO Rate or KD remain readable. The tables show the real values.</div>',
                    unsafe_allow_html=True
                )


# ============================================================
# TAB 2 — FIGHTER EXPLORER
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Fighter explorer</div>', unsafe_allow_html=True)

    fighter_names = sorted(fighters["Full_Name"].dropna().unique().tolist())
    selected_fighter = st.selectbox("Choose a fighter", fighter_names, key="fighter_explorer")

    row = get_fighter_row(selected_fighter)

    if row is not None:
        st.markdown(fighter_card_html(selected_fighter, row), unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Height", display_num(row.get("Height", np.nan)))
        with m2:
            st.metric("Weight", display_num(row.get("Weight", np.nan)))
        with m3:
            st.metric("Reach", display_num(row.get("Reach", np.nan)))

        m4, m5, m6 = st.columns(3)
        with m4:
            st.metric("Wins", display_num(row.get("W", np.nan), 0))
        with m5:
            st.metric("Losses", display_num(row.get("L", np.nan), 0))
        with m6:
            st.metric("Draws", display_num(row.get("D", np.nan), 0))

        render_metric_cards("Striking profile", STRIKING_METRICS, row, row, selected_fighter, selected_fighter)

        fig, ax = plt.subplots(figsize=(8, 4.7))
        strike_vals = [safe_num(row.get(m["key"], np.nan)) for m in STRIKING_METRICS]
        strike_labels = [m["label"] for m in STRIKING_METRICS]
        ax.barh(strike_labels, strike_vals)
        ax.set_title(f"Striking Profile — {selected_fighter}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        render_metric_cards("Grappling profile", GRAPPLING_METRICS, row, row, selected_fighter, selected_fighter)

        fig, ax = plt.subplots(figsize=(8, 4.7))
        grap_vals = [safe_num(row.get(m["key"], np.nan)) for m in GRAPPLING_METRICS]
        grap_labels = [m["label"] for m in GRAPPLING_METRICS]
        ax.barh(grap_labels, grap_vals)
        ax.set_title(f"Grappling Profile — {selected_fighter}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        if model_base is not None and "Fighter_1" in model_base.columns and "Fighter_2" in model_base.columns:
            as_f1 = int((model_base["Fighter_1"] == selected_fighter).sum())
            as_f2 = int((model_base["Fighter_2"] == selected_fighter).sum())
            st.markdown(
                f'<div class="info-box">Historical presence in cleaned base table: '
                f'<b>{as_f1}</b> rows as Fighter 1, <b>{as_f2}</b> rows as Fighter 2.</div>',
                unsafe_allow_html=True
            )


# ============================================================
# TAB 3 — PROJECT INSIGHTS
# ============================================================
with tab3:
    st.markdown('<div class="section-title">Project insights</div>', unsafe_allow_html=True)

    if comparison_df is not None:
        st.markdown("#### Model comparison")
        st.dataframe(comparison_df, use_container_width=True)

        if "model" in comparison_df.columns and "test_roc_auc" in comparison_df.columns:
            plot_df = comparison_df[["model", "feature_set", "test_roc_auc"]].copy()
            plot_df["label"] = plot_df["model"] + " (" + plot_df["feature_set"] + ")"

            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            ax.barh(plot_df["label"], plot_df["test_roc_auc"])
            ax.set_title("Test ROC-AUC by Model")
            ax.set_xlabel("ROC-AUC")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

    st.markdown("#### Hypothesis testing summary")
    st.markdown('<div class="good-box">✓ Grappling advantage was statistically significant (p = 0.000).</div>', unsafe_allow_html=True)
    st.markdown('<div class="good-box">✓ Experience advantage was statistically significant (p = 0.000).</div>', unsafe_allow_html=True)
    st.markdown('<div class="good-box">✓ Striking advantage was statistically significant (p = 0.000).</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-box">✗ Reach advantage was not statistically significant (p = 0.371).</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-box">✗ Southpaw stance was not statistically significant (p = 0.205).</div>', unsafe_allow_html=True)

    if perm_df is not None and not perm_df.empty:
        st.markdown("#### Top feature importance")
        top_perm = perm_df.sort_values("importance_mean", ascending=False).head(10).copy()
        st.dataframe(top_perm, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        ax.barh(top_perm["feature"][::-1], top_perm["importance_mean"][::-1])
        ax.set_title("Top 10 Permutation Importances")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(
        '<div class="info-box">This demo uses the saved best-performing model from the modeling notebook. '
        'The project distinguishes between a safer baseline pipeline and an enriched exploratory pipeline '
        'to keep methodology transparent.</div>',
        unsafe_allow_html=True
    )