from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="CareSight | Readmission intelligence",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding: 2rem 3.5rem 3rem; max-width: 1500px; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, .18); }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(31, 111, 235, .10), rgba(31, 111, 235, .03));
        border: 1px solid rgba(31, 111, 235, .18);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: #6b7280; }
    [data-testid="stMetricValue"] { font-weight: 700; }
    .hero {
        background: linear-gradient(120deg, #102a43 0%, #155e75 55%, #0f766e 100%);
        border-radius: 24px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 14px 35px rgba(15, 61, 80, .20);
    }
    .hero h1 { margin: 0; color: white; font-size: 2.25rem; letter-spacing: -.03em; }
    .hero p { margin: .5rem 0 0; color: rgba(255,255,255,.78); font-size: 1.05rem; }
    .eyebrow {
        color: #67e8f9;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }
    .section-label {
        color: #0f766e;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 1.5rem 0 .4rem;
    }
    .insight {
        background: rgba(15, 118, 110, .08);
        border-left: 4px solid #0f766e;
        border-radius: 0 12px 12px 0;
        padding: .8rem 1rem;
        margin: .5rem 0 1rem;
    }
    .muted { color: #6b7280; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics_file = ARTIFACTS / "metrics.json"
summary_file = ARTIFACTS / "analytics_summary.csv"
predictions_file = ARTIFACTS / "predictions_sample.csv"

if not all(path.exists() for path in [metrics_file, summary_file, predictions_file]):
    st.error("Project results are not available yet.", icon=":material/error:")
    st.markdown(
        "Run `run_pipeline.bat` first, then restart this dashboard.",
    )
    st.stop()

with open(metrics_file, encoding="utf-8") as file:
    metrics = json.load(file)

summary = pd.read_csv(summary_file)
predictions = pd.read_csv(predictions_file)
predictions["risk_percent"] = predictions["readmission_probability"] * 100
predictions["risk_band"] = pd.cut(
    predictions["risk_percent"],
    bins=[-1, 10, 20, 35, 100],
    labels=["Low", "Watch", "Elevated", "High"],
)


with st.sidebar:
    st.markdown("## :material/health_and_safety: CareSight")
    st.caption("Healthcare predictive analytics")
    st.markdown("---")
    st.markdown("### Explore predictions")
    threshold = st.slider(
        "Readmission threshold",
        min_value=0.10,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Rows at or above this probability are labelled as predicted early readmission.",
    )
    selected_band = st.multiselect(
        "Risk bands",
        options=["Low", "Watch", "Elevated", "High"],
        default=["Low", "Watch", "Elevated", "High"],
    )
    selected_gender = st.multiselect(
        "Gender",
        options=sorted(predictions["gender"].dropna().unique()),
        default=sorted(predictions["gender"].dropna().unique()),
    )
    st.markdown("---")
    st.caption("Prototype status")
    st.badge("Historical dataset", icon=":material/database:", color="blue")
    st.caption("Model: Spark MLlib Logistic Regression")

filtered = predictions[
    predictions["risk_band"].isin(selected_band)
    & predictions["gender"].isin(selected_gender)
].copy()
filtered["threshold_prediction"] = (
    filtered["readmission_probability"] >= threshold
).astype(float)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Clinical operations dashboard</div>
        <h1>Readmission intelligence</h1>
        <p>Understand early readmission risk across held-out hospital encounters.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

total = int(metrics["cleaned_records"])
early_rate = float(predictions["label"].mean())
above_threshold = int(filtered["threshold_prediction"].sum())
highest_risk = float(predictions["risk_percent"].max())

st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Clean encounters", f"{total:,}", help="Records remaining after cleaning.")
kpi_2.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
kpi_3.metric("Early readmission rate", f"{early_rate:.1%}", help="Observed rate in the displayed held-out sample.")
kpi_4.metric("Highest observed risk", f"{highest_risk:.1f}%")

st.markdown(
    f'<div class="insight"><strong>{above_threshold:,}</strong> of {len(filtered):,} filtered records meet the current '
    f'<strong>{threshold:.0%}</strong> prediction threshold.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown('<div class="section-label">Risk by age group</div>', unsafe_allow_html=True)
    summary["rate_percent"] = summary["early_readmission_rate"] * 100
    age_fig = px.bar(
        summary,
        x="age",
        y="rate_percent",
        color="rate_percent",
        color_continuous_scale=["#ccfbf1", "#0f766e"],
        text="rate_percent",
        hover_data={"encounters": ":,", "rate_percent": ":.1f"},
        labels={"age": "Age group", "rate_percent": "Early readmission rate (%)"},
    )
    age_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    age_fig.update_layout(
        height=355,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title=None, ticksuffix="%", gridcolor="rgba(128,128,128,.16)"),
        xaxis=dict(title=None),
    )
    st.plotly_chart(age_fig, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-label">Observed outcomes</div>', unsafe_allow_html=True)
    outcomes = predictions["label"].map({1.0: "Readmitted within 30 days", 0.0: "Not readmitted"})
    outcome_counts = outcomes.value_counts().rename_axis("outcome").reset_index(name="records")
    outcome_fig = go.Figure(
        go.Pie(
            labels=outcome_counts["outcome"],
            values=outcome_counts["records"],
            hole=0.68,
            marker=dict(colors=["#0f766e", "#cbd5e1"]),
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:,} records<extra></extra>",
        )
    )
    outcome_fig.update_layout(
        height=355,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True,
        legend=dict(orientation="h", y=-.05),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(outcome_fig, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="section-label">Probability distribution</div>', unsafe_allow_html=True)
risk_fig = px.histogram(
    predictions,
    x="risk_percent",
    nbins=24,
    color_discrete_sequence=["#155e75"],
    labels={"risk_percent": "Readmission probability (%)", "count": "Records"},
)
risk_fig.add_vline(
    x=threshold * 100,
    line_dash="dash",
    line_color="#f97316",
    annotation_text=f"Threshold {threshold:.0%}",
    annotation_position="top",
)
risk_fig.update_layout(
    height=270,
    margin=dict(l=0, r=0, t=35, b=0),
    bargap=.08,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title=None, ticksuffix="%"),
    yaxis=dict(title=None, gridcolor="rgba(128,128,128,.16)"),
)
st.plotly_chart(risk_fig, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="section-label">Model quality</div>', unsafe_allow_html=True)
quality_1, quality_2, quality_3, quality_4 = st.columns(4)
quality_1.metric("ROC-AUC", "N/A" if metrics.get("roc_auc") is None else f"{metrics['roc_auc']:.3f}")
quality_2.metric("Weighted precision", f"{metrics['weighted_precision']:.3f}")
quality_3.metric("Weighted recall", f"{metrics['weighted_recall']:.3f}")
quality_4.metric("Weighted F1", f"{metrics['f1_score']:.3f}")

st.markdown('<div class="section-label">Held-out prediction records</div>', unsafe_allow_html=True)
st.caption("Use the controls in the sidebar to explore risk bands and test different decision thresholds.")
display = filtered.copy()
display["Actual outcome"] = display["label"].map(
    {1.0: "Readmitted within 30 days", 0.0: "Not readmitted"}
)
display["Model prediction"] = display["threshold_prediction"].map(
    {1.0: "Predicted early readmission", 0.0: "Not predicted"}
)
display["Risk"] = display["risk_percent"].round(1).map(lambda value: f"{value:.1f}%")
display = display.sort_values("readmission_probability", ascending=False)

st.dataframe(
    display[
        [
            "age",
            "gender",
            "time_in_hospital",
            "Actual outcome",
            "Model prediction",
            "risk_band",
            "Risk",
        ]
    ],
    width="stretch",
    height=430,
    hide_index=True,
    column_config={
        "age": st.column_config.TextColumn("Age group"),
        "gender": st.column_config.TextColumn("Gender"),
        "time_in_hospital": st.column_config.NumberColumn("Days in hospital", format="%d"),
        "risk_band": st.column_config.TextColumn("Risk band"),
        "Risk": st.column_config.TextColumn("Readmission risk"),
    },
)

download = display.to_csv(index=False)
st.download_button(
    "Download filtered predictions",
    data=download,
    file_name="healthcare_predictions.csv",
    mime="text/csv",
    icon=":material/download:",
)

st.caption(
    "Academic prototype using historical hospital data. It is not clinically validated and must not be used for diagnosis or treatment decisions."
)
