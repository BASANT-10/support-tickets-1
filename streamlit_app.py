# ─────────────────────────────────────────────────────────────
#  streamlit_evaluator.py
#  Evaluate tactic classifications produced by streamlit_app.py
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from io import StringIO
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

st.set_page_config(page_title="📈 Tactic Classifier Evaluator", layout="wide")
st.title("📈 Tactic Classifier Evaluator")

# ───── Default tactic dictionaries (same as the classifier app) ─────
default_tactics = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        'elegance', 'heritage', 'sophistication', 'refined', 'timeless', 'grace',
        'legacy', 'opulence', 'bespoke', 'tailored', 'understated', 'prestige',
        'quality', 'craftsmanship', 'heirloom', 'classic', 'tradition', 'iconic',
        'enduring', 'rich', 'authentic', 'luxury', 'fine', 'pure', 'exclusive',
        'elite', 'mastery', 'immaculate', 'flawless', 'distinction', 'noble',
        'chic', 'serene', 'clean', 'minimal', 'poised', 'balanced', 'eternal',
        'neutral', 'subtle', 'grand', 'timelessness', 'tasteful', 'quiet', 'sublime'
    ]
}

# ───────────────────────── Sidebar – file inputs ─────────────────────
with st.sidebar:
    st.header("📂 Upload files")
    pred_file = st.file_uploader("Classified results (from previous app)",
                                 type="csv")
    truth_mode = st.radio("Ground‑truth source",
                          ("Column inside same file",
                           "Separate file (ID + true_label)",
                           "None / exploratory only"))

    truth_file = None
    if truth_mode == "Separate file (ID + true_label)":
        truth_file = st.file_uploader("Ground‑truth CSV", type="csv")

# ───────────────────── Helper functions ───────────────────────────────
def _to_list(x):
    """
    Parse label cell into list.
    Accepts list‑string, comma‑separated string, or single label.
    """
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    x = str(x)
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
    except Exception:
        pass
    # fall back to comma‑split or single token
    return [t.strip() for t in x.split(",") if t.strip()]

def _tokenize(txt):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(txt).lower()).split()

# ───────────────────── Main processing block ──────────────────────────
if pred_file:
    pred_df = pd.read_csv(pred_file)

    # ─── Basic preview
    st.subheader("🔎 Preview of classified data")
    st.dataframe(pred_df.head())

    # ─── Column selections
    with st.expander("⚙️ Column settings", expanded=False):
        id_col = st.selectbox("ID column (optional)",
                              options=["<None>"] + list(pred_df.columns))
        cleaned_col = st.selectbox("Column containing cleaned text",
                                   options=list(pred_df.columns),
                                   index=list(pred_df.columns).index("cleaned")
                                   if "cleaned" in pred_df.columns else 0)
        categories_col = st.selectbox("Predicted categories column",
                                      options=list(pred_df.columns),
                                      index=list(pred_df.columns).index("categories")
                                      if "categories" in pred_df.columns else 0)

    # ensure prediction labels are lists
    pred_df["pred_labels"] = pred_df[categories_col].apply(_to_list)

    # ─── Merge ground truth if provided ────────────────────────────
    has_truth = False
    if truth_mode == "Column inside same file":
        possible_truth_cols = [c for c in pred_df.columns if "true" in c.lower()]
        if not possible_truth_cols:
            st.warning("No obvious true‑label column detected in file.")
        else:
            truth_col = st.selectbox("Select true‑label column",
                                     options=possible_truth_cols)
            pred_df["true_labels"] = pred_df[truth_col].apply(_to_list)
            has_truth = True
    elif truth_mode == "Separate file (ID + true_label)":
        if truth_file:
            truth_df = pd.read_csv(truth_file)
            truth_id_col = st.selectbox("ID column in truth file",
                                        options=list(truth_df.columns))
            truth_label_col = st.selectbox("Label column in truth file",
                                           options=[c for c in truth_df.columns
                                                    if c != truth_id_col])
            truth_df["true_labels"] = truth_df[truth_label_col].apply(_to_list)
            # merge
            pred_df = pred_df.merge(
                truth_df[[truth_id_col, "true_labels"]],
                left_on=id_col if id_col != "<None>" else pred_df.index,
                right_on=truth_id_col,
                how="left"
            )
            has_truth = True
        else:
            st.info("Upload ground‑truth CSV to enable metrics.")

    # ─── WORD‑LEVEL METRICS aggregated by ID ────────────────────────
    st.header("📝 Word‑level metrics (aggregated by ID)")

    # choose tactics to analyse
    tactics_to_use = st.multiselect("Tactics to include",
                                    list(default_tactics.keys()),
                                    default=list(default_tactics.keys()))

    # build per‑row word counts
    rows_data = []
    for _, row in pred_df.iterrows():
        toks = _tokenize(row[cleaned_col])
        total_words = len(toks) if toks else 1  # avoid div/0
        tactic_word_counts = {t: 0 for t in tactics_to_use}
        for t in tactics_to_use:
            tactic_word_counts[t] = sum(1 for tok in toks
                                        if tok in default_tactics[t])
        rows_data.append({
            "row_id": _ if id_col == "<None>" else row[id_col],
            "total_words": total_words,
            **tactic_word_counts
        })

    word_df = pd.DataFrame(rows_data)

    # aggregate to ID level
    agg_funcs = {t: "sum" for t in tactics_to_use}
    agg_funcs["total_words"] = "sum"
    id_word_df = (word_df
                  .groupby("row_id", as_index=False)
                  .agg(agg_funcs))

    # percentage columns
    for t in tactics_to_use:
        id_word_df[f"{t}_pct_words"] = (
            id_word_df[t] / id_word_df["total_words"] * 100).round(2)

    st.dataframe(id_word_df.head())

    # download
    st.download_button("📥 Download ID‑level word metrics",
                       id_word_df.to_csv(index=False).encode(),
                       file_name="id_word_metrics.csv",
                       mime="text/csv")

    # simple bar chart for a chosen tactic
    tactic_for_chart = st.selectbox("Select tactic for bar chart",
                                    tactics_to_use)
    chart_df = id_word_df.sort_values(f"{tactic_for_chart}_pct_words",
                                      ascending=False)
    st.bar_chart(chart_df.set_index("row_id")[f"{tactic_for_chart}_pct_words"],
                 height=350)

    # ─── CLASSIFIER METRICS (precision / recall / F1) ───────────────
    st.header("🏁 Classification metrics (per row)")

    if not has_truth:
        st.info("Provide ground‑truth labels to compute precision, recall & F1.")
    else:
        metric_rows = []
        for tactic in tactics_to_use:
            # binary vectors
            y_true = pred_df["true_labels"].apply(lambda lst: tactic in lst)
            y_pred = pred_df["pred_labels"].apply(lambda lst: tactic in lst)

            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)
            metric_rows.append({
                "Tactic": tactic,
                "Precision": round(p, 3),
                "Recall":    round(r, 3),
                "F1":        round(f, 3)
            })

        metrics_df = pd.DataFrame(metric_rows)\
                         .sort_values("F1", ascending=False)
        st.dataframe(metrics_df.set_index("Tactic"))

        st.download_button("📥 Download metrics CSV",
                           metrics_df.to_csv(index=False).encode(),
                           file_name="tactic_metrics.csv",
                           mime="text/csv")

        # bar chart of F1 scores
        st.subheader("🔢 F1 score by tactic")
        st.bar_chart(metrics_df.set_index("Tactic")["F1"], height=350)

else:
    st.info("Upload the classified results CSV in the sidebar to begin.")
