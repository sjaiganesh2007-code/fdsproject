import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from rapidfuzz import process, fuzz


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoBill Verify",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CGHS_PATH = BASE_DIR / "data" / "cghs_rate_list.csv"
BILLING_PATH = BASE_DIR / "data" / "billing_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "isolation_forest.joblib"


# =========================================================
# PREMIUM UI
# =========================================================

st.markdown("""
<style>

/* Remove Streamlit top header background */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}
/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    /* =========================================================
   FINAL SIDEBAR + AUDITOR VISIBILITY
   ========================================================= */

[data-testid="stSidebar"] {
    background: #081525 !important;
    border-right: 1px solid rgba(148, 163, 184, 0.14) !important;
}

[data-testid="stSidebar"] > div {
    background: #081525 !important;
}

[data-testid="stSidebar"] section {
    background: #081525 !important;
}

[data-testid="stSidebar"] * {
    color: #dbe7f5 !important;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #dbe7f5 !important;
}

/* Bill Auditor headings and labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
.stSelectbox label,
.stNumberInput label,
.stFileUploader label {
    color: #dbe7f5 !important;
    font-weight: 600 !important;
}

/* Radio options */
[data-testid="stRadio"] [role="radiogroup"] {
    gap: 14px;
}

[data-testid="stRadio"] [role="radio"] {
    border-color: #94a3b8 !important;
}

[data-testid="stRadio"] [role="radio"][aria-checked="true"] {
    border-color: #38bdf8 !important;
}

/* Select box */
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #0f172a !important;
}

/* Number input */
.stNumberInput input {
    background: #ffffff !important;
    color: #0f172a !important;
}
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(56,189,248,0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 5% 85%,
            rgba(59,130,246,0.08),
            transparent 30%
        ),
        #07111f;

    color: #e5edf7;
}
.block-container {
    max-width: 1450px;
    padding-top: 0.8rem;


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: #081525;
    border-right: 1px solid rgba(148,163,184,0.14);
}

[data-testid="stSidebar"] * {
    color: #dbe7f5;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;

    padding: 42px 46px;

    border-radius: 26px;

    background:
        radial-gradient(
            circle at var(--mouse-x, 85%) var(--mouse-y, 20%),
            rgba(56,189,248,0.20),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #0b192b 0%,
            #102944 100%
        );

    border: 1px solid rgba(125,211,252,0.18);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.30);

    margin-bottom: 26px;

    transition:
        background 0.15s ease,
        box-shadow 0.25s ease;
}
.hero:hover {
    border-color: rgba(125,211,252,0.35);

    box-shadow:
        0 30px 80px rgba(0,0,0,0.38),
        0 0 35px rgba(56,189,248,0.06);
}
.hero::after {
    content: "";
    position: absolute;

    width: 240px;
    height: 240px;

    right: -80px;
    top: -100px;

    border-radius: 50%;

    background: rgba(56,189,248,0.08);

    filter: blur(5px);
}

.hero-badge {
    display: inline-block;

    color: #67e8f9;

    font-size: 12px;
    font-weight: 800;

    letter-spacing: 1.5px;

    margin-bottom: 12px;
}

.hero h1 {
    margin: 0;

    color: #f8fbff;

    font-size: clamp(34px, 5vw, 56px);

    line-height: 1.05;

    font-weight: 900;

    letter-spacing: -1.5px;
}

.hero-subtitle {
    margin-top: 10px;

    color: #7dd3fc;

    font-size: clamp(17px, 2vw, 22px);

    font-weight: 700;
}

.hero-description {
    max-width: 760px;

    margin-top: 14px;

    color: #9fb3c8;

    font-size: 15px;

    line-height: 1.7;
}


/* =========================================================
   KPI CARDS
   ========================================================= */

.kpi-card {
    min-height: 125px;

    padding: 22px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(15,35,56,0.92),
            rgba(10,25,42,0.92)
        );

    border: 1px solid rgba(148,163,184,0.14);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-5px);

    border-color: rgba(125,211,252,0.35);

    box-shadow:
        0 18px 45px rgba(0,0,0,0.28);
}

.kpi-label {
    color: #7dd3fc;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.3px;
}

.kpi-value {
    margin-top: 8px;

    color: #ffffff;

    font-size: clamp(25px, 3vw, 34px);

    font-weight: 900;
}

.kpi-sub {
    margin-top: 4px;

    color: #71879d;

    font-size: 12px;
}


/* =========================================================
   QUICK ACTIONS
   ========================================================= */

.action-card {
    display: flex;

    align-items: center;

    gap: 18px;

    min-height: 120px;

    padding: 22px;

    border-radius: 18px;

    background:
        rgba(12,29,48,0.88);

    border: 1px solid rgba(148,163,184,0.13);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        background 0.25s ease;
}

.action-card:hover {
    transform: translateY(-4px);

    border-color: rgba(56,189,248,0.35);

    background:
        rgba(16,39,63,0.95);
}

.action-icon {
    min-width: 52px;
    height: 52px;

    display: flex;

    align-items: center;
    justify-content: center;

    border-radius: 14px;

    background: rgba(56,189,248,0.10);

    font-size: 24px;
}

.action-title {
    color: #f8fbff;

    font-size: 18px;

    font-weight: 800;
}

.action-text {
    margin-top: 5px;

    color: #8ea5ba;

    font-size: 13px;

    line-height: 1.55;
}


/* =========================================================
   PIPELINE
   ========================================================= */

.pipeline-card {
    min-height: 155px;

    padding: 20px;

    border-radius: 17px;

    background:
        rgba(12,29,48,0.78);

    border: 1px solid rgba(148,163,184,0.12);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease;
}

.pipeline-card:hover {
    transform: translateY(-4px);

    border-color: rgba(125,211,252,0.28);
}

.pipeline-number {
    color: #38bdf8;

    font-size: 12px;

    font-weight: 900;

    letter-spacing: 1px;
}

.pipeline-title {
    margin-top: 12px;

    color: #f1f7ff;

    font-size: 17px;

    font-weight: 800;
}

.pipeline-text {
    margin-top: 7px;

    color: #849bb0;

    font-size: 12px;

    line-height: 1.55;
}


/* =========================================================
   FOOTER NOTE
   ========================================================= */

.dashboard-note {
    margin-top: 25px;

    padding: 14px 18px;

    border-radius: 12px;

    background: rgba(15,31,50,0.65);

    border: 1px solid rgba(148,163,184,0.10);

    color: #7890a5;

    font-size: 12px;

    text-align: center;
}


/* =========================================================
   SPACING
   ========================================================= */

.section-gap {
    height: 25px;
}

.section-gap-small {
    height: 14px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

div.stButton > button {
    border-radius: 12px;

    border: 1px solid rgba(125,211,252,0.25);

    background:
        linear-gradient(
            135deg,
            rgba(14,45,75,0.95),
            rgba(12,30,50,0.95)
        );

    color: #e0f7ff;

    font-weight: 700;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    border-color: #7dd3fc;
}


/* =========================================================
   RESPONSIVE DESIGN
   ========================================================= */

@media (max-width: 1100px) {

    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .hero {
        padding: 32px;
    }

}

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        padding: 28px 22px;
        border-radius: 20px;
    }

    .hero h1 {
        font-size: 36px;
    }

    .hero-subtitle {
        font-size: 17px;
    }

    .hero-description {
        font-size: 13px;
    }

    .kpi-card {
        min-height: 105px;
        padding: 18px;
    }

    .action-card {
        min-height: auto;
        padding: 18px;
    }

    .pipeline-card {
        min-height: auto;
    }

}

@media (max-width: 480px) {

    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .hero {
        padding: 24px 18px;
    }

    .hero h1 {
        font-size: 30px;
    }

    .hero-badge {
        font-size: 10px;
    }

    .kpi-value {
        font-size: 24px;
    }

}

</style>
""", unsafe_allow_html=True)
st.markdown(
    """
<script>
(function () {

    document.addEventListener("mousemove", function (event) {

        const x = (event.clientX / window.innerWidth) * 100;
        const y = (event.clientY / window.innerHeight) * 100;

        document.documentElement.style.setProperty(
            "--mouse-x",
            x + "%"
        );

        document.documentElement.style.setProperty(
            "--mouse-y",
            y + "%"
        );

    });

})();
</script>
""",
    unsafe_allow_html=True
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_cghs():
    if not CGHS_PATH.exists():
        return None

    return pd.read_csv(CGHS_PATH)


@st.cache_data
def load_billing():
    if not BILLING_PATH.exists():
        return None

    return pd.read_csv(BILLING_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


cghs_df = load_cghs()
billing_df = load_billing()
model = load_model()


# =========================================================
# BASIC VALIDATION
# =========================================================

if cghs_df is None:
    st.error("CGHS dataset could not be loaded.")
    st.stop()

if billing_df is None:
    st.error("Billing dataset could not be loaded.")
    st.stop()

if model is None:
    st.error("Isolation Forest model could not be loaded.")
    st.stop()


cghs_names = cghs_df["item_name"].astype(str).tolist()


# =========================================================
# FUNCTIONS
# =========================================================

@st.cache_data
def match_item(item_name):

    result = process.extractOne(
        str(item_name),
        cghs_names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=40
    )

    if not result:
        return None

    index = result[2]
    score = float(result[1])

    row = cghs_df.iloc[index]

    return {
        "matched_name": row["item_name"],
        "item_code": row["item_code"],
        "category": row["category"],
        "cghs_rate": float(row["cghs_rate"]),
        "std_dev": float(row["std_dev"]),
        "match_score": score
    }


def classify_risk(price_ratio, z_score, anomaly):

    if anomaly or price_ratio > 2.50 or z_score > 4:
        return "HIGH ANOMALY RISK", "danger"

    if price_ratio > 1.25 or z_score > 2:
        return "MODERATE RISK", "warning"

    return "NORMAL", "normal"


def audit_item(item_name, billed_price):

    info = match_item(item_name)

    if info is None:
        return None

    cghs_rate = info["cghs_rate"]
    std_dev = info["std_dev"]

    price_ratio = billed_price / cghs_rate if cghs_rate > 0 else 1

    z_score = (
        (billed_price - cghs_rate) / std_dev
        if std_dev > 0
        else 0
    )

    prediction = model.predict(
        [[price_ratio, z_score]]
    )[0]

    anomaly = prediction == -1

    risk, risk_class = classify_risk(
        price_ratio,
        z_score,
        anomaly
    )

    excess = max(
        0,
        billed_price - cghs_rate
    )

    return {
        "match": info,
        "price_ratio": price_ratio,
        "z_score": z_score,
        "prediction": prediction,
        "anomaly": anomaly,
        "risk": risk,
        "risk_class": risk_class,
        "excess": excess
    }
    # =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🏥 AUTO BILL VERIFY
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Bill Auditor"
        ]
    )




# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        """<div class="hero">
<div class="hero-badge">● AI BILL VERIFICATION SYSTEM</div>
<h1>AUTO BILL VERIFY</h1>
<div class="hero-subtitle">Smart Hospital Bill Fraud & Anomaly Detection</div>
<div class="hero-description">Verify hospital billing charges using benchmark comparison, intelligent matching, statistical analysis and machine learning.</div>
</div>""",
        unsafe_allow_html=True
    )

    # KPI CARDS
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""<div class="kpi-card">
<div class="kpi-label">BILLING RECORDS</div>
<div class="kpi-value">{len(billing_df):,}</div>
<div class="kpi-sub">Hospital billing dataset</div>
</div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""<div class="kpi-card">
<div class="kpi-label">BENCHMARK ITEMS</div>
<div class="kpi-value">{len(cghs_df):,}</div>
<div class="kpi-sub">CGHS reference catalog</div>
</div>""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """<div class="kpi-card">
<div class="kpi-label">ML ENGINE</div>
<div class="kpi-value">ACTIVE</div>
<div class="kpi-sub">Isolation Forest</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='section-gap'></div>",
        unsafe_allow_html=True
    )

    # QUICK ACTIONS
    st.markdown("### Quick Actions")

    q1, q2 = st.columns(2)

    with q1:
        st.markdown(
            """<div class="action-card">
<div class="action-icon">🔍</div>
<div class="action-content">
<div class="action-title">Single Bill Audit</div>
<div class="action-text">Verify an individual hospital charge against the CGHS benchmark and detect potential anomalies.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    with q2:
        st.markdown(
            """<div class="action-card">
<div class="action-icon">📄</div>
<div class="action-content">
<div class="action-title">Bulk Bill Audit</div>
<div class="action-text">Upload a hospital billing CSV and analyse multiple billing entries in one operation.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='section-gap-small'></div>",
        unsafe_allow_html=True
    )

    # DETECTION PIPELINE
    st.markdown("### Detection Pipeline")

    p1, p2, p3, p4 = st.columns(4)

    pipeline = [
        ("01", "RapidFuzz", "Match billing descriptions with benchmark items."),
        ("02", "Price Ratio", "Compare billed price with benchmark rate."),
        ("03", "Z-Score", "Measure statistical price deviation."),
        ("04", "Isolation Forest", "Identify unusual billing patterns.")
    ]

    pipeline_columns = [p1, p2, p3, p4]

    for col, item in zip(pipeline_columns, pipeline):

        number, title, description = item

        with col:
            st.markdown(
                f"""<div class="pipeline-card">
<div class="pipeline-number">{number}</div>
<div class="pipeline-title">{title}</div>
<div class="pipeline-text">{description}</div>
</div>""",
                unsafe_allow_html=True
            )

    st.markdown(
        """<div class="dashboard-note">
AutoBill Verify is a decision-support system for preliminary billing verification. Flagged entries require human review.
</div>""",
        unsafe_allow_html=True
    )


# =========================================================
# BILL AUDITOR
# =========================================================

elif page == "🔍 Bill Auditor":

    st.markdown("## 🔍 Bill Auditor")

    st.caption(
        "Compare hospital billing items against CGHS benchmark rates "
        "and detect potential anomalies."
    )

    mode = st.radio(
        "Audit Mode",
        [
            "Single Item",
            "Bulk CSV"
        ],
        horizontal=True
    )

    # -----------------------------------------------------
    # SINGLE ITEM
    # -----------------------------------------------------

    if mode == "Single Item":

        st.markdown("### Single Bill Verification")

        item = st.selectbox(
            "Select CGHS / Hospital Service Item",
            cghs_names
        )

        matched = match_item(item)

        if matched:

            suggested = matched["cghs_rate"] * 1.2

            price = st.number_input(
                "Billed Unit Price (₹)",
                min_value=1.0,
                value=float(round(suggested, 2)),
                step=50.0
            )

            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1,
                step=1
            )

            if st.button(
                "🔍 VERIFY BILL",
                type="primary",
                use_container_width=True
            ):

                result = audit_item(
                    item,
                    price
                )

                if result:

                    info = result["match"]

                    st.divider()

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "CGHS Benchmark",
                        f"₹{info['cghs_rate']:,.2f}"
                    )

                    b.metric(
                        "Billed Price",
                        f"₹{price:,.2f}"
                    )

                    c.metric(
                        "Price Ratio",
                        f"{result['price_ratio']:.2f}x"
                    )

                    d.metric(
                        "Z-Score",
                        f"{result['z_score']:+.2f}"
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    if result["risk_class"] == "danger":

                        st.markdown(f"""
                        <div class="result-danger">

                        ## 🔴 HIGH ANOMALY RISK

                        **Potentially suspicious billing detected.**

                        Billed amount is
                        **{result['price_ratio']:.2f}×**
                        the CGHS benchmark.

                        Potential excess:
                        **₹{result['excess']:,.2f}**

                        Isolation Forest:
                        **Anomaly**

                        </div>
                        """, unsafe_allow_html=True)

                    elif result["risk_class"] == "warning":

                        st.markdown(f"""
                        <div class="result-warning">

                        ## 🟡 MODERATE RISK

                        Billing exceeds the benchmark
                        and requires manual review.

                        Price ratio:
                        **{result['price_ratio']:.2f}×**

                        Potential excess:
                        **₹{result['excess']:,.2f}**

                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown(f"""
                        <div class="result-normal">

                        ## 🟢 NORMAL

                        The billed amount is within the
                        expected benchmark range.

                        Price ratio:
                        **{result['price_ratio']:.2f}×**

                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### 🧠 Verification Explanation")

                    e1, e2 = st.columns(2)

                    with e1:

                        st.write(
                            f"**Matched CGHS Item:** "
                            f"{info['matched_name']}"
                        )

                        st.write(
                            f"**Match Confidence:** "
                            f"{info['match_score']:.1f}%"
                        )

                        st.write(
                            f"**Category:** "
                            f"{info['category']}"
                        )

                    with e2:

                        st.write(
                            f"**Price Ratio:** "
                            f"{result['price_ratio']:.2f}x"
                        )

                        st.write(
                            f"**Z-Score:** "
                            f"{result['z_score']:+.2f}"
                        )

                        st.write(
                            f"**Isolation Forest:** "
                            f"{'Anomaly' if result['anomaly'] else 'Normal'}"
                        )


    # -----------------------------------------------------
    # BULK CSV
    # -----------------------------------------------------

    else:

        st.markdown("### 📄 Bulk CSV Audit")

        uploaded = st.file_uploader(
            "Upload Hospital Billing CSV",
            type=["csv"]
        )

        if uploaded:

            raw = pd.read_csv(uploaded)

            st.write(
                f"Loaded **{len(raw):,} rows**."
            )

            name_col = next(
                (
                    c for c in raw.columns
                    if any(
                        x in c.lower()
                        for x in ["item", "name", "desc"]
                    )
                ),
                raw.columns[0]
            )

            price_col = next(
                (
                    c for c in raw.columns
                    if any(
                        x in c.lower()
                        for x in ["price", "rate", "amount"]
                    )
                ),
                raw.columns[1]
            )

            st.info(
                f"Item column: `{name_col}`  |  "
                f"Price column: `{price_col}`"
            )

            if st.button(
                "⚡ RUN BULK AUDIT",
                type="primary"
            ):

                clean = raw.copy()

                clean[price_col] = pd.to_numeric(
                    clean[price_col],
                    errors="coerce"
                )

                clean = clean.dropna(
                    subset=[name_col, price_col]
                )

                unique_names = (
                    clean[name_col]
                    .astype(str)
                    .unique()
                )

                match_map = {}

                progress = st.progress(0)

                for i, name in enumerate(unique_names):

                    match_map[name] = match_item(name)

                    progress.progress(
                        (i + 1) / len(unique_names)
                    )

                matched_names = []
                cghs_rates = []
                z_scores = []
                ratios = []
                scores = []

                for _, row in clean.iterrows():

                    name = str(row[name_col])
                    price = float(row[price_col])

                    info = match_map.get(name)

                    if info:

                        rate = info["cghs_rate"]
                        std = info["std_dev"]
                        score = info["match_score"]

                    else:

                        rate = price
                        std = max(1.0, price * 0.1)
                        score = 0

                    ratio = (
                        price / rate
                        if rate > 0
                        else 1
                    )

                    z = (
                        (price - rate) / std
                        if std > 0
                        else 0
                    )

                    matched_names.append(
                        info["matched_name"]
                        if info
                        else "UNMATCHED"
                    )

                    cghs_rates.append(rate)
                    ratios.append(ratio)
                    z_scores.append(z)
                    scores.append(score)

                clean["matched_cghs_name"] = matched_names
                clean["cghs_rate"] = cghs_rates
                clean["price_ratio"] = ratios
                clean["z_score"] = z_scores
                clean["match_score"] = scores

                predictions = model.predict(
                    clean[
                        ["price_ratio", "z_score"]
                    ]
                )

                clean["isolation_forest"] = np.where(
                    predictions == -1,
                    "Anomaly",
                    "Normal"
                )

                clean["risk_classification"] = [
                    classify_risk(
                        ratio,
                        z,
                        pred == -1
                    )[0]
                    for ratio, z, pred
                    in zip(
                        clean["price_ratio"],
                        clean["z_score"],
                        predictions
                    )
                ]

                clean["potential_excess"] = np.maximum(
                    0,
                    clean[price_col] - clean["cghs_rate"]
                )

                st.session_state["bulk_result"] = clean

                st.success(
                    f"Audit completed for {len(clean):,} items."
                )

        if "bulk_result" in st.session_state:

            result_df = st.session_state["bulk_result"]

            st.divider()

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Audited Items",
                f"{len(result_df):,}"
            )

            r2.metric(
                "High Risk",
                str(
                    (
                        result_df["risk_classification"]
                        == "HIGH ANOMALY RISK"
                    ).sum()
                )
            )

            r3.metric(
                "Potential Excess",
                f"₹{result_df['potential_excess'].sum():,.0f}"
            )

            st.dataframe(
                result_df,
                use_container_width=True
            )

            st.download_button(
                "📥 DOWNLOAD AUDIT REPORT",
                result_df.to_csv(index=False),
                "autobill_audit_report.csv",
                "text/csv"
            )


# =========================================================
# ANALYTICS
# =========================================================

else:

    st.markdown("## 📊 Analytics")

    st.caption(
        "Dataset-level analytics without running unnecessary "
        "machine-learning inference during application startup."
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Billing Records",
        f"{len(billing_df):,}"
    )

    a2.metric(
        "CGHS Catalog Items",
        f"{len(cghs_df):,}"
    )

    if "billed_price" in billing_df.columns:

        avg_price = billing_df["billed_price"].mean()

        a3.metric(
            "Average Billed Price",
            f"₹{avg_price:,.0f}"
        )

    if "category" in billing_df.columns:

        a4.metric(
            "Categories",
            f"{billing_df['category'].nunique():,}"
        )

    st.divider()

    if "category" in billing_df.columns:

        st.markdown("### Category Distribution")

        category_counts = (
            billing_df["category"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(category_counts)

    if "billed_price" in billing_df.columns:

        st.markdown("### Billed Price Distribution")

        price_series = (
            billing_df["billed_price"]
            .clip(
                upper=billing_df["billed_price"].quantile(0.99)
            )
        )

        histogram = pd.cut(
            price_series,
            bins=15
        ).value_counts().sort_index()

        histogram.index = histogram.index.astype(str)

        st.bar_chart(histogram)

    st.divider()

    st.markdown("""
    ### 🔬 Model Features

    The anomaly detection model uses:

    - **Price Ratio** — billed price compared with CGHS benchmark
    - **Z-Score** — statistical deviation from benchmark variation
    - **Isolation Forest** — unsupervised anomaly detection
    - **RapidFuzz** — approximate matching of bill descriptions

    The system highlights **potential billing anomalies** and
    is intended for preliminary verification and human review.
    """)