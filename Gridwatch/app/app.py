import sys
from pathlib import Path
import os

import pandas as pd
import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))
sys.path.append(str(APP_DIR))

from researcher import show_researcher_dashboard

from policy_recommendation_from_shap import (
    load_policy_docs,
    build_chunk_index,
    build_embedding_index,
    shap_to_query,
    retrieve_top_chunks,
    generate_grounded_recommendation,
    extract_program_names,
)
from rag_pipeline import build_index, retrieve_context

st.set_page_config(page_title="GridWatch", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Core card ─────────────────────────────────────────────────── */
.gw-card {
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  margin-bottom: 1rem;
}
.gw-section-header {
  border-left: 3px solid #3B82F6;
  padding-left: 12px;
  margin-bottom: 1.5rem;
  margin-top: 1.5rem;
}

/* ── Risk badges ───────────────────────────────────────────────── */
.gw-badge-high {
  background: rgba(220,38,38,0.12); color: #EF4444;
  border: 1px solid rgba(220,38,38,0.3);
  padding: 4px 12px; border-radius: 20px; font-weight: 600;
  font-size: 0.85rem; display: inline-block;
}
.gw-badge-low {
  background: rgba(22,163,74,0.12); color: #22C55E;
  border: 1px solid rgba(22,163,74,0.3);
  padding: 4px 12px; border-radius: 20px; font-weight: 600;
  font-size: 0.85rem; display: inline-block;
}

/* ── Stat cards ────────────────────────────────────────────────── */
.gw-stat-card {
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px;
  padding: 1rem 1.2rem; text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.gw-stat-label {
  font-size: 0.72rem; font-weight: 500; color: var(--text-color); opacity: 0.6;
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;
}
.gw-stat-value { font-size: 1.8rem; font-weight: 700; color: var(--text-color); line-height: 1.2; }
.gw-stat-sub { font-size: 0.78rem; color: var(--text-color); opacity: 0.5; margin-top: 0.3rem; }

/* ── Policy boxes ──────────────────────────────────────────────── */
.gw-policy-immediate {
  background: rgba(245,158,11,0.1); border-left: 4px solid #F59E0B;
  border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem;
}
.gw-policy-medium {
  background: rgba(59,130,246,0.1); border-left: 4px solid #3B82F6;
  border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem;
}
.gw-policy-gaps {
  background: rgba(139,92,246,0.1); border-left: 4px solid #8B5CF6;
  border-radius: 0 12px 12px 0; padding: 1.2rem 1.5rem;
}

/* ── Landing page ──────────────────────────────────────────────── */
.gw-hero { text-align: center; padding: 3rem 1rem 1.5rem; }
.gw-hero h1 {
  font-size: 2.5rem; font-weight: 700; letter-spacing: -0.5px; margin: 0;
  color: var(--text-color);
}
.gw-hero p {
  font-size: 1.1rem; font-weight: 400; margin: 0.5rem 0 1.5rem;
  color: var(--text-color); opacity: 0.65;
}
.gw-landing-card {
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 16px; padding: 2rem;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* ── Alert / info boxes ────────────────────────────────────────── */
.gw-alert-success {
  background: rgba(22,163,74,0.1);
  border: 1px solid rgba(22,163,74,0.3);
  border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
}
.gw-alert-info {
  background: rgba(59,130,246,0.1);
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
}
.gw-savings-box {
  background: rgba(22,163,74,0.08);
  border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
}

/* ── City snapshot grid ────────────────────────────────────────── */
.gw-snapshot-grid { display: flex; gap: 12px; margin: 1rem 0; flex-wrap: wrap; }
.gw-snapshot-card {
  flex: 1; min-width: 130px;
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px; padding: 1.2rem; text-align: center;
}
.gw-snap-label {
  font-size: 0.7rem; font-weight: 500; color: var(--text-color); opacity: 0.6;
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;
}
.gw-snap-sub { font-size: 0.75rem; color: var(--text-color); opacity: 0.5; margin-top: 0.3rem; }

/* ── Researcher top bar ────────────────────────────────────────── */
.gw-topbar {
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  padding: 0.75rem 2rem; margin-bottom: 16px;
  font-size: 0.9rem; border-radius: 8px;
}

/* ── Program cards ─────────────────────────────────────────────── */
.gw-program-card {
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
  background: var(--secondary-background-color);
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ── Not-qualified card ────────────────────────────────────────── */
.gw-not-qualified {
  background: var(--secondary-background-color);
  border: 1px solid rgba(148,163,184,0.2);
  border-radius: 10px; padding: 20px 24px;
  color: var(--text-color); line-height: 1.7;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* ── Forecast box ──────────────────────────────────────────────── */
.gw-forecast-box {
  background: rgba(59,130,246,0.1);
  border: 1px solid rgba(59,130,246,0.25);
  border-radius: 6px; padding: 12px 16px; margin-bottom: 12px;
}

/* ── Utility helpers ───────────────────────────────────────────── */
.gw-text-muted { color: var(--text-color); opacity: 0.55; }
.gw-footer { color: var(--text-color); opacity: 0.45; font-size: 0.8rem; margin-top: 2rem; }

/* ── Mobile responsive ─────────────────────────────────────────── */
@media (max-width: 768px) {
  .gw-hero h1 { font-size: 1.85rem; }
  .gw-hero p  { font-size: 0.95rem; }
  .gw-hero    { padding: 1.5rem 0.5rem 1rem; }
  .gw-landing-card { padding: 1.25rem; }
  .gw-card, .gw-program-card { padding: 0.9rem 1rem; }
  .gw-stat-value { font-size: 1.4rem; }
  .gw-snapshot-grid { flex-direction: column; }
  .gw-snapshot-card { min-width: unset; }
  .gw-topbar { padding: 0.6rem 1rem; }
  /* Stack Streamlit columns vertically */
  [data-testid="column"] { min-width: min(100%, 300px) !important; }
}

/* ── Streamlit overrides ────────────────────────────────────────── */
[data-testid="stDataFrame"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── paths ─────────────────────────────────────────────────────────────────────
data_path   = ROOT_DIR / "data" / "processed" / "city_master_dataset_50.csv"
shap_path   = ROOT_DIR / "data" / "processed" / "shap_values_50.csv"
policy_path = ROOT_DIR / "data" / "policy"

# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(data_path), pd.read_csv(shap_path)

@st.cache_resource
def load_policy_index():
    docs          = load_policy_docs(policy_path)
    chunk_records = build_chunk_index(docs)
    embeddings    = build_embedding_index(chunk_records)
    return chunk_records, embeddings

@st.cache_resource
def load_chroma_index():
    return build_index()

data_df, shap_df          = load_data()
chunk_records, embeddings = load_policy_index()
load_chroma_index()

# ── LLM client ────────────────────────────────────────────────────────────────
BASE_URL   = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
client     = OpenAI(base_url=BASE_URL, api_key=os.getenv("TINKER_API_KEY"))

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_city" not in st.session_state:
    st.session_state.last_city = None
if "eval_history" not in st.session_state:
    st.session_state.eval_history = []
if "user_mode" not in st.session_state:
    st.session_state.user_mode = None
if "resident_step" not in st.session_state:
    st.session_state.resident_step = 1
if "res_city" not in st.session_state:
    st.session_state.res_city = None
if "res_hh_size" not in st.session_state:
    st.session_state.res_hh_size = None
if "res_income_range" not in st.session_state:
    st.session_state.res_income_range = None

# ── helpers ───────────────────────────────────────────────────────────────────
def build_city_context(city_row, impacts):
    top_drivers = "\n".join([f"  - {k}: {v:.4f}" for k, v in impacts.items()])
    return (
        f"City: {city_row['city']}\n"
        f"  Energy burden: {city_row['energy_burden']:.2f}%\n"
        f"  Household income: ${city_row['household_income']:,.0f}\n"
        f"  Avg annual energy cost: ${city_row['avg_annual_energy_cost']:,.0f}\n"
        f"  Risk label: {'HIGH' if city_row['risk_label'] == 1 else 'LOW'}\n"
        f"  Top SHAP drivers:\n{top_drivers}"
    )


def build_all_cities_table(df):
    lines = ["city,energy_burden(%),household_income($),avg_annual_energy_cost($),risk"]
    for _, row in df.iterrows():
        risk = "HIGH" if row["risk_label"] == 1 else "LOW"
        lines.append(
            f"{row['city']},{row['energy_burden']:.2f},"
            f"{row['household_income']:.0f},"
            f"{row['avg_annual_energy_cost']:.0f},{risk}"
        )
    return "\n".join(lines)


def get_eligible_programs(income: float) -> list:
    """City-level eligibility based on median household income (for official view)."""
    programs = []
    if income < 66_527:
        programs.append("LIHEAP")
    if income < 64_300:
        programs.append("CARE")
    if 64_300 <= income <= 80_375:
        programs.append("FERA")
    if income < 64_300:
        programs.append("WAP")
    if income < 64_300:
        programs.append("ESA")
    return programs


def get_resident_eligible_programs(income: float, hh_size: int) -> list:
    """Resident-facing eligibility scaled by actual household size."""
    scale = {1: 0.60, 2: 0.75, 3: 0.88, 4: 1.00, 5: 1.12, 6: 1.25, 7: 1.37, 8: 1.50}
    s = scale.get(hh_size, 1.50)
    liheap_t = 66_527 * s
    care_t   = 64_300 * s
    fera_t   = 80_375 * s
    wap_t    = 64_300 * s
    esa_t    = 64_300 * s

    programs = []
    if income < liheap_t:
        programs.append("LIHEAP")
    if income < care_t:
        programs.append("CARE")
    if care_t <= income < fera_t:
        programs.append("FERA")
    if income < wap_t:
        programs.append("WAP")
    if income < esa_t:
        programs.append("ESA")
    return programs


def evaluate_response(question: str, answer: str, city_name: str, risk_level: str) -> dict:
    eval_prompt = f"""You are evaluating an AI answer about energy affordability in {city_name} (risk level: {risk_level}).

Here is the FULL AI answer you must evaluate:
\"\"\"
{answer}
\"\"\"

User question that was asked: {question}

---
STEP 1 — Read the answer carefully. Before scoring, find and quote a specific phrase from the answer as evidence for each dimension.

STEP 2 — Score each dimension based ONLY on what is present in the answer above.

ACCURACY — Is what the answer claims factually appropriate for this city?
- If the city is LOW RISK and the answer correctly states that fewer programs apply or the city is not high-priority, that is accurate — score it HIGH (4–5).
- If the city is HIGH RISK and the answer cites specific dollar amounts, income thresholds, or program eligibility details, score HIGH (4–5).
- If the answer is vague or generic but not wrong, score 3.
- Only score 1–2 if the answer makes a factually incorrect claim.

CLARITY — How readable and well-structured is the answer?
- Score based on actual sentence length, organization, and use of plain language in the text above.
- A short but perfectly clear answer can score 5. A long rambling answer should score lower.

ACTIONABILITY — Does the answer give the user something concrete to do?
- For LOW RISK cities: if the answer correctly explains the user likely does not qualify for most programs, that IS actionable — score 3–4.
- For HIGH RISK cities: score 5 only if the answer includes a specific next step (apply to X, call Y, check income limit of $Z).
- Score based on what IS in the answer, not what is absent.

RELEVANCE — Does the answer address this specific city's situation?
- Quote the part of the answer that references {city_name}, its risk level, or its data.
- If the answer is city-specific, score 4–5. If it is entirely generic, score 2–3.

Scoring scale: 5=Excellent, 4=Good, 3=Adequate, 2=Weak, 1=Poor

Return ONLY valid JSON, nothing else:
{{"accuracy": <int>, "clarity": <int>, "actionability": <int>, "relevance": <int>, "rationale": "<one quoted phrase + one-sentence reason per dimension, separated by | >"}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
        )
        import json
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        return {"accuracy": 0, "clarity": 0, "actionability": 0, "relevance": 0, "rationale": f"Evaluation failed: {e}"}


# ── resident experience constants ──────────────────────────────────────────────
_INCOME_MIDPOINTS = {
    "Under $20,000":       15_000,
    "$20,000 – $35,000":   27_500,
    "$35,000 – $50,000":   42_500,
    "$50,000 – $65,000":   57_500,
    "$65,000 – $80,000":   72_500,
    "$80,000 – $100,000":  90_000,
    "Over $100,000":      110_000,
}

PROGRAM_INFO = {
    "LIHEAP": {
        "full_name": "LIHEAP — Low Income Home Energy Assistance Program",
        "what": (
            "LIHEAP (Low Income Home Energy Assistance Program) is a federal program funded by the "
            "U.S. Department of Health and Human Services. It provides one-time financial assistance "
            "to help low-income households pay their energy bills — especially during extreme heat or cold. "
            "This is not a loan. You do not pay it back."
        ),
        "trust": (
            "Yes. LIHEAP is a federal program administered by the California Department of Community "
            "Services and Development (CSD). It has operated since 1981. You can verify the program at "
            '<a href="https://www.csd.ca.gov/pages/liheapprogram.aspx" target="_blank" rel="noopener noreferrer">'
            "csd.ca.gov</a>. GridWatch does not collect or store your personal information."
        ),
        "steps": [
            '<strong>Find the official program page</strong> — Visit '
            '<a href="https://www.csd.ca.gov/pages/liheapprogram.aspx" target="_blank" rel="noopener noreferrer">csd.ca.gov/liheap</a> '
            "or call <strong>1-866-675-6623</strong> to find your local agency.",

            "<strong>Gather documents</strong> — You will need: proof of income (pay stub or tax return), "
            "a recent utility bill, and ID for all household members.",

            '<strong>Apply online at <a href="https://www.caliheapapply.com" target="_blank" rel="noopener noreferrer">caliheapapply.com</a></strong> '
            "— this is California's official LIHEAP portal. Or call <strong>1-866-675-6623</strong> to apply by phone.",

            "<strong>Wait for approval</strong> — Processing takes 2–4 weeks. If approved, funds go "
            "directly to your utility company — you do not handle the money.",
        ],
        "savings_type": "flat",
        "bill_type":    "energy",
    },
    "CARE": {
        "full_name": "CARE — California Alternate Rates for Energy",
        "what": (
            "CARE (California Alternate Rates for Energy) is a California state program run by your "
            "utility company under the California Public Utilities Commission (CPUC). It gives qualifying "
            "households <strong>up to 32.5% off</strong> their electricity bills — a permanent monthly discount. "
            "Once enrolled, the discount applies automatically every month — no reapplication needed annually."
        ),
        "trust": (
            "Yes. CARE is regulated by the "
            '<a href="https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-costs/care-fera-program" '
            'target="_blank" rel="noopener noreferrer">California Public Utilities Commission (CPUC)</a>, '
            "a state government agency. It is administered directly by your utility company — SCE, PG&amp;E, "
            "SDG&amp;E, or others. GridWatch does not collect or store your personal information."
        ),
        "steps": [
            '<strong>SCE customers:</strong> Visit '
            '<a href="https://www.sce.com/save-money/income-qualified-programs/care-fera" target="_blank" rel="noopener noreferrer">sce.com/care-fera</a> '
            "or call <strong>1-800-447-6620</strong>. The online application takes 2 minutes and no documents are needed upfront.<br>"
            '<strong>PG&amp;E customers:</strong> Visit <a href="https://www.pge.com/en/account/rate-plans/care-fera.html" target="_blank" rel="noopener noreferrer">pge.com/care-fera</a> or call <strong>1-800-743-5000</strong>.<br>'
            '<strong>SDG&amp;E customers:</strong> Visit <a href="https://www.sdge.com/residential/savings-center/care-fera" target="_blank" rel="noopener noreferrer">sdge.com/care-fera</a> or call <strong>1-800-411-7343</strong>.',

            "<strong>Tell them you want to apply for CARE</strong> — they will ask for your household size and income.",

            "<strong>Provide proof of income if requested</strong> — a pay stub, benefits letter, or tax return.",

            "<strong>Once approved, the discount appears on your very next bill automatically.</strong>",
        ],
        "savings_type": "pct",
        "savings_pct":  0.32,
        "bill_type":    "electricity",
    },
    "FERA": {
        "full_name": "FERA — Family Electric Rate Assistance",
        "what": (
            "FERA (Family Electric Rate Assistance) is a California state program for households that "
            "earn slightly too much for CARE but still struggle with energy costs. It provides a permanent "
            "<strong>18% monthly discount</strong> specifically on electricity bills. FERA requires a household "
            "of <strong>3 or more people</strong>. Like CARE, it is administered by your utility company under "
            "the CPUC — it is a legitimate government program, not a third-party service."
        ),
        "trust": (
            "Yes. FERA is regulated by the "
            '<a href="https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-costs/care-fera-program" '
            'target="_blank" rel="noopener noreferrer">California Public Utilities Commission (CPUC)</a>, '
            "a state government agency. It is administered directly by your utility company — SCE, PG&amp;E, "
            "SDG&amp;E, or others. GridWatch does not collect or store your personal information."
        ),
        "steps": [
            '<strong>SCE customers:</strong> Visit '
            '<a href="https://www.sce.com/save-money/income-qualified-programs/care-fera" target="_blank" rel="noopener noreferrer">sce.com/care-fera</a> '
            "or call <strong>1-800-447-6620</strong>.<br>"
            '<strong>PG&amp;E customers:</strong> Visit <a href="https://www.pge.com/en/account/rate-plans/care-fera.html" target="_blank" rel="noopener noreferrer">pge.com/care-fera</a> or call <strong>1-800-743-5000</strong>.<br>'
            '<strong>SDG&amp;E customers:</strong> Visit <a href="https://www.sdge.com/residential/savings-center/care-fera" target="_blank" rel="noopener noreferrer">sdge.com/care-fera</a> or call <strong>1-800-411-7343</strong>.',

            "<strong>Tell them you want to apply for FERA</strong> — mention your household has 3 or more people.",

            "<strong>Provide proof of income and household size</strong> — a pay stub or tax return, plus something showing everyone who lives with you.",

            "<strong>Once approved, the 18% discount applies to your electricity bill every month automatically.</strong>",
        ],
        "savings_type": "pct",
        "savings_pct":  0.18,
        "bill_type":    "electricity",
    },
    "WAP": {
        "full_name": "WAP — Weatherization Assistance Program",
        "what": (
            "WAP (Weatherization Assistance Program) is a federal program run by the U.S. Department "
            "of Energy. It sends qualified technicians to your home to make free energy efficiency "
            "improvements — things like insulation, sealing drafts, and upgrading heating or cooling systems. "
            "These improvements reduce your energy bills permanently. This is a one-time service, not ongoing payments. "
            "WAP has helped over 7.2 million families since 1976. On average, households save "
            "<strong>$372 or more per year</strong> after improvements."
        ),
        "trust": (
            "Yes. WAP is a federal program administered by the U.S. Department of Energy and delivered "
            "through local agencies. You can verify the program at "
            '<a href="https://www.energy.gov/cmei/scep/wap/how-apply-weatherization-assistance" '
            'target="_blank" rel="noopener noreferrer">energy.gov/wap</a>. '
            "GridWatch does not collect or store your personal information."
        ),
        "steps": [
            '<strong>Find your local provider</strong> — Visit '
            '<a href="https://www.energy.gov/cmei/scep/wap/how-apply-weatherization-assistance" target="_blank" rel="noopener noreferrer">energy.gov/wap</a> '
            "or call <strong>1-866-675-6623</strong> (administered by the same California agency as LIHEAP). "
            "Urgent cases — elderly residents, children under 5, or people with disabilities — are prioritized.",

            "<strong>Apply through your local Community Action Agency</strong> — they manage the waitlist and scheduling.",

            "<strong>A program representative will contact you</strong> to schedule a free home energy audit at no cost.",

            "<strong>Qualified technicians will visit your home</strong> and complete all improvements free of charge.",

            "<strong>Improvements are permanent</strong> — no repayment needed.",
        ],
        "savings_type": "pct_onetime",
        "savings_pct":  0.15,
        "bill_type":    "energy",
    },
    "ESA": {
        "full_name": "ESA — Energy Savings Assistance Program",
        "what": (
            "ESA (Energy Savings Assistance Program) is a California program run by utility companies. "
            "It provides free energy-efficient upgrades to your home — such as LED lighting, efficient "
            "appliances, and weatherproofing. Like WAP, this is a one-time service that permanently "
            "lowers your monthly bills."
        ),
        "trust": (
            "Yes. ESA is regulated by the "
            '<a href="https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-costs/care-fera-program" '
            'target="_blank" rel="noopener noreferrer">California Public Utilities Commission (CPUC)</a> '
            "and administered by California utility companies. "
            "GridWatch does not collect or store your personal information."
        ),
        "steps": [
            '<strong>Contact your utility company:</strong><br>'
            '- SCE: <a href="https://www.sce.com/residential/rebates-savings/esa-program" target="_blank" rel="noopener noreferrer">sce.com/esa</a> or <strong>1-800-736-4777</strong><br>'
            '- PG&amp;E: <a href="https://www.pge.com/en/save-energy-and-money/energy-saving-programs/energy-saving-assistance-program.html" target="_blank" rel="noopener noreferrer">pge.com/esa</a> or <strong>1-800-743-5000</strong>',

            "<strong>Request an ESA home visit</strong> — a program representative will schedule a time.",

            "<strong>A technician will assess your home</strong> and install free energy-efficient upgrades.",

            "<strong>All work is free</strong> — no cost to you at any point.",
        ],
        "savings_type": "pct_onetime",
        "savings_pct":  0.12,
        "bill_type":    "energy",
    },
}

_RESIDENT_FOOTER = (
    "GridWatch is a research tool. Information is provided for guidance only. "
    "Contact program administrators to confirm your eligibility."
)

ALL_CITIES_TABLE = build_all_cities_table(data_df)

SYSTEM_PROMPT = f"""You are GridWatch AI, a knowledgeable and direct energy affordability advisor.
Speak like a trusted expert — conversational, clear, and human. Never use section labels, headers, or numbered lists in your responses.

You have data for all 50 California cities:

{ALL_CITIES_TABLE}

How to respond:
- Write in natural flowing prose, maximum 3-4 short paragraphs.
- Weave your reasoning naturally into the answer — never label it "Reasoning:" or "Step 1:".
- Cite policy sources inline and naturally, e.g. "according to LIHEAP guidelines..." or "the WAP program covers...".
- Use real numbers from the data (income, burden %, energy cost) to make the answer specific to this city.
- End with one clear, specific recommendation the user can act on today.
- Risk is HIGH when energy burden >= 3% of household income.
- Do not hallucinate data or program details not present in the context."""


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING — switch view button + mode gate
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.user_mode is not None:
    _sw_cols = st.columns([9, 1])
    with _sw_cols[1]:
        if st.button("↩ Switch view", key="switch_view_btn", type="secondary"):
            st.session_state.user_mode = None
            st.session_state.resident_step = 1
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.user_mode is None:
    st.markdown(
        "<div class='gw-hero'>"
        "<h1>GridWatch</h1>"
        "<p>Energy Affordability Intelligence for California</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    land_l, land_r = st.columns(2, gap="large")

    with land_l:
        st.markdown(
            "<div class='gw-landing-card'>"
            "<div style='font-size:2rem;margin-bottom:0.75rem;'>🏠</div>"
            "<h3 style='color:var(--text-color);font-weight:600;margin:0 0 0.5rem;'>I'm a Resident</h3>"
            "<p style='color:var(--text-color);opacity:0.65;margin:0;font-size:0.95rem;'>"
            "Find out if you qualify for energy assistance programs in California</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Check My Eligibility →", key="btn_resident", use_container_width=True, type="primary"):
            st.session_state.user_mode = "resident"
            st.session_state.resident_step = 1
            st.rerun()

    with land_r:
        st.markdown(
            "<div class='gw-landing-card'>"
            "<div style='font-size:2rem;margin-bottom:0.75rem;'>📊</div>"
            "<h3 style='color:var(--text-color);font-weight:600;margin:0 0 0.5rem;'>I'm a City Official or Researcher</h3>"
            "<p style='color:var(--text-color);opacity:0.65;margin:0;font-size:0.95rem;'>"
            "Explore city-level energy risk data and policy recommendations</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Research Dashboard →", key="btn_official", use_container_width=True, type="secondary"):
            st.session_state.user_mode = "official"
            st.rerun()

    st.markdown(
        "<p class='gw-footer' style='text-align:center;margin-top:3rem;'>"
        "GridWatch is a research tool. Information is provided for guidance only.</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RESIDENT EXPERIENCE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.user_mode == "resident":
    cities_list = sorted(data_df["city"].unique())
    step = st.session_state.resident_step

    # progress bar
    prog_pct = 0.33 if step == 1 else 0.80
    st.progress(prog_pct)
    st.caption(f"Step {step} of 2")

    # ── STEP 1: Input form ────────────────────────────────────────────────────
    if step == 1:
        st.markdown(
            "<h2 style='margin-bottom:0;color:var(--text-color);'>Find Your Energy Assistance</h2>"
            "<p style='color:var(--text-color);opacity:0.65;margin-top:4px;'>Answer 3 quick questions — takes less than 1 minute. "
            "We do not store or share your information.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        with st.form("resident_input_form"):
            res_city = st.selectbox(
                "Which city do you live in?",
                ["— Select your city —"] + cities_list,
            )

            hh_options = ["1", "2", "3", "4", "5", "6", "7", "8+"]
            res_hh_str = st.selectbox(
                "How many people live in your household (including yourself)?",
                ["— Select —"] + hh_options,
            )

            income_options = [
                "Under $20,000",
                "$20,000 – $35,000",
                "$35,000 – $50,000",
                "$50,000 – $65,000",
                "$65,000 – $80,000",
                "$80,000 – $100,000",
                "Over $100,000",
            ]
            res_income = st.selectbox(
                "What is your approximate total household income before taxes? "
                "(We use ranges to protect your privacy)",
                ["— Select —"] + income_options,
            )

            submit = st.form_submit_button("Check My Eligibility →", type="primary", use_container_width=True)

        if submit:
            if res_city == "— Select your city —" or res_hh_str == "— Select —" or res_income == "— Select —":
                st.warning("Please answer all three questions to continue.")
            else:
                hh_size = 8 if res_hh_str == "8+" else int(res_hh_str)
                st.session_state.res_city        = res_city
                st.session_state.res_hh_size     = hh_size
                st.session_state.res_income_range = res_income
                st.session_state.resident_step   = 2
                st.rerun()

        st.markdown(
            f"<p class='gw-footer'>{_RESIDENT_FOOTER}</p>",
            unsafe_allow_html=True,
        )

    # ── STEP 2: Results ───────────────────────────────────────────────────────
    else:
        res_city_name  = st.session_state.res_city
        res_hh_size    = st.session_state.res_hh_size
        res_income_lbl = st.session_state.res_income_range
        user_income    = _INCOME_MIDPOINTS[res_income_lbl]

        city_row      = data_df[data_df["city"] == res_city_name].iloc[0]
        energy_cost   = float(city_row["avg_annual_energy_cost"])

        qualified = get_resident_eligible_programs(user_income, res_hh_size)

        st.markdown(f"<h2 style='color:var(--text-color);'>Your Results for {res_city_name}</h2>", unsafe_allow_html=True)

        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Start over", key="back_btn"):
                st.session_state.resident_step = 1
                st.rerun()

        st.markdown("---")

        # ── Qualified path ────────────────────────────────────────────────────
        if qualified:
            st.markdown(
                "<div class='gw-alert-success'>"
                f"<span style='font-size:1.1rem;color:#22C55E;'>Good news — based on your "
                f"information, you likely qualify for energy assistance programs in {res_city_name}.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:var(--text-color);opacity:0.6;font-size:0.82rem;margin-bottom:1.5rem;'>"
                "Eligibility shown is an estimate based on income ranges. "
                "Final eligibility is confirmed by program administrators.</p>",
                unsafe_allow_html=True,
            )

            # Program cards
            for prog in qualified:
                info = PROGRAM_INFO[prog]

                # Calculate savings
                stype = info["savings_type"]
                if stype == "flat":
                    savings_html = (
                        "<p style='font-size:1.5rem;font-weight:700;color:#22C55E;margin:4px 0;'>"
                        "Up to $1,000 one-time</p>"
                        "<p style='color:var(--text-color);margin:0;'>Applied directly to your utility account.</p>"
                    )
                elif stype == "pct":
                    annual = energy_cost * info["savings_pct"]
                    monthly = annual / 12
                    savings_html = (
                        f"<p style='font-size:1.5rem;font-weight:700;color:#22C55E;margin:4px 0;'>"
                        f"~${annual:,.0f}/year</p>"
                        f"<p style='color:var(--text-color);margin:0;'>That's roughly <strong>${monthly:,.0f}/month</strong> "
                        f"off your {info['bill_type']} bill.</p>"
                    )
                else:  # pct_onetime
                    annual = energy_cost * info["savings_pct"]
                    savings_html = (
                        f"<p style='font-size:1.5rem;font-weight:700;color:#22C55E;margin:4px 0;'>"
                        f"~${annual:,.0f}/year in reduced costs</p>"
                        f"<p style='color:var(--text-color);margin:0;'>Estimated savings after free home improvements "
                        f"are completed.</p>"
                    )

                steps_html = "".join(
                    f"<li style='margin-bottom:10px;line-height:1.55;'>{s}</li>"
                    for s in info["steps"]
                )

                st.markdown(
                    f"<div class='gw-program-card'>"

                    f"<h3 style='color:#3B82F6;margin-top:0;'>{info['full_name']}</h3>"

                    f"<p style='color:var(--text-color);opacity:0.55;font-size:0.78rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:16px 0 4px;font-weight:500;'>What is this program?</p>"
                    f"<p style='color:var(--text-color);line-height:1.6;margin:0 0 16px;'>{info['what']}</p>"

                    f"<p style='color:var(--text-color);opacity:0.55;font-size:0.78rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:0 0 4px;font-weight:500;'>How much could you save?</p>"
                    f"<div class='gw-savings-box'>"
                    f"<p style='color:var(--text-color);opacity:0.7;margin:0 0 6px;font-size:0.9rem;'>"
                    f"Based on {res_city_name}'s average annual energy cost of ${energy_cost:,.0f}:</p>"
                    f"{savings_html}"
                    f"</div>"

                    f"<p style='color:var(--text-color);opacity:0.55;font-size:0.78rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:0 0 4px;font-weight:500;'>Is this program trustworthy?</p>"
                    f"<p style='color:var(--text-color);line-height:1.6;margin:0 0 16px;'>{info['trust']}</p>"

                    f"<p style='color:var(--text-color);opacity:0.55;font-size:0.78rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin:0 0 8px;font-weight:500;'>How to apply — step by step</p>"
                    f"<ol style='color:var(--text-color);line-height:1.6;padding-left:1.2rem;margin:0;'>"
                    f"{steps_html}</ol>"

                    f"</div>",
                    unsafe_allow_html=True,
                )

            # PDF download (plain text summary)
            pdf_lines = [
                f"GridWatch — Energy Assistance Results for {res_city_name}",
                f"Household size: {res_hh_size}  |  Income range: {res_income_lbl}",
                f"Average annual energy cost in {res_city_name}: ${energy_cost:,.0f}",
                "",
                "Programs you likely qualify for:",
                "=" * 50,
            ]
            for prog in qualified:
                info = PROGRAM_INFO[prog]
                pdf_lines.append(f"\n{info['full_name']}")
                pdf_lines.append("-" * 40)
                pdf_lines.append(f"What it is: {info['what']}")
                pdf_lines.append("\nHow to apply:")
                for i, step in enumerate(info["steps"], 1):
                    clean_step = step.replace("**", "")
                    pdf_lines.append(f"  {i}. {clean_step}")
                pdf_lines.append("")
            pdf_lines.append(_RESIDENT_FOOTER)

            st.download_button(
                label="💾 Save this information as PDF (text)",
                data="\n".join(pdf_lines).encode("utf-8"),
                file_name=f"gridwatch_{res_city_name.replace(' ', '_')}_eligibility.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # Step 3 — What's next checklist
            st.markdown("---")
            st.progress(1.0)
            st.caption("Step 2 of 2 — complete")
            st.markdown("### What's next?")
            st.markdown(
                "- [ ] Note which programs you qualify for above\n"
                "- [ ] Gather your documents (income proof + recent utility bill)\n"
                "- [ ] Contact your utility company or the program agency\n"
                "- [ ] Apply — most approvals take 2–4 weeks"
            )

        # ── Not qualified path ────────────────────────────────────────────────
        else:
            st.markdown(
                "<div class='gw-alert-info'>"
                "<span style='font-size:1.05rem;color:#3B82F6;'>"
                "Based on your income, you are currently above the eligibility threshold for "
                "most California energy assistance programs.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='gw-not-qualified'>"
                "<p>This means your household income is considered stable enough that you are not "
                "prioritized for subsidized assistance — which is a good thing.</p>"
                "<p>However, you still have options:</p>"
                "<p><strong>1. Budget Billing</strong> — Contact your utility company and ask about "
                "budget billing or equal payment plans. These spread your annual energy costs evenly "
                "across 12 months so there are no surprise high bills in summer or winter.</p>"
                "<p><strong>2. Energy Efficiency Tips</strong> — Your utility company offers free home "
                "energy audits and rebates on efficient appliances regardless of income. Ask about their "
                "rebate programs.</p>"
                "<p><strong>3. If Your Situation Changes</strong> — If your income drops due to job loss, "
                "medical expenses, or family changes, you may qualify for CARE, FERA, or LIHEAP. Come back "
                "and check again anytime.</p>"
                "<p><strong>4. REACH Program</strong> — Some utility companies offer the REACH program for "
                "customers facing temporary hardship regardless of income. Ask your utility if they offer it.</p>"
                "<p style='color:var(--text-color);opacity:0.55;font-size:0.85rem;margin-top:16px;margin-bottom:0;'>"
                "GridWatch does not store your information. Your answers are used only to generate this "
                "result and are discarded when you close this page.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<p class='gw-footer'>{_RESIDENT_FOOTER}</p>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# OFFICIAL / RESEARCHER EXPERIENCE
# ═════════════════════════════════════════════════════════════════════════════
else:
    show_researcher_dashboard(data_df, shap_df, client, MODEL_NAME)

