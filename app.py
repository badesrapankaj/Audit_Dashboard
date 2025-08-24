import io
import altair as alt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =============================================================================
# Page
# =============================================================================
st.set_page_config(page_title="Audit Dashboard", layout="wide")
st.title("📊 Audit Dashboard")

# =============================================================================
# Sidebar: File upload + Appearance
# =============================================================================
with st.sidebar:
    st.header("➡️ Upload File")
    uploaded = st.file_uploader(
        "Upload the **Files Audit** Excel (.xlsx)",
        type=["xlsx", "xlsm", "xls"],
        accept_multiple_files=False,
    )
    st.caption("Header is expected on row 4 (index 3). Adjust if needed.")
    header_row = st.number_input("Header row (0-indexed)", min_value=0, max_value=50, value=3, step=1)

    st.markdown("---")
    st.subheader("🎨 Appearance")

    THEMES = {
        "HLF (Hinduja Leyland)": {"primary": "#0079b2", "accent": "#312e81", "bg1": "#ffffff", "bg2": "#eef4ff"},
        "Ocean":                  {"primary": "#2563eb", "accent": "#22d3ee", "bg1": "#ffffff", "bg2": "#eef4ff"},
        "Sunset":                 {"primary": "#f97316", "accent": "#ef4444", "bg1": "#ffffff", "bg2": "#fff4ec"},
        "Amethyst":               {"primary": "#7c3aed", "accent": "#a78bfa", "bg1": "#ffffff", "bg2": "#f6f4ff"},
        "Emerald":                {"primary": "#059669", "accent": "#10b981", "bg1": "#ffffff", "bg2": "#ecfff7"},
    }
    theme_choice = st.selectbox("Theme colors", list(THEMES.keys()),
                                index=list(THEMES.keys()).index("HLF (Hinduja Leyland)"))
    mode = st.radio("Mode", ["Dark", "Light"], index=0, horizontal=True)

# ---- derive palette (dark by default) ----
is_dark = (mode == "Dark")
_base = THEMES[theme_choice]
primary, accent = _base["primary"], _base["accent"]
if is_dark:
    bg1, bg2 = "#0b1220", "#111827"
    text, text_muted = "#e5e7eb", "#9ca3af"
    card_bg, border, head_bg = "#0f172a", "rgba(255,255,255,0.10)", "#1f2937"
else:
    bg1, bg2 = _base["bg1"], _base["bg2"]
    text, text_muted = "#0f172a", "#475569"
    card_bg, border, head_bg = "#ffffff", "#e5e7eb", "#f8fafc"

# ---- inject CSS ----
st.markdown(
    f"""
    <style>
    :root {{
      --primary: {primary};
      --accent: {accent};
      --bg1: {bg1};
      --bg2: {bg2};
      --text: {text};
      --text-muted: {text_muted};
      --card: {card_bg};
      --border: {border};
      --thead: {head_bg};
      --popover: {"#1f2937" if is_dark else "#ffffff"};
      --chip-bg: {"rgba(255,255,255,0.10)" if is_dark else "#eef2ff"};
      --chip-text: {text};
      --btn-bg: {("#1f2937" if is_dark else "#ffffff")};
      --btn-text: {("#e5e7eb" if is_dark else "#0f172a")};
      --hover: {"rgba(255,255,255,0.10)" if is_dark else "rgba(0,0,0,0.06)"};
    }}
    .stApp {{ background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%); }}
    html, body, [data-testid="stAppViewContainer"] * {{ color: var(--text); }}
    [data-testid="stMetric"] {{
      background: var(--card); border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,{0.35 if is_dark else 0.08});
    }}
    button[role="tab"] {{ border-radius: 0 !important; color: var(--text) !important; }}
    button[role="tab"][aria-selected="true"] {{ background: var(--primary) !important; color: #fff !important; }}
    section[data-testid="stSidebar"] {{
      background: {("#0f172a" if is_dark else "#f8faff")}; border-right: 1px solid var(--border);
    }}
    header[data-testid="stHeader"], div[data-testid="stToolbar"] {{
      background: {("#0b1220" if is_dark else "#ffffff")} !important;
      color: var(--text) !important;
      border-bottom: 1px solid var(--border) !important;
    }}
    header[data-testid="stHeader"] * , div[data-testid="stToolbar"] * {{ color: var(--text) !important; fill: var(--text) !important; }}
    .stTextInput > div > div, div[data-baseweb="input"] {{
      background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
    }}
    .stTextInput input, div[data-baseweb="input"] input {{ color: var(--text) !important; }}
    .stTextInput input::placeholder, div[data-baseweb="input"] input::placeholder {{ color: var(--text-muted) !important; opacity: 1; }}
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] div[role="grid"],
    [data-testid="stDataFrame"] thead th,
    [data-testid="stDataFrame"] tbody td {{ background: var(--card) !important; }}
    [data-testid="stDataFrame"] thead th {{
      text-align: center !important; color: var(--text) !important; border-bottom: 1px solid var(--border) !important;
    }}
    [data-testid="stDataFrame"] tbody td div[data-testid="cell"] {{ justify-content: center; text-align: center; color: var(--text) !important; }}
    [data-testid="stDataFrame"] tbody td {{ border-bottom: 1px solid {("rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)")} !important; }}
    [data-testid="stDataFrame"] button,
    [data-testid="stDataFrame"] [role="button"] {{
      background: var(--btn-bg) !important; color: var(--btn-text) !important; border: 1px solid var(--border) !important; border-radius: 6px !important;
    }}
    [data-testid="stDownloadButton"] > button {{
      background: var(--primary) !important; color: #fff !important; border: 1px solid var(--primary) !important; border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] * {{ color: var(--text) !important; }}
    [data-testid="stFileUploaderDropzone"] {{ background: var(--card) !important; border: 1px dashed var(--border) !important; }}
    [data-testid="stFileUploader"] button {{
      background: var(--primary) !important; color: #fff !important; border: 1px solid var(--primary) !important; border-radius: 8px !important;
    }}
    label, .stSelectbox label, .stMultiSelect label {{ color: var(--text) !important; }}
    div[data-baseweb="input"] input::placeholder, textarea::placeholder {{ color: var(--text-muted) !important; opacity: 1; }}
    div[data-baseweb="select"] > div {{ color: var(--text) !important; background: var(--card) !important; border-color: var(--border) !important; }}
    body > div[data-baseweb="layer"] [data-baseweb="popover"],
    body > div[data-baseweb="layer"] [data-baseweb="menu"],
    body > div[data-baseweb="layer"] [role="listbox"],
    body > div[data-baseweb="layer"] ul[role="listbox"] {{
      background: var(--popover) !important; color: var(--text) !important; border: 1px solid var(--border) !important; box-shadow: 0 10px 30px rgba(0,0,0,0.12) !important;
    }}
    body > div[data-baseweb="layer"] [role="option"],
    body > div[data-baseweb="layer"] li[role="option"] {{ color: var(--text) !important; background: transparent !important; }}
    body > div[data-baseweb="layer"] [role="option"][aria-selected="true"],
    body > div[data-baseweb="layer"] [role="option"]:hover,
    body > div[data-baseweb="layer"] li[role="option"][aria-selected="true"],
    body > div[data-baseweb="layer"] li[role="option"]:hover {{ background: var(--hover) !important; }}
    body > div[data-baseweb="layer"] [role="listbox"]::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    body > div[data-baseweb="layer"] [role="listbox"]::-webkit-scrollbar-thumb {{ background: {("#384152" if is_dark else "#cbd5e1")} !important; border-radius: 6px; }}
    .vega-embed, .vega-embed canvas {{ background: var(--card) !important; }}
    .vega-embed:fullscreen, .vega-embed:-webkit-full-screen {{ background: var(--card) !important; }}
    .vega-actions a {{ background: var(--btn-bg) !important; color: var(--btn-text) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }}
    .vega-tooltip {{ background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }}
    [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] h4 {{ position: relative; }}
    [data-testid="stAppViewContainer"] h2::after, [data-testid="stAppViewContainer"] h3::after, [data-testid="stAppViewContainer"] h4::after {{
      content: ""; position: absolute; left: 0; bottom: -6px; height: 2px; width: 60px; background: var(--accent); border-radius: 1px;
    }}
    section[data-testid="stSidebar"] h2::after, section[data-testid="stSidebar"] h3::after, section[data-testid="stSidebar"] h4::after {{ content: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Altair Theme
# =============================================================================
# --- Altair Theme (RISE) ---
import altair as alt
@alt.theme.register("rise", enable=True)
def rise_theme():
    grid_col = "#334155" if is_dark else "#e5e7eb"
    chart_bg = "transparent" if is_dark else "#ffffff"
    return alt.theme.ThemeConfig({
        "config": {
            "background": chart_bg,
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFont": "Poppins", "titleFont": "Poppins",
                "grid": True, "domain": False,
                "gridColor": grid_col, "tickColor": grid_col,
                "labelColor": "#e5e7eb" if is_dark else "#111827",
                "titleColor": "#e5e7eb" if is_dark else "#111827",
            },
            "legend": {
                "labelFont": "Poppins", "titleFont": "Poppins",
                "labelColor": "#e5e7eb" if is_dark else "#111827",
                "titleColor": "#e5e7eb" if is_dark else "#111827",
            },
            "range": {
                "category": [
                    primary, accent, "#f59e0b", "#ef4444", "#14b8a6",
                    "#a78bfa", "#22c55e", "#eab308", "#06b6d4", "#fb7185",
                    "#7dd3fc", "#f472b6", "#93c5fd", "#60a5fa", "#34d399",
                    "#fde047", "#fca5a5", "#c4b5fd", "#86efac", "#fcd34d",
                ]
            },
        }
    })



# =============================================================================
# Data helpers
# =============================================================================
EXPECTED_SHEETS = {
    "policy_norms": ["policy norms"],
    "pd_fi_tvr_hca": ["pd fi tvr-hca", "pd fi tvr hca", "pd fi", "tvr-hca"],
    "kyc_loan_kits": ["kyc and loan kit", "kyc & loan kit", "kyc loan kit"],
    "prs_assessment": ["prs assessment", "pvr assessment"],
}

# Canonical column names (UPPER)
BASE_COLS_CANONICAL = [
    "QUARTER",
    "NAME OF AUDITOR",
    "CONTRACT NUMBER",
    "CONTRACT DATE",
    "STATE",
    "HUB",
    "LOCATION",
    "CUSTOMER NAME",
    "CUSTOMER CODE",
    "LOAN TYPE",
    "FINANCE AMOUNT",
    "ERROR",
    "QUERY",
]

def _norm_header(s: str) -> str:
    return " ".join(str(s or "").split()).upper()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_header(c) for c in df.columns]
    for base in BASE_COLS_CANONICAL:
        if base not in df.columns:
            df[base] = np.nan
    if "CONTRACT DATE" in df.columns:
        df["CONTRACT DATE"] = pd.to_datetime(df["CONTRACT DATE"], errors="coerce")
    if "FINANCE AMOUNT" in df.columns:
        df["FINANCE AMOUNT"] = pd.to_numeric(df["FINANCE AMOUNT"], errors="coerce")
    return df

def sheet_key_match(sheet_names: List[str]) -> Dict[str, str]:
    found = {}
    collapsed_map = {}
    for s in sheet_names:
        key = s.lower().replace("&", "and").replace("-", " ").replace("_", " ").strip()
        key = " ".join(key.split())
        collapsed_map[key] = s
    for key, variants in EXPECTED_SHEETS.items():
        for v in variants:
            for collapsed, original in collapsed_map.items():
                if v == collapsed:
                    found[key] = original
                    break
            if key in found:
                break
    return found

def melt_long(df: pd.DataFrame, sheet_label: str) -> pd.DataFrame:
    df = normalize_df(df)
    base_set = set(BASE_COLS_CANONICAL)
    issue_cols = [c for c in df.columns if c not in base_set]
    if not issue_cols:
        return pd.DataFrame(columns=["Sheet","Issue","Value","is_issue",*BASE_COLS_CANONICAL])
    long_df = df.melt(
        id_vars=[c for c in BASE_COLS_CANONICAL if c in df.columns],
        value_vars=issue_cols,
        var_name="Issue",
        value_name="Value",
    )
    def compute_is_issue(x):
        if pd.isna(x): return False
        s = str(x).strip()
        return bool(s) and (s.casefold() != "no query".casefold())
    long_df["is_issue"] = long_df["Value"].apply(compute_is_issue)
    long_df["Sheet"] = sheet_label
    return long_df

@st.cache_data(show_spinner=False)
def load_excel(uploaded_bytes: bytes, header_row: int = 3) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, list]]:
    excel = pd.ExcelFile(io.BytesIO(uploaded_bytes))
    key_to_actual = sheet_key_match(excel.sheet_names)

    sheets: Dict[str, pd.DataFrame] = {}
    missing_map: Dict[str, list] = {}

    for key, actual in key_to_actual.items():
        try:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=header_row, dtype=str)
        except Exception:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=header_row)

        tmp_cols = {_norm_header(c) for c in df_raw.columns}
        missing_required = [c for c in BASE_COLS_CANONICAL if c not in tmp_cols]

        df = normalize_df(df_raw)
        sheets[key] = df
        missing_map[key] = missing_required

    return sheets, key_to_actual, missing_map

def multi_select_or_all(label: str, series: pd.Series):
    vals = [v for v in sorted(series.dropna().astype(str).unique()) if v != "nan"]
    sel = st.multiselect(label, options=vals, default=[])
    return set(sel) if sel else set(vals)

def render_download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ----- Table helpers: TOTAL row + Sr. No. -----
def _prepend_total_and_srno(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    avg_spec: Optional[Tuple[str, str, str]] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        out = df.copy()
        out.insert(0, "Sr. No.", [])
        return out

    df2 = df.copy()
    num_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    total_row = {c: df2[c].sum() if c in num_cols else "" for c in df2.columns}

    if avg_spec:
        num_col, denom_col, target_col = avg_spec
        if num_col in df2.columns and denom_col in df2.columns and target_col in df2.columns:
            denom_sum = df2[denom_col].sum()
            num_sum = df2[num_col].sum()
            total_row[target_col] = round(num_sum / denom_sum, 2) if denom_sum else 0.0

    if label_col and label_col in df2.columns:
        total_row[label_col] = "TOTAL"

    df_with_total = pd.concat([pd.DataFrame([total_row]), df2], ignore_index=True)

    # Make Sr. No. a pure string column to satisfy Arrow
    sr = ["—"] + [str(i) for i in range(1, len(df2) + 1)]
    df_with_total.insert(0, "Sr. No.", sr)
    df_with_total["Sr. No."] = df_with_total["Sr. No."].astype(str)
    return df_with_total


def show_table(
    df: pd.DataFrame,
    label_col: Optional[str],
    download_name: str,
    avg_spec: Optional[Tuple[str, str, str]] = None,
    total_overrides: Optional[Dict[str, object]] = None,
):
    table = _prepend_total_and_srno(df, label_col=label_col, avg_spec=avg_spec)
    if total_overrides and len(table) > 0:
        for col, val in total_overrides.items():
            if col in table.columns:
                table.loc[0, col] = val
    if "Sr. No." in table.columns:
        table["Sr. No."] = table["Sr. No."].astype(str)     
    st.dataframe(table, use_container_width=True, hide_index=True)
    render_download_button(table, f"Download {download_name}", download_name)

# ----- Charts -----
def chart_bar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: Optional[str] = None, limit: int = 40):
    if df is None or df.empty:
        return
    data = df.head(limit)
    enc_color = alt.Color(color, legend=alt.Legend(title=color)) if color else alt.value(primary)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(x, sort="-y", title=x),
            y=alt.Y(y, title=y),
            color=enc_color,
            tooltip=list(data.columns),
        )
        .properties(height=420, title=title or "")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

def chart_heatmap(df: pd.DataFrame, x: str, y: str, value: str, title: str):
    if df is None or df.empty:
        return
    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(x, title=x),
            y=alt.Y(y, title=y),
            color=alt.Color(value, scale=alt.Scale(scheme="reds"), title=value),
            tooltip=list(df.columns),
        )
        .properties(height=420, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

def chart_donut(df: pd.DataFrame, category: str, value: str, title: str):
    if df is None or df.empty:
        return
    c = (
        alt.Chart(df)
        .mark_arc(innerRadius=70, outerRadius=160)
        .encode(theta=alt.Theta(f"{value}:Q"), color=alt.Color(f"{category}:N"), tooltip=list(df.columns))
        .properties(height=420, title=title)
    )
    st.altair_chart(c, use_container_width=True)

# =============================================================================
# File handling
# =============================================================================
if not uploaded:
    st.info("Upload your quarterly **Files Audit** workbook to begin.")
    st.stop()

with st.spinner("Reading workbook…"):
    sheets_map, names_map, missing_map = load_excel(uploaded.getvalue(), header_row=int(header_row))

if not sheets_map:
    st.error("Could not find expected sheets. Ensure the workbook contains: Policy Norms, PD FI TVR-HCA, KYC and Loan Kit, PRS/PVR Assessment.")
    st.stop()

with st.sidebar:
    st.subheader("Detected Sheets")
    for logical, actual in names_map.items():
        st.write(f"• **{logical}** → _{actual}_")

with st.sidebar:
    st.markdown("---")
    st.subheader("Required columns check")
    strict_mode = st.checkbox("Strict mode (stop on missing)", value=True)
    any_missing = False
    for k, actual in names_map.items():
        miss = missing_map.get(k, [])
        if miss:
            any_missing = True
            st.error(f"{actual}: missing {len(miss)} base column(s)")
            with st.expander(f"View missing in {actual}", expanded=False):
                st.write(", ".join(miss))
        else:
            st.success(f"{actual}: all base columns present")
    if strict_mode and any_missing:
        st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("Sheets to include")
    keys_sorted = list(sheets_map.keys())
    sheet_keys_selected = st.multiselect("Select sheets", options=keys_sorted, default=keys_sorted)

if not sheet_keys_selected:
    st.warning("Select at least one sheet.")
    st.stop()

long_frames = []
for k in sheet_keys_selected:
    label = names_map.get(k, k).strip()
    long_frames.append(melt_long(sheets_map[k], label))
long_all = pd.concat(long_frames, ignore_index=True)

# =============================================================================
# Filters
# =============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("Filters")

    quarters = multi_select_or_all("Quarter", long_all.get("QUARTER", pd.Series(dtype=str)))
    hubs = multi_select_or_all("Hub", long_all.get("HUB", pd.Series(dtype=str)))
    states = multi_select_or_all("State", long_all.get("STATE", pd.Series(dtype=str)))
    locations = multi_select_or_all("Location", long_all.get("LOCATION", pd.Series(dtype=str)))
    loan_types = multi_select_or_all("Loan type", long_all.get("LOAN TYPE", pd.Series(dtype=str)))
    auditors = multi_select_or_all("Name Of Auditor", long_all.get("NAME OF AUDITOR", pd.Series(dtype=str)))

    min_date = pd.to_datetime(long_all["CONTRACT DATE"], errors="coerce").min()
    max_date = pd.to_datetime(long_all["CONTRACT DATE"], errors="coerce").max()
    if pd.isna(min_date) or pd.isna(max_date):
        date_range = None
        st.caption("No valid Contract Date found to filter by date.")
    else:
        dr = st.date_input(
            "Contract Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        date_range = dr if isinstance(dr, tuple) else (dr, dr)

    query_text = st.text_input("Search Contract number / Customer code", key="search_text", value="").strip()

# Apply filters
flt = long_all.copy()
if quarters:
    flt = flt[flt["QUARTER"].astype(str).isin(quarters)]
if hubs:
    flt = flt[flt["HUB"].astype(str).isin(hubs)]
if states:
    flt = flt[flt["STATE"].astype(str).isin(states)]
if locations:
    flt = flt[flt["LOCATION"].astype(str).isin(locations)]
if loan_types:
    flt = flt[flt["LOAN TYPE"].astype(str).isin(loan_types)]
if auditors:
    flt = flt[flt["NAME OF AUDITOR"].astype(str).isin(auditors)]
if date_range:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    flt = flt[
        (pd.to_datetime(flt["CONTRACT DATE"], errors="coerce") >= start) &
        (pd.to_datetime(flt["CONTRACT DATE"], errors="coerce") <= end)
    ]
if query_text:
    q = query_text.casefold()
    flt = flt[
        (flt["CONTRACT NUMBER"].astype(str).str.casefold().str.contains(q, na=False)) |
        (flt["CUSTOMER CODE"].astype(str).str.casefold().str.contains(q, na=False))
    ]

# =============================================================================
# KPI Summary (customer-centric)
# =============================================================================
flt_codes = flt["CUSTOMER CODE"].dropna().astype(str)
unique_customers = int(flt_codes.nunique())
total_issue_cells = int(flt.loc[flt["is_issue"]].shape[0])
issues_per_customer = (
    flt.loc[flt["is_issue"]]
    .dropna(subset=["CUSTOMER CODE"])
    .assign(**{"CUSTOMER CODE": lambda d: d["CUSTOMER CODE"].astype(str)})
    .groupby("CUSTOMER CODE")
    .size()
)
customers_gt1 = int((issues_per_customer > 1).sum())
avg_issues_per_file_global = (total_issue_cells / unique_customers) if unique_customers else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Count of Customers", f"{unique_customers:,}")
m2.metric("Total Issues", f"{total_issue_cells:,}")
m3.metric("Customers with >1 issues", f"{customers_gt1:,}")
m4.metric("Avg issues per file", f"{avg_issues_per_file_global:.2f}")

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔎 Issue Counts", "🧮 Summary (Groupable)", "📊 Major Issue Distribution", "📅 Trend", "📥 Data & Exports"
])

group_options = {
    "Quarter": "QUARTER",
    "Hub Name": "HUB",
    "Product (Loan type)": "LOAN TYPE",
    "State": "STATE",
    "Location": "LOCATION",
    "Auditor": "NAME OF AUDITOR",
    "Sheet": "Sheet",
}

# -------------------- Tab 1 --------------------
with tab1:
    st.subheader("Issue counts (cells ≠ 'No Query')")
    counts_issue = (
        flt.loc[flt["is_issue"]]
        .groupby(["Issue"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["Count","Issue"], ascending=[False, True])
        .reset_index(drop=True)
    )

    counts_issue["Unique Files Assessed"] = unique_customers
    counts_issue["Avg Issues per File"] = (
        counts_issue["Count"] / counts_issue["Unique Files Assessed"].replace(0, np.nan)
    ).fillna(0).round(2)

    show_table(
        counts_issue,
        label_col="Issue",
        download_name="issue_counts.csv",
        total_overrides={
            "Unique Files Assessed": unique_customers,
            "Avg Issues per File": round(avg_issues_per_file_global, 2),
        },
    )

    st.markdown("#### Top issues (bar)")
    chart_bar(counts_issue, x="Issue", y="Count", color="Issue", title="Top Issues (current filters)", limit=30)

    st.markdown("#### Heatmap — Top issues by Hub")
    hub_issue = (
        flt.loc[flt["is_issue"]]
        .groupby(["HUB","Issue"], dropna=False)
        .size()
        .reset_index(name="Count")
    )
    top_issues = counts_issue.head(12)["Issue"].tolist()
    heat_df = hub_issue[hub_issue["Issue"].isin(top_issues)].copy()

    if not heat_df.empty:
        uniq_by_hub = (
            flt.groupby("HUB")["CUSTOMER CODE"]
            .apply(lambda s: s.dropna().astype(str).nunique())
            .reset_index(name="Unique Files Assessed")
        )
        heat_df = heat_df.merge(uniq_by_hub, on="HUB", how="left")
        heat_df["Avg Issues per File"] = (
            heat_df["Count"] / heat_df["Unique Files Assessed"].replace(0, np.nan)
        ).fillna(0).round(2)

        base = alt.Chart(heat_df)
        rect = base.mark_rect().encode(
            x=alt.X("HUB:N", title="Hub"),
            y=alt.Y("Issue:N", title="Issue"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="reds"), title="Issue Count"),
            tooltip=["HUB","Issue","Count","Unique Files Assessed","Avg Issues per File"]
        )
        txt_color = "#000000" if not is_dark else "#ffffff"
        text_lbl = base.mark_text(baseline="middle", fontSize=11, color=txt_color).encode(
            x="HUB:N", y="Issue:N", text="Count:Q"
        )
        st.altair_chart(rect + text_lbl, use_container_width=True)

        st.markdown("#### Heatmap — Avg issues per file (Hub × Issue)")
        avg_rect = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("HUB:N", title="Hub"),
                y=alt.Y("Issue:N", title="Issue"),
                color=alt.Color("Avg Issues per File:Q", scale=alt.Scale(scheme="blues"), title="Avg Issues / File"),
                tooltip=["HUB","Issue","Count","Unique Files Assessed","Avg Issues per File"],
            )
            .properties(height=420)
        )
        avg_text = (
            alt.Chart(heat_df)
            .mark_text(baseline="middle", fontSize=11, color=txt_color)
            .encode(x="HUB:N", y="Issue:N", text=alt.Text("Avg Issues per File:Q", format=".2f"))
        )
        st.altair_chart(avg_rect + avg_text, use_container_width=True)

# -------------------- Tab 2 --------------------
with tab2:
    st.subheader("Summary by chosen dimension")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        group_label = st.selectbox("Group by", options=list(group_options.keys()), index=0)
    with col_g2:
        secondary_label = st.selectbox("Secondary group by (optional)", options=["None"] + list(group_options.keys()), index=0)
    group_col = group_options[group_label]
    secondary_col = None if secondary_label == "None" else group_options[secondary_label]

    tmp = flt.copy()
    tmp[group_col] = tmp[group_col].fillna("Unknown")
    if secondary_col:
        tmp[secondary_col] = tmp[secondary_col].fillna("Unknown")
        group_fields = [group_col, secondary_col]
    else:
        group_fields = [group_col]

    uniq = (
        tmp.groupby(group_fields, dropna=False)["CUSTOMER CODE"]
        .apply(lambda s: s.dropna().astype(str).nunique())
        .rename("Count of Unique Cust Code")
    )
    err = (
        tmp.loc[tmp["is_issue"]]
        .groupby(group_fields, dropna=False)
        .size()
        .rename("Count of Errors")
    )
    summary = pd.concat([uniq, err], axis=1).fillna(0).reset_index()
    summary["Avg Error/Issue per file"] = (
        summary["Count of Errors"] / summary["Count of Unique Cust Code"].replace(0, np.nan)
    ).fillna(0).round(2)
    rename_map = {group_col: group_label}
    if secondary_col:
        rename_map[secondary_col] = secondary_label
    summary = summary.rename(columns=rename_map)
    sort_cols = ["Count of Errors", "Count of Unique Cust Code"]
    summary = summary.sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)

    avg_spec = ("Count of Errors", "Count of Unique Cust Code", "Avg Error/Issue per file")
    show_table(summary, label_col=group_label, download_name="summary_grouped.csv", avg_spec=avg_spec)

    st.markdown("#### Visual summary")
    if secondary_col:
        plot_df = summary.rename(columns={group_label: "Group", secondary_label: "Subgroup"})
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("Group:N", sort="-y", title=group_label),
                y=alt.Y("Count of Errors:Q"),
                color=alt.Color("Subgroup:N"),
                tooltip=list(plot_df.columns),
            )
            .properties(height=420, title=f"Errors by {group_label} and {secondary_label}")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        chart_bar(summary, x=group_label, y="Count of Errors", color=group_label, title=f"Errors by {group_label}")

# -------------------- Tab 3 --------------------
with tab3:
    st.subheader("Distribution of a Major Issue")
    major_issues = (
        flt.loc[flt["is_issue"], "Issue"]
        .dropna().astype(str).sort_values().unique().tolist()
    )
    if not major_issues:
        st.info("No issues found under current filters.")
    else:
        sel_issue = st.selectbox("Select an issue", options=major_issues)
        dist_dim_label = st.radio("Group by", options=["Quarter","Hub Name","Product (Loan type)","State","Location","Sheet"], horizontal=True)
        dist_dim = group_options[dist_dim_label]

        dist_df = (
            flt[(flt["is_issue"]) & (flt["Issue"] == sel_issue)]
            .assign(**{dist_dim: lambda d: d[dist_dim].fillna("Unknown")})
            .groupby(dist_dim, dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )

        uniq_by_group = (
            flt.assign(**{dist_dim: flt[dist_dim].fillna("Unknown")})
               .groupby(dist_dim)["CUSTOMER CODE"]
               .apply(lambda s: s.dropna().astype(str).nunique())
               .reset_index(name="Unique Files Assessed")
        )
        dist_df = dist_df.merge(uniq_by_group, on=dist_dim, how="left")
        dist_df["Avg Issues per File"] = (
            dist_df["Count"] / dist_df["Unique Files Assessed"].replace(0, np.nan)
        ).fillna(0).round(2)

        show_table(
            dist_df,
            label_col=dist_dim,
            download_name="issue_distribution.csv",
            total_overrides={
                "Unique Files Assessed": unique_customers,
                "Avg Issues per File": round(avg_issues_per_file_global, 2),
            },
        )

        chart_bar(dist_df, x=dist_dim, y="Count", color=dist_dim, title=f"Distribution of '{sel_issue}' by {dist_dim_label}")
        st.markdown("##### Donut view")
        chart_donut(dist_df, category=dist_dim, value="Count", title=f"{sel_issue} — Share by {dist_dim_label}")

# -------------------- Tab 4 --------------------
with tab4:
    st.subheader("Monthly trend of issues")
    if "CONTRACT DATE" in flt.columns:
        tmp_all = flt.copy()
        tmp_all["Month"] = pd.to_datetime(tmp_all["CONTRACT DATE"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        tmp_all = tmp_all.dropna(subset=["Month"])

        unique_files_by_month = (
            tmp_all.groupby("Month")["CUSTOMER CODE"]
            .apply(lambda s: s.dropna().astype(str).nunique())
            .rename("Unique Files Assessed")
        )
        issues_by_month = (
            tmp_all.loc[tmp_all["is_issue"]]
            .groupby("Month").size()
            .rename("Issue Count")
        )

        trend_df = pd.concat([unique_files_by_month, issues_by_month], axis=1).fillna(0).reset_index()
        trend_df["Avg Issues per File"] = (
            trend_df["Issue Count"] / trend_df["Unique Files Assessed"].replace(0, np.nan)
        ).fillna(0).round(2)

        show_table(
            trend_df,
            label_col="Month",
            download_name="monthly_trend.csv",
            total_overrides={
                "Unique Files Assessed": unique_customers,
                "Avg Issues per File": round(avg_issues_per_file_global, 2),
            },
        )

        if len(trend_df) >= 2:
            line = (
                alt.Chart(trend_df)
                .mark_line(strokeWidth=3, color=primary)
                .encode(
                    x=alt.X("Month:T", title="Month"),
                    y=alt.Y("Issue Count:Q"),
                    tooltip=["Month","Issue Count","Unique Files Assessed","Avg Issues per File"]
                )
                .properties(height=380, title="Issues per Month")
            )
            pts = (
                alt.Chart(trend_df)
                .mark_point(size=90, filled=True, color=accent)
                .encode(x="Month:T", y="Issue Count:Q")
            )
            st.altair_chart(line + pts, use_container_width=True)

            bars = (
                alt.Chart(trend_df)
                .mark_bar()
                .encode(
                    x=alt.X("Month:T", title="Month"),
                    y=alt.Y("Unique Files Assessed:Q"),
                    tooltip=["Month","Unique Files Assessed"]
                )
                .properties(height=220, title="Unique files assessed per Month")
            )
            st.altair_chart(bars, use_container_width=True)

        elif len(trend_df) == 1:
            chart_bar(trend_df, x="Month", y="Issue Count", color=None, title="Issues (single month)")
        else:
            st.info("No monthly data to plot.")
    else:
        st.info("No Contract Date available to compute a trend.")

# -------------------- Tab 5 --------------------
with tab5:
    st.subheader("Filtered long-format data")
    preview_cols = ["Sheet","Issue","Value","is_issue"] + [c for c in BASE_COLS_CANONICAL if c in flt.columns]
    long_view = flt[preview_cols].reset_index(drop=True)
    show_table(long_view, label_col="Sheet", download_name="filtered_long.csv")

    st.markdown("---")
    st.subheader("Per-sheet cleaned data (original wide format)")
    for k in sheet_keys_selected:
        label = names_map.get(k, k).strip()
        dfc = normalize_df(sheets_map[k]).reset_index(drop=True)
        st.markdown(f"**{label}**")
        show_table(dfc, label_col=None, download_name=f"{label.replace(' ','_').lower()}_cleaned.csv")
