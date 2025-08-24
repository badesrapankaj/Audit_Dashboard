from typing import Dict, List, Tuple, Optional
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ========== AUDIT ==========
EXPECTED_SHEETS_AUDIT = {
    "policy_norms": ["policy norms"],
    "pd_fi_tvr_hca": ["pd fi tvr-hca", "pd fi tvr hca", "pd fi", "tvr-hca"],
    "kyc_loan_kits": ["kyc and loan kit", "kyc & loan kit", "kyc loan kit"],
    "prs_assessment": ["prs assessment", "pvr assessment"],
}

BASE_COLS_AUDIT = [
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

def norm_header(s: str) -> str:
    return " ".join(str(s or "").split()).upper()

def normalize_df_audit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_header(c) for c in df.columns]
    for base in BASE_COLS_AUDIT:
        if base not in df.columns:
            df[base] = np.nan
    if "CONTRACT DATE" in df.columns:
        df["CONTRACT DATE"] = pd.to_datetime(df["CONTRACT DATE"], errors="coerce")
    if "FINANCE AMOUNT" in df.columns:
        df["FINANCE AMOUNT"] = pd.to_numeric(df["FINANCE AMOUNT"], errors="coerce")
    return df

def sheet_key_match(sheet_names: List[str]) -> Dict[str, str]:
    found: Dict[str, str] = {}
    collapsed_map: Dict[str, str] = {}
    for s in sheet_names:
        key = s.lower().replace("&", "and").replace("-", " ").replace("_", " ").strip()
        key = " ".join(key.split())
        collapsed_map[key] = s
    for key, variants in EXPECTED_SHEETS_AUDIT.items():
        for v in variants:
            for collapsed, original in collapsed_map.items():
                if v == collapsed:
                    found[key] = original
                    break
            if key in found:
                break
    return found

@st.cache_data(show_spinner=False)
def load_excel_audit(uploaded_bytes: bytes, header_row: int = 3) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, List[str]]]:
    excel = pd.ExcelFile(io.BytesIO(uploaded_bytes))
    key_to_actual = sheet_key_match(excel.sheet_names)

    sheets: Dict[str, pd.DataFrame] = {}
    missing_map: Dict[str, List[str]] = {}

    for key, actual in key_to_actual.items():
        try:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=header_row, dtype=str)
        except Exception:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=header_row)

        tmp_cols = {norm_header(c) for c in df_raw.columns}
        missing_required = [c for c in BASE_COLS_AUDIT if c not in tmp_cols]

        df = normalize_df_audit(df_raw)
        sheets[key] = df
        missing_map[key] = missing_required

    return sheets, key_to_actual, missing_map

def melt_long_audit(df: pd.DataFrame, sheet_label: str) -> pd.DataFrame:
    df = normalize_df_audit(df)
    base_set = set(BASE_COLS_AUDIT)
    issue_cols = [c for c in df.columns if c not in base_set]
    if not issue_cols:
        return pd.DataFrame(columns=["Sheet", "Issue", "Value", "is_issue", *BASE_COLS_AUDIT])
    long_df = df.melt(
        id_vars=[c for c in BASE_COLS_AUDIT if c in df.columns],
        value_vars=issue_cols,
        var_name="Issue",
        value_name="Value",
    )
    def _is_issue(x):
        if pd.isna(x):
            return False
        s = str(x).strip()
        return bool(s) and (s.casefold() != "no query".casefold())
    long_df["is_issue"] = long_df["Value"].apply(_is_issue)
    long_df["Sheet"] = sheet_label
    return long_df

# ==== OPS: sheets, required columns, loaders ====

EXPECTED_SHEETS_OPS = {
    "ta_recovery":      ["ta recovery", "ta_recovery", "ta"],
    "business_report":  ["business report", "business_report", "business"],
    "rep_verification": ["rep verification", "rep_verification", "rep verification "],
    "heat_map":         ["heat map", "heatmap", "heat map "],  # display only
}

BASE_COLS_OPS = {
    "ta_recovery": [
        "STATE","HUB","LOCATION","PRODUCT","DEALER NAME","TA QUERY",
    ],
    "business_report": [  # PRODUCT and DEALER NAME not mandatory here
        "STATE","HUB","LOCATION",
        "RC PENDENCY QUERY","INVOICE PENDING QUERY","FILE PENDENCY QUERY",
        "DEALER PAYMENT QUERY","COLLATERAL EXISTENCE QUERY","ROC CHARGE CREATION PENDINGQUERY",
    ],
    "rep_verification": [  # DEALER NAME not mandatory here
        "STATE","HUB","LOCATION","PRODUCT","QUERY","FINAL IA STATUS",
    ],
    # heat_map has no mandatory columns check
}

def sheet_key_match_ops(sheet_names: List[str]) -> Dict[str, str]:
    found, collapsed_map = {}, {}
    for s in sheet_names:
        key = s.lower().replace("&","and").replace("-"," ").replace("_"," ").strip()
        key = " ".join(key.split())
        collapsed_map[key] = s
    for key, variants in EXPECTED_SHEETS_OPS.items():
        for v in variants:
            for collapsed, original in collapsed_map.items():
                if v == collapsed:
                    found[key] = original
                    break
            if key in found:
                break
    return found

def normalize_df_ops(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_header(c) for c in df.columns]
    for base in required:
        if base not in df.columns:
            df[base] = np.nan
    return df

# auto-detect header row = 1 for TA/Business/Rep; heat_map handled separately
def _detect_header_row(excel_path_or_bytes, sheet_name: str, expected: List[str]) -> int:
    # Try the first 10 rows; succeed when all expected are present after normalization
    for hr in range(0, 10):
        try:
            df = pd.read_excel(excel_path_or_bytes, sheet_name=sheet_name, header=hr, nrows=1)
        except Exception:
            continue
        cols = [norm_header(c) for c in df.columns]
        if all(any(e == c for c in cols) for e in expected):
            return hr
    return 1  # sensible default for this workbook

@st.cache_data(show_spinner=False)
def load_excel_ops(uploaded_bytes: bytes, header_row: int = 1) -> Tuple[Dict[str,pd.DataFrame], Dict[str,str], Dict[str,List[str]]]:
    excel = pd.ExcelFile(io.BytesIO(uploaded_bytes))
    key_to_actual = sheet_key_match_ops(excel.sheet_names)

    sheets: Dict[str,pd.DataFrame] = {}
    names: Dict[str,str] = {}
    missing_map: Dict[str,List[str]] = {}

    for key, actual in key_to_actual.items():
        if key == "heat_map":
            # heat map is displayed raw from the sheet range; placeholder empty df here
            sheets[key] = pd.DataFrame()
            names[key] = actual
            missing_map[key] = []
            continue

        required = BASE_COLS_OPS.get(key, [])
        # Detect header row tailored to each sheet
        hr = _detect_header_row(io.BytesIO(uploaded_bytes), actual, required if required else [])
        try:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=hr, dtype=str)
        except Exception:
            df_raw = pd.read_excel(excel, sheet_name=actual, header=hr)

        tmp_cols = {norm_header(c) for c in df_raw.columns}
        missing_required = [c for c in required if c not in tmp_cols]

        df = normalize_df_ops(df_raw, required)
        sheets[key] = df
        names[key] = actual
        missing_map[key] = missing_required

    return sheets, names, missing_map

# ========== TABLES + CHARTS ==========
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
        if {num_col, denom_col, target_col}.issubset(df2.columns):
            denom_sum = df2[denom_col].sum()
            num_sum = df2[num_col].sum()
            total_row[target_col] = round(num_sum / denom_sum, 2) if denom_sum else 0.0
    if label_col and label_col in df2.columns:
        total_row[label_col] = "TOTAL"
    df_with_total = pd.concat([pd.DataFrame([total_row]), df2], ignore_index=True)
    sr = ["—"] + [str(i) for i in range(1, len(df2) + 1)]  # strings to avoid Arrow type error
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
    st.dataframe(table, use_container_width=True, hide_index=True)
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(label=f"Download {download_name}", data=csv, file_name=download_name, mime="text/csv")

def chart_bar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, title: Optional[str] = None, limit: int = 40):
    if df is None or df.empty:
        return
    data = df.head(limit)
    enc_color = alt.Color(color, legend=alt.Legend(title=color)) if color else alt.value("#0079b2")
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

# ========== ALTAIR THEME (RISE) ==========
@alt.theme.register("rise", enable=False)
def rise_theme():
    grid_col = "#334155"
    chart_bg = "transparent"
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": chart_bg,
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                    "grid": True,
                    "domain": False,
                    "gridColor": grid_col,
                    "tickColor": grid_col,
                    "labelColor": "#e5e7eb",
                    "titleColor": "#e5e7eb",
                },
                "legend": {
                    "labelFont": "Poppins",
                    "titleFont": "Poppins",
                    "labelColor": "#e5e7eb",
                    "titleColor": "#e5e7eb",
                },
                "range": {
                    "category": [
                        "#0079b2", "#312e81", "#f59e0b", "#ef4444", "#14b8a6",
                        "#a78bfa", "#22c55e", "#eab308", "#06b6d4", "#fb7185",
                    ]
                },
            }
        }
    )

def enable_rise_theme():
    alt.theme.enable("rise")
