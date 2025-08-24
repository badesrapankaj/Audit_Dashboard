import io
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from utils_shared import (
    # Audit
    EXPECTED_SHEETS_AUDIT, BASE_COLS_AUDIT, norm_header,
    load_excel_audit, normalize_df_audit, melt_long_audit,
    # Shared UI utils
    show_table, chart_bar, chart_donut, rise_theme,
    # Ops
    EXPECTED_SHEETS_OPS, BASE_COLS_OPS, load_excel_ops, normalize_df_ops,
)

# ---------- Page + theme ----------
st.set_page_config(page_title="Dashboards", layout="wide", initial_sidebar_state="expanded")
alt.theme.register("rise", enable=True)(rise_theme)

# Hide Streamlit built-in page nav in sidebar so only filters remain
st.markdown("""
<style>
section[data-testid="stSidebar"] nav { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Header with view switcher ----------
left, right = st.columns([0.8, 0.2])
with right:
    view = st.segmented_control("View", options=["Audit","Operations"], default="Audit", key="__view__")
with left:
    st.title(f"📊 {view} Dashboard")

ss = st.session_state

# ---------- Guard: require Audit upload for both paths ----------
if "audit_bytes" not in ss or ss["audit_bytes"] is None:
    st.warning("Upload workbooks on the **Upload** page first.")
    st.page_link("pages/00_Upload.py", label="Open Upload page", icon="📤")
    st.stop()

# ---------- Load AUDIT bytes from session ----------
audit_sheets, audit_names, audit_missing = load_excel_audit(
    ss["audit_bytes"], header_row=int(ss.get("audit_header_row", 3))
)

# Build long table across all detected audit sheets
long_frames = []
for k, df in audit_sheets.items():
    label = audit_names.get(k, k)
    long_frames.append(melt_long_audit(df, label))
long_all = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

# ---------- Try load OPS (optional) ----------
ops_loaded = False
ops_sheets: Dict[str, pd.DataFrame] = {}
ops_names: Dict[str, str] = {}
ops_header_row = int(ss.get("ops_header_row", 3))
if "ops_bytes" in ss and ss["ops_bytes"] is not None:
    ops_sheets, ops_names, _ = load_excel_ops(ss["ops_bytes"], header_row=ops_header_row)
    ops_loaded = len(ops_sheets) > 0

# ---------- Small helpers (OPS) ----------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def nunique_nonblank(s: pd.Series) -> int:
    if s is None: return 0
    return s.dropna().astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).dropna().nunique()

def is_yes(x) -> bool:
    if pd.isna(x): return False
    s = str(x).strip().casefold()
    return s in {"yes","y","true","1"}

def filter_by_anycol(df: pd.DataFrame, cols: List[str], selected: List[str]) -> pd.DataFrame:
    if not selected: return df
    mask = None
    for c in cols:
        if c in df.columns:
            m = df[c].astype(str).isin(selected)
            mask = m if mask is None else (mask | m)
    return df if mask is None else df[mask]

# ============================= AUDIT VIEW =============================
if view == "Audit":
    # ---------------- Sidebar filters (Audit) ----------------
    with st.sidebar:
        st.header("Filters")

        sheets_available = sorted(long_all.get("Sheet", pd.Series(dtype=str)).dropna().astype(str).unique())
        sheets_selected = st.multiselect("Sheet", sheets_available)

        quarters   = st.multiselect("Quarter",   sorted(long_all.get("QUARTER", pd.Series(dtype=str)).dropna().astype(str).unique()))
        hubs       = st.multiselect("Hub",       sorted(long_all.get("HUB", pd.Series(dtype=str)).dropna().astype(str).unique()))
        states     = st.multiselect("State",     sorted(long_all.get("STATE", pd.Series(dtype=str)).dropna().astype(str).unique()))
        locations  = st.multiselect("Location",  sorted(long_all.get("LOCATION", pd.Series(dtype=str)).dropna().astype(str).unique()))
        loan_types = st.multiselect("Loan type", sorted(long_all.get("LOAN TYPE", pd.Series(dtype=str)).dropna().astype(str).unique()))
        auditors   = st.multiselect("Name Of Auditor", sorted(long_all.get("NAME OF AUDITOR", pd.Series(dtype=str)).dropna().astype(str).unique()))

        min_date = pd.to_datetime(long_all.get("CONTRACT DATE"), errors="coerce").min()
        max_date = pd.to_datetime(long_all.get("CONTRACT DATE"), errors="coerce").max()
        if pd.isna(min_date) or pd.isna(max_date):
            date_range = None
            st.caption("No valid Contract Date found to filter by date.")
        else:
            dr = st.date_input("Contract Date range",
                               value=(min_date.date(), max_date.date()),
                               min_value=min_date.date(), max_value=max_date.date())
            date_range = dr if isinstance(dr, tuple) else (dr, dr)

        query_text = st.text_input("Search Contract number / Customer code", value="").strip()

    # Apply filters
    flt = long_all.copy()
    if sheets_selected: flt = flt[flt["Sheet"].astype(str).isin(sheets_selected)]
    if quarters:        flt = flt[flt["QUARTER"].astype(str).isin(quarters)]
    if hubs:            flt = flt[flt["HUB"].astype(str).isin(hubs)]
    if states:          flt = flt[flt["STATE"].astype(str).isin(states)]
    if locations:       flt = flt[flt["LOCATION"].astype(str).isin(locations)]
    if loan_types:      flt = flt[flt["LOAN TYPE"].astype(str).isin(loan_types)]
    if auditors:        flt = flt[flt["NAME OF AUDITOR"].astype(str).isin(auditors)]
    if date_range:
        start = pd.to_datetime(date_range[0])
        end   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        cd = pd.to_datetime(flt["CONTRACT DATE"], errors="coerce")
        flt = flt[(cd >= start) & (cd <= end)]
    if query_text:
        q = query_text.casefold()
        flt = flt[
            (flt["CONTRACT NUMBER"].astype(str).str.casefold().str.contains(q, na=False)) |
            (flt["CUSTOMER CODE"].astype(str).str.casefold().str.contains(q, na=False))
        ]

    # ---------------- KPIs ----------------
    flt_codes = flt["CUSTOMER CODE"].dropna().astype(str)
    unique_customers = int(flt_codes.nunique())
    total_issue_cells = int(flt.loc[flt["is_issue"]].shape[0])
    issues_per_customer = (
        flt.loc[flt["is_issue"]]
          .dropna(subset=["CUSTOMER CODE"])
          .assign(**{"CUSTOMER CODE": lambda d: d["CUSTOMER CODE"].astype(str)})
          .groupby("CUSTOMER CODE").size()
    )
    customers_gt1 = int((issues_per_customer > 1).sum())
    avg_issues_per_file_global = (total_issue_cells / unique_customers) if unique_customers else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Count of Customers", f"{unique_customers:,}")
    m2.metric("Total Issues", f"{total_issue_cells:,}")
    m3.metric("Customers with >1 issues", f"{customers_gt1:,}")
    m4.metric("Avg issues per file", f"{avg_issues_per_file_global:.2f}")

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Issue Counts", "🧮 Summary (Groupable)", "📊 Major Issue Distribution", "📅 Trend", "📥 Data & Exports"
    ])

    group_options = {
        "Hub Name": "HUB",
        "Product (Loan type)": "LOAN TYPE",
        "State": "STATE",
        "Location": "LOCATION",
        "Auditor": "NAME OF AUDITOR",
        "Sheet": "Sheet",
        "Quarter": "QUARTER",
    }

    # ---- Tab 1
    with tab1:
        st.subheader("Issue counts (cells ≠ 'No Query')")
        counts_issue = (
            flt.loc[flt["is_issue"]]
               .groupby(["Issue"], dropna=False).size()
               .reset_index(name="Count")
               .sort_values(["Count","Issue"], ascending=[False, True])
               .reset_index(drop=True)
        )
        counts_issue["Unique Files Assessed"] = unique_customers
        counts_issue["Avg Issues per File"] = (
            counts_issue["Count"] / counts_issue["Unique Files Assessed"].replace(0, np.nan)
        ).fillna(0).round(2)
        show_table(
            counts_issue, label_col="Issue", download_name="issue_counts.csv",
            total_overrides={
                "Unique Files Assessed": unique_customers,
                "Avg Issues per File": round(avg_issues_per_file_global, 2),
            },
        )
        st.markdown("#### Top issues (bar)")
        chart_bar(counts_issue, x="Issue", y="Count", color="Issue",
                  title="Top Issues (current filters)", limit=30)

        st.markdown("#### Heatmap — Top issues by Hub")
        hub_issue = (
            flt.loc[flt["is_issue"]]
               .groupby(["HUB","Issue"], dropna=False).size()
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
            txt_color = "#ffffff"
            text_lbl = base.mark_text(baseline="middle", fontSize=11, color=txt_color).encode(
                x="HUB:N", y="Issue:N", text="Count:Q"
            )
            st.altair_chart(rect + text_lbl, use_container_width=True)

            st.markdown("#### Heatmap — Avg issues per file (Hub × Issue)")
            avg_rect = (
                alt.Chart(heat_df).mark_rect().encode(
                    x=alt.X("HUB:N", title="Hub"),
                    y=alt.Y("Issue:N", title="Issue"),
                    color=alt.Color("Avg Issues per File:Q", scale=alt.Scale(scheme="blues"), title="Avg Issues / File"),
                    tooltip=["HUB","Issue","Count","Unique Files Assessed","Avg Issues per File"],
                ).properties(height=420)
            )
            avg_text = (
                alt.Chart(heat_df).mark_text(baseline="middle", fontSize=11, color=txt_color)
                    .encode(x="HUB:N", y="Issue:N", text=alt.Text("Avg Issues per File:Q", format=".2f"))
            )
            st.altair_chart(avg_rect + avg_text, use_container_width=True)

    # ---- Tab 2
    with tab2:
        st.subheader("Summary by chosen dimension")
        c1, c2 = st.columns(2)
        with c1:
            group_label = st.selectbox("Group by", options=list(group_options.keys()), index=0)
        with c2:
            secondary_label = st.selectbox("Secondary group by (optional)", options=["None"] + list(group_options.keys()), index=0)
        group_col = group_options[group_label]
        secondary_col = None if secondary_label == "None" else group_options[secondary_label]

        tmp = flt.copy()
        tmp[group_col] = tmp[group_col].fillna("Unknown")
        group_fields = [group_col]
        if secondary_col:
            tmp[secondary_col] = tmp[secondary_col].fillna("Unknown")
            group_fields.append(secondary_col)

        uniq = (
            tmp.groupby(group_fields, dropna=False)["CUSTOMER CODE"]
               .apply(lambda s: s.dropna().astype(str).nunique())
               .rename("Count of Unique Cust Code")
        )
        err = (
            tmp.loc[tmp["is_issue"]]
               .groupby(group_fields, dropna=False).size()
               .rename("Count of Errors")
        )
        summary = pd.concat([uniq, err], axis=1).fillna(0).reset_index()
        summary["Avg Error/Issue per file"] = (
            summary["Count of Errors"] / summary["Count of Unique Cust Code"].replace(0, np.nan)
        ).fillna(0).round(2)

        rename_map = {group_col: group_label}
        if secondary_col:
            rename_map[secondary_col] = secondary_label
        summary = summary.rename(columns=rename_map)\
                         .sort_values(["Count of Errors","Count of Unique Cust Code"], ascending=[False, False])\
                         .reset_index(drop=True)

        avg_spec = ("Count of Errors", "Count of Unique Cust Code", "Avg Error/Issue per file")
        show_table(summary, label_col=group_label, download_name="summary_grouped.csv", avg_spec=avg_spec)

        st.markdown("#### Visual summary")
        if secondary_col:
            plot_df = summary.rename(columns={group_label: "Group", secondary_label: "Subgroup"})
            chart = (
                alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("Group:N", sort="-y", title=group_label),
                    y=alt.Y("Count of Errors:Q"),
                    color=alt.Color("Subgroup:N"),
                    tooltip=list(plot_df.columns),
                ).properties(height=420, title=f"Errors by {group_label} and {secondary_label}").interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            chart_bar(summary, x=group_label, y="Count of Errors", color=group_label,
                      title=f"Errors by {group_label}")

    # ---- Tab 3
    with tab3:
        st.subheader("Distribution of a Major Issue")
        major_issues = (
            flt.loc[flt["is_issue"], "Issue"].dropna().astype(str).sort_values().unique().tolist()
        )
        if not major_issues:
            st.info("No issues found under current filters.")
        else:
            sel_issue = st.selectbox("Select an issue", options=major_issues)
            dist_dim_label = st.radio(
                "Group by", options=["Hub Name","Product (Loan type)","State","Location","Sheet","Quarter"], horizontal=True
            )
            dist_dim = group_options[dist_dim_label]

            dist_df = (
                flt[(flt["is_issue"]) & (flt["Issue"] == sel_issue)]
                  .assign(**{dist_dim: lambda d: d[dist_dim].fillna("Unknown")})
                  .groupby(dist_dim, dropna=False).size()
                  .reset_index(name="Count")
                  .sort_values("Count", ascending=False).reset_index(drop=True)
            )

            uniq_by_group = (
                flt.assign(**{dist_dim: flt[dist_dim].fillna("Unknown")})
                   .groupby(dist_dim)["CUSTOMER CODE"]
                   .apply(lambda s: s.dropna().astype(str).nunique())
                   .reset_index(name="Unique Files Assessed")
            )
            dist_df = dist_df.merge(uniq_by_group, on=dist_dim, how="left")
            dist_df["Avg Issues per File"] = (
                dist_df["Count"] / dist_df["Unique Files Assessed"].replace(0, np.nan) * 100
            ).fillna(0).round(2)

            show_table(
                dist_df, label_col=dist_dim, download_name="issue_distribution.csv",
                total_overrides={
                    "Unique Files Assessed": int(flt['CUSTOMER CODE'].dropna().astype(str).nunique()),
                    "Avg Issues per File": round(
                        (int(flt.loc[flt['is_issue']].shape[0]) /
                         max(1, int(flt['CUSTOMER CODE'].dropna().astype(str).nunique())))*100, 2),
                },
            )

            chart_bar(dist_df, x=dist_dim, y="Count", color=dist_dim,
                      title=f"Distribution of '{sel_issue}' by {dist_dim_label}")
            st.markdown("##### Donut view")
            chart_donut(dist_df, category=dist_dim, value="Count",
                        title=f"{sel_issue} — Share by {dist_dim_label}")

    # ---- Tab 4
    with tab4:
        st.subheader("Monthly trend of issues")
        if "CONTRACT DATE" in flt.columns:
            tmp_all = flt.copy()
            tmp_all["Month"] = pd.to_datetime(tmp_all["CONTRACT DATE"], errors="coerce")\
                                  .dt.to_period("M").dt.to_timestamp()
            tmp_all = tmp_all.dropna(subset=["Month"])

            unique_files_by_month = (
                tmp_all.groupby("Month")["CUSTOMER CODE"]
                       .apply(lambda s: s.dropna().astype(str).nunique())
                       .rename("Unique Files Assessed")
            )
            issues_by_month = (
                tmp_all.loc[tmp_all["is_issue"]].groupby("Month").size().rename("Issue Count")
            )
            trend_df = pd.concat([unique_files_by_month, issues_by_month], axis=1)\
                         .fillna(0).reset_index()
            trend_df["Avg Issues per File"] = (
                trend_df["Issue Count"] / trend_df["Unique Files Assessed"].replace(0, np.nan) * 100
            ).fillna(0).round(2)

            show_table(
                trend_df, label_col="Month", download_name="monthly_trend.csv",
                total_overrides={
                    "Unique Files Assessed": int(flt['CUSTOMER CODE'].dropna().astype(str).nunique()),
                    "Avg Issues per File": round(
                        (int(flt.loc[flt['is_issue']].shape[0]) /
                         max(1, int(flt['CUSTOMER CODE'].dropna().astype(str).nunique())))*100, 2),
                },
            )

            if len(trend_df) >= 2:
                line = alt.Chart(trend_df).mark_line(strokeWidth=3).encode(
                    x=alt.X("Month:T", title="Month"),
                    y=alt.Y("Issue Count:Q"),
                    tooltip=["Month","Issue Count","Unique Files Assessed","Avg Issues per File"]
                ).properties(height=380, title="Issues per Month")
                pts = alt.Chart(trend_df).mark_point(size=90, filled=True).encode(
                    x="Month:T", y="Issue Count:Q"
                )
                st.altair_chart(line + pts, use_container_width=True)
            elif len(trend_df) == 1:
                chart_bar(trend_df, x="Month", y="Issue Count", title="Issues (single month)")
            else:
                st.info("No monthly data to plot.")
        else:
            st.info("No Contract Date available to compute a trend.")

    # ---- Tab 5
    with tab5:
        st.subheader("Filtered long-format data")
        preview_cols = ["Sheet","Issue","Value","is_issue"] + [c for c in BASE_COLS_AUDIT if c in flt.columns]
        long_view = flt[preview_cols].reset_index(drop=True)
        show_table(long_view, label_col="Sheet", download_name="filtered_long.csv")

# ============================= OPERATIONS VIEW =============================
else:
    if not ops_loaded:
        st.info("Upload the Ops workbook on the **Upload** page.")
        st.page_link("pages/00_Upload.py", label="Open Upload page", icon="📤")
        st.stop()

    # ---------- Sidebar filters (Ops) ----------
    with st.sidebar:
        st.header("Filters")

        # Sheet selector shows actual names from the uploaded Ops file
        ops_labels = [ops_names[k] for k in ops_sheets.keys()]
        ops_label_to_key = {ops_names[k]: k for k in ops_sheets.keys()}
        ops_selected_labels = st.multiselect("OPS Sheets", options=ops_labels, default=ops_labels)

        # Build union lists for filters across selected sheets
        def union_vals_any(cols: List[str]) -> List[str]:
            vals = []
            for lbl in ops_selected_labels:
                key = ops_label_to_key[lbl]
                df = ops_sheets.get(key, pd.DataFrame())
                for c in cols:
                    if c in df.columns:
                        vals.extend(df[c].dropna().astype(str).unique().tolist())
            return sorted(set([v for v in vals if v != "nan"]))

        states_sel   = st.multiselect("State",    union_vals_any(["STATE"]))
        hubs_sel     = st.multiselect("Hub",      union_vals_any(["HUB"]))
        locs_sel     = st.multiselect("Location", union_vals_any(["LOCATION"]))
        # Product filter should work for PRODUCT / PRODUCT CATEGORY / LOAN TYPE
        prods_sel    = st.multiselect("Product",  union_vals_any(["PRODUCT","PRODUCT CATEGORY","LOAN TYPE"]))
        dealers_sel  = st.multiselect("Dealer",   union_vals_any(["DEALER NAME"]))

    # Helper to apply ops filters safely
    def apply_ops_filters(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "STATE" in out.columns and states_sel:
            out = out[out["STATE"].astype(str).isin(states_sel)]
        if "HUB" in out.columns and hubs_sel:
            out = out[out["HUB"].astype(str).isin(hubs_sel)]
        if "LOCATION" in out.columns and locs_sel:
            out = out[out["LOCATION"].astype(str).isin(locs_sel)]
        if prods_sel:
            out = filter_by_anycol(out, ["PRODUCT","PRODUCT CATEGORY","LOAN TYPE"], prods_sel)
        if "DEALER NAME" in out.columns and dealers_sel:
            out = out[out["DEALER NAME"].astype(str).isin(dealers_sel)]
        return out

    # Tabs per Ops sheet
    tabs = st.tabs(["🧾 TA Recovery", "📈 Business Report", "🧪 Rep Verification", "🗺️ Heat Map"])

    # ---------- TA Recovery ----------
    with tabs[0]:
        key = "ta_recovery"
        if ops_names.get(key) in ops_selected_labels and key in ops_sheets:
            df = ops_sheets[key]
            df = normalize_df_ops(df, BASE_COLS_OPS.get(key, []))
            df = apply_ops_filters(df)

            ta_id_col = pick_col(df, ["TA NO","TA NO.","TA NUMBER","TA#","TANO","TA NUM"])
            yes_mask = df["TA QUERY"].apply(is_yes) if "TA QUERY" in df.columns else pd.Series(False, index=df.index)

            total_unique_ta = nunique_nonblank(df[ta_id_col]) if ta_id_col else len(df)
            yes_unique_ta   = nunique_nonblank(df.loc[yes_mask, ta_id_col]) if ta_id_col else int(yes_mask.sum())
            yes_pct = (yes_unique_ta / total_unique_ta * 100) if total_unique_ta else 0.0

            c1,c2,c3 = st.columns(3)
            c1.metric("Unique TA No.", f"{total_unique_ta:,}")
            c2.metric("TA Query = YES", f"{yes_unique_ta:,}")
            c3.metric("YES %", f"{yes_pct:.1f}%")

            # Group by selector
            group_map = {"State":"STATE","Hub":"HUB","Location":"LOCATION","Product":"PRODUCT","Dealer":"DEALER NAME"}
            group_choice = st.selectbox("Group by", options=list(group_map.keys()), index=1)
            gcol = group_map[group_choice]

            if gcol in df.columns and ta_id_col:
                # YES Count = unique TA with YES per group
                yes_group = (
                    df.loc[yes_mask, [gcol, ta_id_col]]
                      .assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                      .dropna(subset=[ta_id_col])
                      .astype({ta_id_col: str})
                      .drop_duplicates([gcol, ta_id_col])
                      .groupby(gcol).size().reset_index(name="YES Count")
                )
                # Total Rows = unique TA per group
                total_group = (
                    df[[gcol, ta_id_col]]
                      .assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                      .dropna(subset=[ta_id_col])
                      .astype({ta_id_col: str})
                      .drop_duplicates([gcol, ta_id_col])
                      .groupby(gcol).size().reset_index(name="Total Rows")
                )
                by_group = total_group.merge(yes_group, on=gcol, how="left").fillna({"YES Count":0})
                by_group["YES Count"] = by_group["YES Count"].astype(int)
                by_group["YES %"] = (by_group["YES Count"] / by_group["Total Rows"].replace(0, np.nan) * 100).fillna(0).round(2)
                by_group = by_group.sort_values(["YES Count","Total Rows"], ascending=[False, False]).reset_index(drop=True)

                show_table(
                    by_group, label_col=gcol, download_name="ta_recovery_yes_by_group.csv",
                    total_overrides={
                        "Total Rows": total_unique_ta,
                        "YES Count": yes_unique_ta,
                        "YES %": round(yes_pct, 2),
                    },
                )
                chart_bar(by_group, x=gcol, y="YES Count", color=gcol, title=f"TA YES by {group_choice}")
            else:
                st.info(f"'{gcol}' or TA number column not present in TA Recovery sheet.")
        else:
            st.info("TA Recovery not selected.")

    # ---------- Business Report ----------
    with tabs[1]:
        key = "business_report"
        if ops_names.get(key) in ops_selected_labels and key in ops_sheets:
            df = ops_sheets[key]
            df = normalize_df_ops(df, BASE_COLS_OPS.get(key, []))
            df = apply_ops_filters(df)

            contract_col = pick_col(df, ["CONTRACT","CONTRACT NO","CONTRACT NO.","CONTRACT NUMBER","CONTRACT#"])
            prod_cat_col = pick_col(df, ["PRODUCT CATEGORY","PRODUCT"])  # prefer PRODUCT CATEGORY
            query_cols = [c for c in df.columns if c.endswith("QUERY")]

            total_unique_contracts = nunique_nonblank(df[contract_col]) if contract_col else len(df)

            # Mini dashboard (ANY YES across queries, unique contracts)
            if contract_col and query_cols:
                yes_any_mask = pd.Series(False, index=df.index)
                for qc in query_cols:
                    yes_any_mask = yes_any_mask | df[qc].apply(is_yes)
                yes_unique_contracts_any = nunique_nonblank(df.loc[yes_any_mask, contract_col])
            else:
                yes_unique_contracts_any = 0
            yes_pct_any = (yes_unique_contracts_any / total_unique_contracts * 100) if total_unique_contracts else 0.0

            c1,c2,c3 = st.columns(3)
            c1.metric("Unique Contracts", f"{total_unique_contracts:,}")
            c2.metric("Contracts with ANY YES", f"{yes_unique_contracts_any:,}")
            c3.metric("YES % (ANY)", f"{yes_pct_any:.1f}%")

            st.subheader("Business Queries — YES counts by issue")
            if not query_cols:
                st.info("No *QUERY columns* found.")
            else:
                rows = []
                for qc in sorted(query_cols):
                    if contract_col:
                        ids_yes = (
                            df.loc[df[qc].apply(is_yes), [contract_col]]
                              .dropna().astype(str).drop_duplicates()[contract_col]
                        )
                        yes_count = int(ids_yes.nunique())
                        total_rows = total_unique_contracts  # denominator = unique contracts
                    else:
                        yes_count = int(df[qc].apply(is_yes).sum())
                        total_rows = int(len(df))
                    rows.append({
                        "Issue": qc,
                        "YES Count": yes_count,
                        "Total Rows": total_rows,
                        "YES %": round((yes_count / total_rows * 100), 2) if total_rows else 0.0
                    })
                issues_df = pd.DataFrame(rows).sort_values("YES Count", ascending=False).reset_index(drop=True)

                # Override TOTAL row so it shows true unique denominator and ANY-YES percentage
                show_table(
                    issues_df, label_col="Issue", download_name="business_report_yes_counts.csv",
                    total_overrides={
                        "Total Rows": total_unique_contracts,
                        "YES Count": yes_unique_contracts_any,
                        "YES %": round(yes_pct_any, 2),
                    },
                )
                chart_bar(issues_df, x="Issue", y="YES Count", color="Issue", title="YES by Business Issue", limit=30)

                # Distribution for a selected issue (unique contracts & percent)
                st.markdown("#### Distribution of a selected Business issue")
                sel_issue = st.selectbox("Select issue", options=issues_df["Issue"].tolist())
                group_map = {
                    "State":"STATE","Hub":"HUB","Location":"LOCATION",
                    "Product Category": prod_cat_col or "PRODUCT CATEGORY","Dealer":"DEALER NAME"
                }
                group_opts = {k:v for k,v in group_map.items() if v is not None}
                group_choice = st.selectbox("Group by", options=list(group_opts.keys()), index=0, key="biz_grp")
                gcol = group_opts[group_choice]

                if sel_issue in df.columns and gcol in df.columns and contract_col:
                    sub_yes = df[df[sel_issue].apply(is_yes)].copy()

                    # YES unique contracts per group
                    yes_grp = (
                        sub_yes[[gcol, contract_col]].assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                              .dropna().astype({contract_col: str})
                              .drop_duplicates([gcol, contract_col])
                              .groupby(gcol).size().reset_index(name="YES Count")
                    )
                    # Total unique contracts per group
                    total_grp = (
                        df[[gcol, contract_col]].assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                          .dropna().astype({contract_col: str})
                          .drop_duplicates([gcol, contract_col])
                          .groupby(gcol).size().reset_index(name="Total Rows")
                    )
                    dist = total_grp.merge(yes_grp, on=gcol, how="left").fillna({"YES Count":0})
                    dist["YES Count"] = dist["YES Count"].astype(int)
                    dist["YES %"] = (dist["YES Count"] / dist["Total Rows"].replace(0, np.nan) * 100).fillna(0).round(2)
                    dist = dist.sort_values(["YES Count","Total Rows"], ascending=[False, False]).reset_index(drop=True)

                    # Total overrides for selected issue
                    yes_count_sel_total = nunique_nonblank(sub_yes[contract_col])
                    show_table(
                        dist, label_col=gcol, download_name="business_issue_distribution.csv",
                        total_overrides={
                            "Total Rows": total_unique_contracts,
                            "YES Count": yes_count_sel_total,
                            "YES %": round((yes_count_sel_total / total_unique_contracts * 100) if total_unique_contracts else 0.0, 2),
                        },
                    )
                    chart_bar(dist, x=gcol, y="YES Count", color=gcol,
                              title=f"YES for '{sel_issue}' by {group_choice}")
                else:
                    st.info("Chosen group or issue/contract column not present.")
        else:
            st.info("Business Report not selected.")

    # ---------- Rep Verification ----------
    with tabs[2]:
        key = "rep_verification"
        if ops_names.get(key) in ops_selected_labels and key in ops_sheets:
            df = ops_sheets[key]
            df = normalize_df_ops(df, BASE_COLS_OPS.get(key, []))
            df = apply_ops_filters(df)

            contract_col = pick_col(df, ["CONTRACT NO","CONTRACT NO.","CONTRACT NUMBER","CONTRACT","CONTRACT#"])
            product_as_loan = pick_col(df, ["LOAN TYPE"])  # Product dimension = Loan Type
            yes_mask = df["QUERY"].apply(is_yes) if "QUERY" in df.columns else pd.Series(False, index=df.index)

            total_unique_contracts = nunique_nonblank(df[contract_col]) if contract_col else len(df)
            yes_unique_contracts   = nunique_nonblank(df.loc[yes_mask, contract_col]) if contract_col else int(yes_mask.sum())
            yes_pct = (yes_unique_contracts / total_unique_contracts * 100) if total_unique_contracts else 0.0

            col1, col2, col3 = st.columns(3)
            col1.metric("Unique Contract No.", f"{total_unique_contracts:,}")
            col2.metric("QUERY = YES (unique)", f"{yes_unique_contracts:,}")
            col3.metric("YES %", f"{yes_pct:.1f}%")

            # Group distribution (unique contracts)
            group_map = {
                "State":"STATE","Hub":"HUB","Location":"LOCATION",
                "Product (Loan Type)": product_as_loan or "LOAN TYPE",
                "Agency":"AGENCY","Name of Yard":"NAME OF YARD"
            }
            group_opts = {k:v for k,v in group_map.items() if v is not None}
            group_choice = st.selectbox("Group YES by", options=list(group_opts.keys()), index=1, key="rep_grp")
            gcol = group_opts[group_choice]

            if gcol in df.columns and contract_col:
                yes_df = df[yes_mask].copy()
                yes_grp = (
                    yes_df[[gcol, contract_col]].assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                          .dropna().astype({contract_col: str})
                          .drop_duplicates([gcol, contract_col])
                          .groupby(gcol).size().reset_index(name="YES Count")
                )
                total_grp = (
                    df[[gcol, contract_col]].assign(**{gcol: lambda d: d[gcol].fillna("Unknown")})
                      .dropna().astype({contract_col: str})
                      .drop_duplicates([gcol, contract_col])
                      .groupby(gcol).size().reset_index(name="Total Rows")
                )
                dist = total_grp.merge(yes_grp, on=gcol, how="left").fillna({"YES Count":0})
                dist["YES Count"] = dist["YES Count"].astype(int)
                dist["YES %"] = (dist["YES Count"] / dist["Total Rows"].replace(0, np.nan) * 100).fillna(0).round(2)
                dist = dist.sort_values(["YES Count","Total Rows"], ascending=[False, False]).reset_index(drop=True)

                show_table(
                    dist, label_col=gcol, download_name="rep_query_yes_by_group.csv",
                    total_overrides={
                        "Total Rows": total_unique_contracts,
                        "YES Count": yes_unique_contracts,
                        "YES %": round(yes_pct, 2),
                    },
                )
                chart_bar(dist, x=gcol, y="YES Count", color=gcol, title=f"QUERY YES by {group_choice}")
            else:
                st.info("Chosen group or Contract column not present.")

            # Final IA Status distribution
            if "FINAL IA STATUS" in df.columns:
                sta = (
                    df["FINAL IA STATUS"].fillna("Unknown").astype(str).value_counts()
                      .reset_index().rename(columns={"index":"Status","FINAL IA STATUS":"Count"})
                )
                st.markdown("#### Final IA Status distribution")
                show_table(sta, label_col="Status", download_name="rep_final_ia_status.csv")
                chart_donut(sta, category="Status", value="Count", title="Final IA Status")
            else:
                st.info("Column 'FINAL IA STATUS' not found.")
        else:
            st.info("Rep Verification not selected.")

    # ---------- Heat Map ----------
    with tabs[3]:
        key = "heat_map"
        if ops_names.get(key) in ops_selected_labels and key in ops_sheets and "ops_bytes" in ss:
            st.subheader("Heat Map — Full sheet view")
            xls = pd.ExcelFile(io.BytesIO(ss["ops_bytes"]))
            sheet_name = ops_names.get(key)
            try:
                # Show the entire sheet as-is
                heat_full = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                st.dataframe(heat_full, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not render entire sheet '{sheet_name}'.")
                st.caption(str(e))
        else:
            st.info("Heat Map not selected.")
