# pages/00_Upload.py

import io
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from utils_shared import (
    EXPECTED_SHEETS_AUDIT,
    BASE_COLS_AUDIT,
    load_excel_audit,
    EXPECTED_SHEETS_OPS,
    BASE_COLS_OPS,
    load_excel_ops,
)

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("📤 Data Upload & Checks")

# ------------------- Upload widgets -------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Audit workbook (.xlsx)")
    up_audit = st.file_uploader(
        "Drop Files Audit workbook",
        type=["xlsx", "xlsm", "xls"],
        key="__up_audit__",
    )
    audit_header_row = st.number_input(
        "Header row (0-indexed, Audit)",
        min_value=0,
        max_value=50,
        value=3,
        step=1,
        key="__hdr_audit__",
    )

with colB:
    st.subheader("Ops workbook (.xlsx)")
    up_ops = st.file_uploader(
        "Drop Ops workbook",
        type=["xlsx", "xlsm", "xls"],
        key="__up_ops__",
    )
    st.caption("OPS headers are auto-detected per sheet. No row setting needed.")

st.markdown("---")

# ------------------- Mandatory columns -------------------
with st.expander("Mandatory columns (Audit)"):
    st.code(", ".join(BASE_COLS_AUDIT), language="text")

with st.expander("Mandatory columns (Ops)"):
    st.markdown("**TA Recovery**")
    st.code(", ".join(BASE_COLS_OPS["ta_recovery"]), language="text")
    st.markdown("**Business Report**")
    st.code(", ".join(BASE_COLS_OPS["business_report"]), language="text")
    st.markdown("**Rep Verification**")
    st.code(", ".join(BASE_COLS_OPS["rep_verification"]), language="text")
    st.caption("Heat Map sheet has no mandatory columns; it is displayed as-is.")

# ------------------- Persist + Validate: Audit -------------------
if up_audit is not None:
    st.session_state["audit_bytes"] = up_audit.getvalue()
    st.session_state["audit_header_row"] = int(audit_header_row)

    sheets_map, names_map, missing_map = load_excel_audit(
        st.session_state["audit_bytes"], header_row=int(audit_header_row)
    )

    st.subheader("Audit workbook checks")
    st.write("**Detected Sheets**")
    for logical, actual in names_map.items():
        st.write(f"• {logical} → _{actual}_")

    st.write("**Required columns check**")
    for k, actual in names_map.items():
        miss = missing_map.get(k, [])
        if miss:
            st.error(f"{actual}: missing {len(miss)} base column(s)")
            with st.expander(f"View missing in {actual}", expanded=False):
                st.write(", ".join(miss))
        else:
            st.success(f"{actual}: all base columns present")

# ------------------- Persist + Validate: Ops -------------------
if up_ops is not None:
    st.session_state["ops_bytes"] = up_ops.getvalue()

    ops_sheets, ops_names, ops_missing = load_excel_ops(
        st.session_state["ops_bytes"], header_row=1  # ignored by loader; auto-detects per sheet
    )

    st.subheader("Ops workbook checks")
    st.write("**Detected Sheets**")
    for logical, actual in ops_names.items():
        st.write(f"• {logical} → _{actual}_")

    st.write("**Required columns check**")
    for k, actual in ops_names.items():
        miss = ops_missing.get(k, [])
        if k == "heat_map":
            st.info(f"{actual}: no mandatory columns (display only)")
        else:
            if miss:
                st.error(f"{actual}: missing {len(miss)} base column(s)")
                with st.expander(f"View missing in {actual}", expanded=False):
                    st.write(", ".join(miss))
            else:
                st.success(f"{actual}: all base columns present")

st.markdown("---")

# ------------------- Go to dashboards -------------------
c1, c2 = st.columns([0.25, 0.75])
with c1:
    if st.button("Load Dashboards"):
        try:
            st.switch_page("app.py")
        except Exception:
            st.success("Files stored. Open the main page to view dashboards.")
            st.page_link("app.py", label="Open Dashboards", icon="📊")
