import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------------- PAGE CONTROL -----------------------
st.set_page_config(page_title="Supply Forecast Tool", layout="wide")

if "supply_page" not in st.session_state:
    st.session_state.supply_page = "config"

# ------------------------------ PAGE 1 - SUPPLY CONFIGURATION ------------------------------
if st.session_state.supply_page == "config":
    st.title("🚚 Supply Forecast Tool")

    st.markdown("""
    ### 📄 File Format Instructions

    Please upload a **CSV file** in the following format:

    | Product Name | Product ID | Sales Date | Sales Quantity | Current Stock |
    |--------------|------------|------------|----------------|----------------|
    | Apple        | 123456     | 06.05.2024 | 1000           | 500            |

    - ✅ Accepted file type: `.csv`
    - 📂 Required columns: `Product Name`, `Product ID`, `Sales Date`, `Sales Quantity`, `Current Stock`
    """)

    uploaded_file = st.file_uploader("📂 Upload your sales CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            st.write("#### ✅ Preview of Uploaded Data")
            st.dataframe(df.head())

            with st.expander("📌 Select Columns"):
                name_col = st.selectbox("Select the column for 'Product Name':", df.columns)
                id_col = st.selectbox("Select the column for 'Product ID':", df.columns)
                date_col = st.selectbox("Select the column for 'Sales Date':", df.columns)
                qty_col = st.selectbox("Select the column for 'Sales Quantity':", df.columns)
                stock_col = st.selectbox("Select the column for 'Current Stock':", df.columns)

            df = df.rename(columns={
                name_col: "Product_Name",
                id_col: "Product_ID",
                date_col: "Sales_Date",
                qty_col: "Sales_Quantity",
                stock_col: "Current_Stock"
            })

            df["Sales_Date"] = pd.to_datetime(df["Sales_Date"], errors="coerce")
            df["Sales_Quantity"] = pd.to_numeric(df["Sales_Quantity"], errors="coerce")
            df = df.dropna(subset=["Sales_Date", "Sales_Quantity"])
            df["Sales_Quantity"] = df["Sales_Quantity"].astype(int)

            df = df.sort_values("Sales_Date")
            df["Week_Num"] = df["Sales_Date"].rank(method="dense").astype(int)
            df["Week_Num"] = df["Week_Num"] - df["Week_Num"].min() + 1  # Start from 1

            df = df.groupby(["Week_Num", "Product_ID", "Product_Name"], as_index=False).agg({
                "Sales_Quantity": "sum",
                "Current_Stock": "last"
            })

            st.session_state.supply_df = df
            st.success("✅ File loaded and prepared successfully! Now configure your supply forecast.")

        except Exception as e:
            st.error("❌ Could not process the uploaded file. Please check the format.")

    st.markdown("---")
    st.markdown("## 🔮 Supply Forecast Configuration")

    lead_time = st.number_input("Enter supplier lead time (in weeks):", min_value=1, max_value=12, value=3)
    stock_cover = st.number_input("Desired stock coverage (in weeks):", min_value=1, max_value=12, value=4)
    forecast_weeks = st.selectbox("Forecast how many weeks ahead?", [1, 2, 4, 8, 12], index=2)

    if st.button("Run Supply Forecast"):
        if "supply_df" not in st.session_state:
            st.warning("⚠️ Please upload and process the sales data first.")
            st.stop()

        df = st.session_state.supply_df.copy()

        result_df = pd.DataFrame()
        for pid, group in df.groupby("Product_ID"):
            product_name = group["Product_Name"].iloc[0]
            stock = group["Current_Stock"].iloc[-1]
            weekly_sales = group["Sales_Quantity"].mean()
            stock_cov_weeks = round(stock / weekly_sales, 1) if weekly_sales > 0 else 0
            projected_stock = max(stock - weekly_sales * lead_time, 0)
            forecast_qty = int((lead_time + stock_cover) * weekly_sales - stock)

            if stock_cov_weeks < 1:
                risk = "High Stock Out Risk"
            elif stock_cov_weeks < 2:
                risk = "Alert"
            elif stock_cov_weeks < 4:
                risk = "Need to Order"
            else:
                risk = "OK"

            result_df = pd.concat([result_df, pd.DataFrame({
                "Product_ID": [pid],
                "Product_Name": [product_name],
                "Weekly_Sales_Quantity": [int(round(weekly_sales))],
                "Current_Stock": [int(stock)],
                "Current_Stock_Coverage_Weeks": [stock_cov_weeks],
                "Supplier_Lead_Time (weeks)": [lead_time],
                "Projected_Stock_at_Arrival": [int(projected_stock)],
                "Suggested_Order_Quantity": [max(forecast_qty, 0)],
                "Risk_Level": [risk]
            })])

        st.session_state.supply_result_df = result_df
        st.session_state.supply_page = "result"

# ------------------------------ PAGE 2 - SUPPLY RESULT ------------------------------
elif st.session_state.supply_page == "result":
    st.title("🚚 Supply Forecast Result")

    df = st.session_state.supply_df
    result_df = st.session_state.supply_result_df

    # Forecast result table preview (top section)
    st.subheader("🔍 Supply Forecast Table")
    product_list = sorted(result_df["Product_Name"].unique().tolist())
    selected_product_table = st.selectbox("Filter forecast table by product:", ["All Products"] + product_list)

    if selected_product_table == "All Products":
        filtered_result = result_df
    else:
        filtered_result = result_df[result_df["Product_Name"] == selected_product_table]

    st.dataframe(filtered_result.head(100))

    # Download option
    download_format = st.selectbox("Download Format:", ["CSV", "Excel"])
    if download_format == "CSV":
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast (CSV)", data=csv, file_name="supply_forecast.csv", mime="text/csv")
    elif download_format == "Excel":
        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Supply Forecast")
            st.download_button("⬇️ Download Forecast (Excel)", data=buffer.getvalue(),
                               file_name="supply_forecast.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ModuleNotFoundError:
            st.error("❌ Excel export failed. Please install `xlsxwriter`: `pip install xlsxwriter`")

    # Weekly Sales Trend (bottom section)
    st.markdown("### 📈 Weekly Sales Trend")
    product_list = sorted(df["Product_Name"].unique().tolist())
    selected_graph_product = st.selectbox("Select a product to visualize trend:", product_list)

    product_df = df[df["Product_Name"] == selected_graph_product].copy()
    product_df = product_df.sort_values("Week_Num")
    product_df["Week_Label"] = product_df["Week_Num"].apply(lambda x: f"W{x}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=product_df["Week_Label"],
        y=product_df["Sales_Quantity"],
        mode="lines+markers+text",
        name=selected_graph_product,
        text=product_df["Sales_Quantity"],
        textposition="top center",
        marker=dict(size=8),
        line=dict(width=3)
    ))

    fig.update_layout(
        title=f"Weekly Sales Trend: {selected_graph_product}",
        xaxis_title="Week",
        yaxis_title="Sales Quantity",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        font=dict(size=14, color="black"),
        margin=dict(l=30, r=30, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Back button
    if st.button("🔁 Back to Configuration"):
        st.session_state.supply_page = "config"
