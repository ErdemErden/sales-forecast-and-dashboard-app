import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import plotly.express as px
import datetime

# ---------------------- PAGE CONTROL -----------------------
st.set_page_config(page_title="Forecast Tool", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "config"

# ------------------------------ PAGE 1 - CONFIGURATION ------------------------------
if st.session_state.page == "config":
    st.title("📈 Sales Forecast Tool")

    st.markdown("""
    ### 📄 File Format Instructions

    Please upload a **CSV file** in the following format:

    | Product Name | Product ID | Sales Date | Sales Quantity |
    |--------------|------------|------------|----------------|
    | Apple        | 123456     | 06.05.2024 | 1000           |

    - ✅ Accepted file type: `.csv`
    - 📂 Required columns: `Product Name`, `Product ID`, `Sales Date`, `Sales Quantity`
    """)

    uploaded_file = st.file_uploader("📂 Upload your sales CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()

            st.write("#### ✅ Preview of Uploaded Data")
            st.dataframe(df.head())

            with st.expander("📌 Select Columns"):
                product_name_col = st.selectbox("Select the column for 'Product Name':", df.columns)
                product_id_col = st.selectbox("Select the column for 'Product ID':", df.columns)
                date_col = st.selectbox("Select the column for 'Sales Date':", df.columns)
                sales_col = st.selectbox("Select the column for 'Sales Quantity':", df.columns)

            df = df.rename(columns={
                product_name_col: "Product_Name",
                product_id_col: "Product_ID",
                date_col: "Sales_Date",
                sales_col: "Sales_Quantity"
            })

            required_cols = ["Product_Name", "Product_ID", "Sales_Date", "Sales_Quantity"]
            optional_cols = [col for col in df.columns if col not in required_cols]
            st.session_state["optional_columns"] = optional_cols


            # FULL DF (all columns for the graphics)
            df_full = df.copy()
            st.session_state["df_full"] = df_full

            df["Sales_Date"] = pd.to_datetime(df["Sales_Date"], errors="coerce")
            df["Sales_Quantity"] = pd.to_numeric(df["Sales_Quantity"], errors="coerce")
            df = df.dropna(subset=["Sales_Date", "Sales_Quantity"])
            df["Sales_Quantity"] = df["Sales_Quantity"].astype(int)

            df = df.sort_values("Sales_Date")
            df["Week_Num"] = df["Sales_Date"].rank(method="dense").astype(int)

            df = df.groupby(["Week_Num", "Product_ID", "Product_Name"], as_index=False).agg({
                "Sales_Quantity": "sum"
            })

            st.session_state["df"] = df
            st.success("✅ File loaded and prepared successfully! Now configure the forecast.")

        except Exception as e:
            st.error("❌ Could not process the uploaded file. Please check the format.")

    st.markdown("---")
    st.markdown("## 🔮 Forecast Configuration")

    forecast_model = st.selectbox("Select Forecast Model:", [
        "Moving Average",
        "Linear Regression",
        "Exponential Smoothing"
    ], help= "Select the algorithm to be used for forecasting.")

    forecast_duration = st.selectbox("How many months do you want to forecast ahead?",
                                     ["1 month", "3 months", "6 months", "12 months", "18 months"],
                                     help= "This determines how far into the future you want to forecast.")
    forecast_duration = int(forecast_duration.split()[0])

    data_granularity = st.selectbox("What is the granularity of your sales history data?",
                                    ["Daily", "Weekly", "Monthly"],
                                    help= "Indicate whether your uploaded sales data is recorded daily, weekly, or monthly.")

    # Lookback seçeneklerini veri tipine göre değiştir
    if data_granularity == "Daily":
        lookback_options = [7, 14, 21, 30]
    elif data_granularity == "Weekly":
        lookback_options = [4, 8, 12, 16, 20, 24]
    else:
        lookback_options = [1, 3, 6, 9, 12]

    lookback_window = st.selectbox("Select Lookback Period:",
                                   lookback_options, index=1,
                                   help= "How many past data points should be used to calculate each forecast? (e.g., last 14 days or last 6 months)")
    lookback_window = int(lookback_window)


    if st.button("Run Forecast and Analysis"):
        if "df" not in st.session_state:
            st.warning("⚠️ Please upload and process the sales data first.")
            st.stop()

        df = st.session_state["df"]
        if data_granularity == "Weekly":
            forecast_periods = forecast_duration * 4
            date_prefix = "W"
        elif data_granularity == "Monthly":
            forecast_periods = forecast_duration
            date_prefix = "M"
        elif data_granularity == "Daily":
            forecast_periods = forecast_duration * 30
            date_prefix = "D"

        result_df = pd.DataFrame()


        for pid, group in df.groupby("Product_ID"):
            product_name = group["Product_Name"].iloc[0]
            group = group.sort_values("Week_Num")
            history = group["Sales_Quantity"].tolist()

            if len(history) < 3:
                continue

            if forecast_model == "Moving Average":
                extended_history = history.copy()
                forecast = []

                for _ in range(forecast_periods):
                    if len(extended_history) < lookback_window:
                        avg = np.mean(extended_history)
                    else:
                        avg = np.mean(extended_history[-lookback_window:])
                    forecast.append(avg)
                    extended_history.append(avg)

            elif forecast_model == "Linear Regression":
                X = np.arange(len(history)).reshape(-1, 1)
                y = np.array(history)
                model = LinearRegression().fit(X, y)
                future_X = np.arange(len(history), len(history) + forecast_periods).reshape(-1, 1)
                forecast = model.predict(future_X).tolist()

            elif forecast_model == "Exponential Smoothing":
                smoothed_history = pd.Series(history).rolling(window=lookback_window,
                                                              min_periods=1).mean().dropna().values
                alpha = 0.3
                model = SimpleExpSmoothing(smoothed_history)
                fitted_model = model.fit(smoothing_level=alpha, optimized=False)
                forecast = fitted_model.forecast(forecast_periods)


            # Tahmin başlangıcını son satış tarihine göre ayarla
            last_date = pd.to_datetime(st.session_state["df_full"]["Sales_Date"].max()).date() \
                if "df_full" in st.session_state else datetime.date.today()
            forecast_dates = [last_date + datetime.timedelta(weeks=i + 1) for i in range(forecast_periods)]
            forecast_periods_iso = [f"{d.isocalendar()[0]}-W{str(d.isocalendar()[1]).zfill(2)}" for d in forecast_dates]

            for i, f in enumerate(forecast):
                period_label = f"W{i + 1}"
                result_df = pd.concat([result_df, pd.DataFrame({
                    "Product_ID": [pid],
                    "Product_Name": [product_name],
                    "Forecast_Period": [period_label],
                    "Forecast_Quantity": [int(f)],
                    "Method": [forecast_model]
                })])


        st.session_state["forecast_df"] = result_df
        st.session_state.page = "result"




# ------------------------------ PAGE 2 - RESULTS ------------------------------
elif st.session_state.page == "result":
    st.title("📊 Forecast Results")

    df = st.session_state["df"]
    result_df = st.session_state["forecast_df"]
    df_full = st.session_state.get("df_full")

    # Preview table at the top
    st.subheader("🔍 Forecast Preview Table")
    forecast_products = sorted(result_df["Product_Name"].unique().tolist())
    forecast_products.insert(0, "All Products")
    selected_forecast_product = st.selectbox("Filter forecast table by product:", forecast_products)

    if selected_forecast_product == "All Products":
        filtered_forecast = result_df
    else:
        filtered_forecast = result_df[result_df["Product_Name"] == selected_forecast_product]

    st.dataframe(filtered_forecast.head(100))

    # ------------------ FIRST ROW: Sales Trend Left, Monthly Right ------------------
    st.markdown("### 📈 Sales Trends Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📅 Weekly Sales Trend")
        product_list = sorted(df["Product_Name"].unique().tolist())
        selected_products = st.multiselect("Select products to display:", ["All Products"] + product_list, default=["All Products"])
        show_labels = st.checkbox("Show Value Labels", value=False)

        import plotly.graph_objects as go
        fig = go.Figure()

        if "All Products" in selected_products:
            total_df = df.groupby("Week_Num", as_index=False)["Sales_Quantity"].sum()
            total_df["Week_Label"] = total_df["Week_Num"].apply(lambda x: f"W{x}")
            fig.add_trace(go.Scatter(x=total_df["Week_Label"], y=total_df["Sales_Quantity"],
                                     mode="lines+markers+text" if show_labels else "lines+markers",
                                     name="All Products", text=total_df["Sales_Quantity"] if show_labels else None,
                                     textposition="top center", marker=dict(size=8), line=dict(width=3)))
        for product in [p for p in selected_products if p != "All Products"]:
            prod_df = df[df["Product_Name"] == product]
            prod_df["Week_Label"] = prod_df["Week_Num"].apply(lambda x: f"W{x}")
            fig.add_trace(go.Scatter(x=prod_df["Week_Label"], y=prod_df["Sales_Quantity"],
                                     mode="lines+markers+text" if show_labels else "lines+markers",
                                     name=product, text=prod_df["Sales_Quantity"] if show_labels else None,
                                     textposition="top center", marker=dict(size=8)))

        fig.update_layout(title="Weekly Sales Trend", xaxis_title="Week", yaxis_title="Sales Quantity",
                          template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📆 Monthly Sales Trend")
        monthly_sales = df.copy()
        if "Month" not in monthly_sales.columns:
            monthly_sales["Month"] = ((monthly_sales["Week_Num"] - 1) / 4.33 + 1).astype(int)
        monthly_grouped = monthly_sales.groupby("Month")["Sales_Quantity"].sum().reset_index()
        fig = px.bar(monthly_grouped, x="Month", y="Sales_Quantity", title="Monthly Sales Trend",
                     labels={"Sales_Quantity": "Total Sales"})
        fig.update_layout(xaxis=dict(dtick=1), template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ SECOND ROW: Category Pie Chart and Custom Builder ------------------
    st.markdown("### 📊 Additional Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🥧 Product Share in Selected Category")
        possible_names = ["category", "cat", "categorie", "categorié", "kategori"]
        optional_cols = st.session_state.get("optional_columns", [])
        category_col = None
        for col in optional_cols:
            cleaned = col.strip().lower().replace(".", "").replace("é", "e")
            if any(name in cleaned for name in possible_names):
                category_col = col
                break

        if category_col:
            category_list = df_full[category_col].dropna().unique().tolist()
            selected_category = st.selectbox("Select a Category:", category_list)
            filtered_cat = df_full[df_full[category_col] == selected_category]
            product_share = filtered_cat.groupby("Product_Name")["Sales_Quantity"].sum().reset_index()
            fig2 = px.pie(product_share, names="Product_Name", values="Sales_Quantity",
                          title=f"Product Sales Distribution – {selected_category}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ No suitable 'Category' column found in your dataset.")

    with col4:
        st.markdown("#### 🧩 Build Your Own Chart")

        all_columns = df_full.columns.tolist()

        filter_columns = st.multiselect("Select columns to filter (optional):", options=all_columns)
        filtered_df = df_full.copy()
        for col in filter_columns:
            unique_vals = df_full[col].dropna().unique().tolist()
            selected_vals = st.multiselect(f"Filter by {col}:", options=unique_vals, default=unique_vals)
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        x_axis = st.selectbox("Select X-axis:", options=all_columns)
        y_axis = st.selectbox("Select Y-axis:", options=all_columns)
        chart_type = st.selectbox("Select Chart Type:", ["Bar", "Line", "Area", "Pie", "Scatter", "Box"])

        if filtered_df.empty:
            st.warning("⚠️ No data available after filtering.")
        elif not x_axis or not y_axis:
            st.info("ℹ️ Please select both X and Y axes.")
        else:
            st.markdown(f"#### {chart_type} Chart: {y_axis} by {x_axis}")
            if chart_type == "Bar":
                fig = px.bar(filtered_df, x=x_axis, y=y_axis)
            elif chart_type == "Line":
                fig = px.line(filtered_df, x=x_axis, y=y_axis)
            elif chart_type == "Area":
                fig = px.area(filtered_df, x=x_axis, y=y_axis)
            elif chart_type == "Pie":
                fig = px.pie(filtered_df, names=x_axis, values=y_axis)
            elif chart_type == "Scatter":
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
            elif chart_type == "Box":
                fig = px.box(filtered_df, x=x_axis, y=y_axis)

            fig.update_layout(template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

#Run the App - streamlit run 1_Demand_Forecast.py
