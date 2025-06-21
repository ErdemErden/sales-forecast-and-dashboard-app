import streamlit as st

st.set_page_config(page_title="Sales Forecast Tool", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Sales Forecast Tool</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Please choose one of the forecast types below:</h4>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    top1, top2, top3 = st.columns([1, 2, 1])
    with top2:
        st.page_link("pages/1_Demand_Forecast.py", label="➡️ Go to Demand Forecast", use_container_width=True)

    st.markdown("""
    <div style='border:2px solid #00cc66; border-radius:10px; padding:20px; text-align:center; margin-top:10px;'>
        <h3 style='margin-top:10px;'>📈 Demand Forecast</h3>
        <p style='font-size:20px;'><b>Your Criteria</b></p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Historical Sales</li>
            <li>Seasonal Volatility</li>
            <li>Upcoming Promos</li>
        </ul>
        <p style='font-size:18px; text-align:center;'>➕</p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Product Category</li>
            <li>Product Type</li>
            <li>Essentiality</li>
        </ul>
        <p style='font-size:18px; text-align:center;'>➕</p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Forecast Model</li>
            <li>Forecast Period</li>
        </ul>
        <p style='font-size:25px;'>🟦 <b>Result:</b> Historical Sales Analysis & Forecast</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    top4, top5, top6 = st.columns([1, 2, 1])
    with top5:
        st.page_link("pages/2_Supply_Forecast.py", label="➡️ Go to Supply Forecast", use_container_width=True)

    st.markdown("""
    <div style='border:2px solid #ff6600; border-radius:10px; padding:20px; text-align:center; margin-top:10px;'>
        <h3 style='margin-top:10px;'>🚚 Supply Forecast</h3>
        <p style='font-size:20px;'><b>Your Criteria</b></p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Historical Sales</li>
            <li>Seasonal Volatility</li>
            <li>Current Stocks</li>
            <li>Ongoing Orders</li>
            <li>Supplier Delivery Lead Time</li>
            <li>Supplier MOQ</li>
            <li>Upcoming Promos</li>
        </ul>
        <p style='font-size:18px; text-align:center;'>➕</p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Product Category</li>
            <li>Product Type</li>
            <li>Essentiality</li>
        </ul>
        <p style='font-size:18px; text-align:center;'>➕</p>
        <ul style='list-style:none; font-size:20px; padding-left:0;'>
            <li>Forecast Model</li>
            <li>Forecast Period</li>
        </ul>
        <p style='font-size:25px;'>🟦 <b>Result:</b> Historical Sales Analysis & Forecast</p>
    </div>
    """, unsafe_allow_html=True)

#Run the App - streamlit run Home.py