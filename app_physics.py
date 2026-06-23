import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="22 MW CFBC Hybrid Digital Twin",
    layout="wide"
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("22 MW CFBC Power Plant")
st.sidebar.subheader("Hybrid Digital Twin Dashboard")

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)",
    1,
    60,
    5
)

# =====================================================
# LOAD MODELS
# =====================================================

boiler_model = joblib.load(
    "BoilerPrediction_Final_RF_Model.pkl"
)

power_model = joblib.load(
    "PowerPrediction_Final_RF_Model.pkl"
)

health_model = joblib.load(
    "PlantHealthPrediction_Final_XGB_Model.pkl"
)

maintenance_model = joblib.load(
    "MaintenancePrediction_Final_RF_Model.pkl"
)
# =====================================================
# SQLITE DATABASE
# =====================================================

conn = sqlite3.connect(
    "plant_history.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute(
"""
CREATE TABLE IF NOT EXISTS plant_data(

timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

boiler_eff REAL,

tg_generation REAL,

health_index REAL,

maintenance_index REAL

)

"""
)

conn.commit()

# =====================================================
# TITLE
# =====================================================

st.title("22 MW CFBC Power Plant Hybrid Digital Twin")

# =====================================================
# INPUT PARAMETERS
# =====================================================

st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:

    coal_feed = st.number_input(
        "Coal Feed Rate",
        value=11.0
    )

    steam_flow = st.number_input(
        "Steam Flow",
        value=49.0
    )

    total_export = st.number_input(
        "Total Export",
        value=31564.0
    )

    total_import = st.number_input(
        "Total Import",
        value=57721.0
    )

with col2:

    sms = st.number_input(
        "SMS Consumption",
        value=557921.0
    )

    rm = st.number_input(
        "RM Consumption",
        value=87031.0
    )

    cpp_aux = st.number_input(
        "CPP Aux",
        value=51746.0
    )

    dri = st.number_input(
        "DRI",
        value=31613.0
    )

    plf = st.number_input(
        "PLF",
        value=77.0
    )
    # =====================================================
# BOILER INPUTS
# =====================================================

coal_feed = st.number_input(
    "CFBC Coal Consumption",
    value=11.0
)

char_consumption = st.number_input(
    "CFBC Char Consumption",
    value=0.5
)

d_dust = st.number_input(
    "D Dust",
    value=0.3
)

mix_gcv = st.number_input(
    "Mix GCV",
    value=3500.0
)

total_steam_generation = st.number_input(
    "Total Steam Generation",
    value=49.0
)

tg_inlet_steam_flow = st.number_input(
    "TG Inlet Steam Flow",
    value=47.0
)

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict"):

    # Boiler Efficiency
    st.write("Coal =", coal_feed)
    st.write("Char =", char_consumption)
    st.write("D Dust =", d_dust)
    st.write("Mix GCV =", mix_gcv)
    st.write("Steam Generation =", total_steam_generation)
    st.write("TG Inlet Steam Flow =", tg_inlet_steam_flow)
    boiler_input = [[

    coal_feed,
    char_consumption,
    d_dust,
    mix_gcv,
    total_steam_generation,
    tg_inlet_steam_flow

    ]]

    st.write("Boiler Input =", boiler_input)

    boiler_eff = boiler_model.predict(
        boiler_input
    )[0]

    st.write("Boiler Eff =", boiler_eff)

    # TG Generation
    tg_generation = power_model.predict(
        [[
            total_export,
            total_import,
            sms,
            rm,
            cpp_aux,
            dri,
            plf
        ]]
    )[0]

    # Fixed values
    thr = 2150
    shr = 2278
    sfc = 0.85
    power_loss = 1

    # Plant Health
    health_index = health_model.predict(
        [[
            plf,
            thr,
            shr,
            sfc,
            power_loss
        ]]
    )[0]

    # Maintenance Index
    maintenance_index = maintenance_model.predict(
        [[
            plf,
            thr,
            shr,
            sfc,
            power_loss,
            boiler_eff,
            health_index
        ]]
    )[0]
    st.write("Boiler Eff =", boiler_eff)
    st.write("TG Generation =", tg_generation)
    st.write("Health Index =", health_index)
    st.write("Maintenance Index =", maintenance_index)
    
    # =====================================================
    # SAVE TO DATABASE
    # =====================================================

    cursor.execute(
        """
        INSERT INTO plant_data(
            boiler_eff,
            tg_generation,
            health_index,
            maintenance_index
        )
        VALUES (?,?,?,?)
        """,
        (
            float(boiler_eff),
            float(tg_generation),
            float(health_index),
            float(maintenance_index),
        ),
    )

    conn.commit()

    # =====================================================
    # PREDICTIONS
    # =====================================================

    st.header("Predictions")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Boiler Efficiency (%)",
            round(boiler_eff,2)
        )

    with c2:
        st.metric(
            "TG Generation",
            round(tg_generation,2)
        )

    with c3:
        st.metric(
            "Plant Health Index",
            round(health_index,2)
        )

    with c4:
        st.metric(
            "Maintenance Index",
            round(maintenance_index,2)
        )
        # =====================================================
    # EMS PANEL
    # =====================================================

    st.header("Energy Management System (EMS)")

    # Critical Alarm
    critical_alarm = (
        health_index < 50
    )

    # Preventive Maintenance
    preventive_maintenance = (
        maintenance_index < 70
    ) and not critical_alarm

    # Performance Warning
    performance_warning = (
        boiler_eff < 80
        or tg_generation < 350000
    ) and not critical_alarm and not preventive_maintenance

    # Optimization Required
    optimization_required = (
        70 < boiler_eff < 85
    ) and not critical_alarm and not preventive_maintenance and not performance_warning

    # Normal Operation
    normal_operation = (
        not critical_alarm
        and not preventive_maintenance
        and not performance_warning
        and not optimization_required
    )

    # =====================================================
    # DISPLAY EMS STATUS
    # =====================================================

    if critical_alarm:

        st.error(
            "🔴 CRITICAL ALARM\n\n"
            "Immediate inspection required.\n"
            "Plant health has fallen below acceptable limits."
        )

    elif preventive_maintenance:

        st.warning(
            "🔵 PREVENTIVE MAINTENANCE REQUIRED\n\n"
            "Maintenance index is low.\n"
            "Schedule maintenance activities."
        )

    elif performance_warning:

        st.warning(
            "🟡 PERFORMANCE WARNING\n\n"
            "Efficiency or generation level is below desired values."
        )

    elif optimization_required:

        st.info(
            "🟠 OPTIMIZATION REQUIRED\n\n"
            "Plant is operating normally, but efficiency can be improved."
        )

    else:

        st.success(
            "🟢 NORMAL OPERATION\n\n"
            "Plant is operating under healthy conditions."
        )

    # =====================================================
    # EMS FLAGS
    # =====================================================

    st.subheader("EMS Flags")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Normal",
            int(normal_operation)
        )

    with col2:

        st.metric(
            "Performance",
            int(performance_warning)
        )

    with col3:

        st.metric(
            "Optimization",
            int(optimization_required)
        )

    with col4:

        st.metric(
            "Maintenance",
            int(preventive_maintenance)
        )

    with col5:

        st.metric(
            "Critical",
            int(critical_alarm)
        )
        # =====================================================
    # DIGITAL TWIN PERFORMANCE DASHBOARD
    # =====================================================

    st.header("Digital Twin Performance Dashboard")

    row1_col1, row1_col2 = st.columns(2)

    # Boiler Efficiency Gauge
    with row1_col1:

        fig1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=boiler_eff,
                title={'text': "Boiler Efficiency (%)"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0,60], 'color': "red"},
                        {'range': [60,80], 'color': "yellow"},
                        {'range': [80,100], 'color': "lightgreen"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )


    # Plant Health Gauge
    with row1_col2:

        fig2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health_index,
                title={'text': "Plant Health Index"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0,50], 'color': "red"},
                        {'range': [50,80], 'color': "yellow"},
                        {'range': [80,100], 'color': "lightgreen"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    row2_col1, row2_col2 = st.columns(2)

    # Maintenance Gauge
    with row2_col1:

        fig3 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=maintenance_index,
                title={'text': "Maintenance Index"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0,50], 'color': "red"},
                        {'range': [50,80], 'color': "yellow"},
                        {'range': [80,100], 'color': "lightgreen"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )


    # TG Generation Gauge
    with row2_col2:

        fig4 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=tg_generation,
                title={'text': "TG Generation"},
                gauge={
                    'axis': {'range': [0,550000]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0,250000], 'color': "red"},
                        {'range': [250000,400000], 'color': "yellow"},
                        {'range': [400000,550000], 'color': "lightgreen"}
                    ]
                }
            )
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )


    # =====================================================
    # HISTORICAL TREND GRAPHS
    # =====================================================

    # =====================================================
    # REAL HISTORICAL DATA
    # =====================================================

    history = pd.read_sql(

     """

    SELECT *

    FROM plant_data

    ORDER BY timestamp ASC

    """,

    conn

 )


    fig_be = px.line(

    history,

    x="timestamp",

    y="boiler_eff",

    title="Boiler Efficiency Trend"

)

    st.plotly_chart(
        fig_be,
        use_container_width=True
    )


    fig_health = px.line(

    history,

    x="timestamp",

    y="health_index",

    title="Plant Health Trend"

)

    st.plotly_chart(
        fig_health,
        use_container_width=True
    )


    fig_maint = px.line(
    history,
    x="timestamp",
    y="maintenance_index",
    title="Maintenance Index Trend",
    markers=True
    )

    fig_maint.update_yaxes(
    range=[
        history["maintenance_index"].min()-1,
        history["maintenance_index"].max()+1
    ]
    )

    st.plotly_chart(
        fig_maint,
        use_container_width=True
    )


    fig_power = px.line(
    history,
    x="timestamp",
    y="tg_generation",
    title="TG Generation Trend",
    markers=True
    )

    fig_power.update_yaxes(
    range=[
        history["tg_generation"].min()-10000,
        history["tg_generation"].max()+10000
    ]
   )

    st.plotly_chart(
        fig_power,
        use_container_width=True
    )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = pd.DataFrame({

        "Parameter":[
            "Boiler Efficiency",
            "TG Generation",
            "Plant Health Index",
            "Maintenance Index"
        ],

        "Value":[
            round(boiler_eff,2),
            round(tg_generation,2),
            round(health_index,2),
            round(maintenance_index,2)
        ]

    })


    csv = report.to_csv(
        index=False
    )


    st.download_button(

        "Download Report",

        csv,

        file_name="DigitalTwin_Report.csv",

        mime="text/csv"

    )