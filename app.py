import streamlit as st
import joblib
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import time

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

st.title(
    "22 MW CFBC Power Plant Hybrid Digital Twin"
)

# =====================================================
# TWIN MODE
# =====================================================

mode = st.sidebar.radio(

    "Twin Mode",

    [

        "ML Twin",

        "Physics Twin"

    ]

)

# =====================================================
# DEFAULT VALUES
# =====================================================

boiler_eff = 0
tg_generation = 0
health_index = 0
maintenance_index = 0

critical_alarm = False
preventive_maintenance = False
performance_warning = False
optimization_required = False
normal_operation = False

heat_rate = 0
shr = 0
sfc = 0

plant_alarm = 0
recommendation = 0
alarm_state = 0
maintenance_required = False

# =====================================================
# MODE DISPLAY
# =====================================================

if mode == "ML Twin":

    st.success(
        "🟢 ML Twin Mode"
    )

else:

    st.info(
        "🔵 Physics Twin Mode"
    )
# =====================================================
# ML TWIN
# =====================================================

if mode == "ML Twin":

    st.header("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:

        total_export = st.number_input(
            "Total Export",
            value=31564.0
        )

        total_import = st.number_input(
            "Total Import",
            value=57721.0
        )

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

    with col2:

        dri = st.number_input(
            "DRI",
            value=31613.0
        )

        plf = st.number_input(
            "PLF",
            value=77.0
        )

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

    if st.button("Predict"):

        # Boiler Model
        boiler_input = [[

            coal_feed,
            char_consumption,
            d_dust,
            mix_gcv,
            total_steam_generation,
            tg_inlet_steam_flow

        ]]

        boiler_eff = boiler_model.predict(
            boiler_input
        )[0]

        # Power Model

        power_input = [[

            total_export,
            total_import,
            sms,
            rm,
            cpp_aux,
            dri,
            plf

        ]]

        tg_generation = power_model.predict(
            power_input
        )[0]

        # Fixed Parameters

        thr = 2150
        shr = 2278
        sfc = 0.85
        power_loss = 1

        # Health Model

        health_input = [[

            plf,
            thr,
            shr,
            sfc,
            power_loss

        ]]

        health_index = health_model.predict(
            health_input
        )[0]

        # Maintenance Model

        maintenance_input = [[

            plf,
            thr,
            shr,
            sfc,
            power_loss,
            boiler_eff,
            health_index

        ]]

        maintenance_index = maintenance_model.predict(
            maintenance_input
        )[0]
# =====================================================
# PHYSICS TWIN
# =====================================================

if mode == "Physics Twin":

    sim_data = pd.read_csv(
        "simulink_outputs.csv"
    )

    boiler_eff = sim_data[
        "PredictedBoilerEfficiency"
    ][0]

    tg_generation = sim_data[
        "TGGeneration"
    ][0]

    health_index = sim_data[
        "HealthIndex"
    ][0]

    maintenance_index = sim_data[
        "MaintenanceIndex"
    ][0]

    heat_rate = sim_data[
        "HeatRate"
    ][0]

    shr = sim_data[
        "SHR"
    ][0]

    sfc = sim_data[
        "SFC"
    ][0]

    plant_alarm = sim_data[
        "PlantAlarm"
    ][0]

    recommendation = sim_data[
        "Recommendation"
    ][0]

    alarm_state = sim_data[
        "AlarmState"
    ][0]

    performance_warning = sim_data[
        "PerformanceWarning"
    ][0]

    maintenance_required = sim_data[
        "MaintenanceRequired"
    ][0]

    critical_alarm = sim_data[
        "CriticalAlarm"
    ][0]

    st.header(
        "Physics Twin Information"
    )

    st.write(
        "Heat Rate :",
        round(heat_rate,2)
    )

    st.write(
        "SHR :",
        round(shr,2)
    )

    st.write(
        "SFC :",
        round(sfc,2)
    )

    st.write(
        "Plant Alarm :",
        plant_alarm
    )

    st.write(
        "Recommendation :",
        recommendation
    )
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
        float(maintenance_index)
    )
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

if mode == "ML Twin":

    critical_alarm = (
        health_index < 50
    )

    preventive_maintenance = (
        maintenance_index < 70
    ) and not critical_alarm

    performance_warning = (
        boiler_eff < 80
        or tg_generation < 350000
    ) and not critical_alarm and not preventive_maintenance

    optimization_required = (
        70 < boiler_eff < 85
    ) and not critical_alarm and not preventive_maintenance and not performance_warning

    normal_operation = (
        not critical_alarm
        and not preventive_maintenance
        and not performance_warning
        and not optimization_required
    )

else:

    preventive_maintenance = bool(
        maintenance_required
    )

    optimization_required = False

    normal_operation = (
        not critical_alarm
        and not performance_warning
        and not preventive_maintenance
    )
if critical_alarm:

    st.error(
        "🔴 CRITICAL ALARM"
    )

elif preventive_maintenance:

    st.warning(
        "🔵 PREVENTIVE MAINTENANCE REQUIRED"
    )

elif performance_warning:

    st.warning(
        "🟡 PERFORMANCE WARNING"
    )

elif optimization_required:

    st.info(
        "🟠 OPTIMIZATION REQUIRED"
    )

else:

    st.success(
        "🟢 NORMAL OPERATION"
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

st.header("Historical Trend Analysis")

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

st.header("Download Digital Twin Report")

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
if mode == "Physics Twin":

    time.sleep(
        refresh_rate
    )

    st.rerun()
st.markdown("---")

st.markdown(

"""
Developed By BIBEK SUNDARAY 

**22 MW CFBC Power Plant Hybrid Dynamic Digital Twin**

Features:

- Physics Twin (Simulink)
- AI Twin (Random Forest + XGBoost)
- EMS Layer
- Historical Trends
- SQLite Historian
- Download Reports
- Streamlit Dashboard

"""
)
history = pd.read_sql(
"""
SELECT *
FROM plant_data
""",
conn
)

history.to_csv(
"powerbi_data.csv",
index=False
)
 