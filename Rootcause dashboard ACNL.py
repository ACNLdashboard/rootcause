# Importeer alle benodigde bibliotheken
import streamlit as st # Voor het bouwen van de webapplicatie (dashboard)
import pandas as pd # Voor data manipulatie en analyse (dataframes)
import requests # Voor het maken van HTTP API calls (naar FlightAware)
import plotly.graph_objects as go # Voor het maken van interactieve grafieken
from datetime import datetime, timedelta # Voor het werken met datums en tijden
from sklearn.linear_model import LinearRegression # Machine learning model voor trendvoorspelling
import numpy as np # Voor numerieke berekeningen
import re # Voor reguliere expressies (tekstpatronen zoeken)
import os # Voor interactie met het besturingssysteem
import io # Voor het inlezen van bestanden als strings (voor de CSV upload)
from fpdf import FPDF # Voor het genereren van PDF rapportages

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Flight Operations Performance Insights", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 10px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .analysis-card {
        background-color: #ffffff; padding: 25px; border-radius: 12px;
        border-left: 5px solid #0369a1; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        height: 100%;
    }
    .ml-card {
        background-color: #f8fafc; padding: 25px; border-radius: 12px;
        border: 1px solid #e2e8f0; border-left: 5px solid #8b5cf6;
        height: 100%;
    }
    .pattern-item { margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #f1f5f9; font-size: 0.95em; }
    .stPlotlyChart { margin-top: 5px !important; margin-bottom: 20px !important; }
    .stDataFrame { margin-top: 15px !important; margin-bottom: 50px !important; }
    </style>
    """, unsafe_allow_html=True)

if 'run_main' not in st.session_state: st.session_state['run_main'] = False

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def create_pdf_report(df_res, flight_num, date_str, buffer_info="", intelligence_info="", current_season="", seasonal_log_df=None):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    gen_now = datetime.now().strftime("%d-%m-%Y %H:%M")
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 5, f"Report Generated on: {gen_now}", ln=True, align='R')
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Flight Operations Intelligence Report: {flight_num}", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {date_str} | Season: {current_season}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "1. Flight Rotation Analysis", ln=True)
    pdf.set_font("Arial", 'B', 7)
    cols = ["Flight", "Origin", "Dest", "STD", "ATD", "STA", "ATA", "Dep Dly", "Arr Dly", "Block", "Turn"]
    cw = [25, 30, 30, 22, 22, 22, 22, 20, 20, 25, 25]
    for i, col in enumerate(cols):
        pdf.cell(cw[i], 10, col, border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 7)
    for _, row in df_res.iterrows():
        pdf.cell(cw[0], 8, str(row['Flight']), border=1)
        pdf.cell(cw[1], 8, str(row['Origin']), border=1)
        pdf.cell(cw[2], 8, str(row['Dest']), border=1)
        pdf.cell(cw[3], 8, str(row['Slot Out (STD)']), border=1)
        pdf.cell(cw[4], 8, str(row['Actual Out (ATD)']), border=1)
        pdf.cell(cw[5], 8, str(row['Slot In (STA)']), border=1)
        pdf.cell(cw[6], 8, str(row['Actual In (ATA)']), border=1)
        pdf.cell(cw[7], 8, str(row['Dep Delay']), border=1)
        pdf.cell(cw[8], 8, str(row['Arr Delay']), border=1)
        pdf.cell(cw[9], 8, str(row['Block (S/A)']), border=1)
        pdf.cell(cw[10], 8, str(row['Turn (S/A)']), border=1)
        pdf.ln()
    if buffer_info:
        pdf.ln(5); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 10, "2. Operational Buffer Analysis", ln=True)
        pdf.set_font("Arial", '', 9); pdf.multi_cell(0, 8, buffer_info)
    if intelligence_info:
        pdf.ln(5); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 10, "3. Strategic Trend Analysis & Prediction", ln=True)
        pdf.set_font("Arial", '', 9); pdf.multi_cell(0, 8, intelligence_info)
    if seasonal_log_df is not None and not seasonal_log_df.empty:
        pdf.add_page(); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 10, "4. Full Seasonal Rotation Log", ln=True)
        pdf.set_font("Arial", 'B', 8); s_cols = ["Date", "Reg", "Max Turn Loss", "Max Flight Loss"]; s_cw = [40, 40, 90, 90]
        for i, col in enumerate(s_cols): pdf.cell(s_cw[i], 10, col, border=1, align='C')
        pdf.ln(); pdf.set_font("Arial", '', 8)
        for _, s_row in seasonal_log_df.iterrows():
            pdf.cell(s_cw[0], 8, str(s_row['Date']), border=1); pdf.cell(s_cw[1], 8, str(s_row['Reg']), border=1)
            pdf.cell(s_cw[2], 8, str(s_row['Max Turn Loss']), border=1); pdf.cell(s_cw[3], 8, str(s_row['Max Flight Loss']), border=1); pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

def get_iata_season_bounds(dt):
    year = dt.year
    last_sunday_march = max(week for week in [datetime(year, 3, d) for d in range(25, 32)] if week.weekday() == 6).date()
    last_sunday_october = max(week for week in [datetime(year, 10, d) for d in range(25, 32)] if week.weekday() == 6).date()
    if dt < last_sunday_march:
        prev_year = year - 1
        s_start = max(week for week in [datetime(prev_year, 10, d) for d in range(25, 32)] if week.weekday() == 6).date()
        return s_start, last_sunday_march - timedelta(days=1), f"W{str(prev_year)[2:]}", f"W{str(year)[2:]}"
    elif dt < last_sunday_october:
        return last_sunday_march, last_sunday_october - timedelta(days=1), f"S{str(year)[2:]}", f"S{str(year+1)[2:]}"
    else:
        next_year_march = max(week for week in [datetime(year+1, 3, d) for d in range(25, 32)] if week.weekday() == 6).date()
        return last_sunday_october, next_year_march - timedelta(days=1), f"W{str(year)[2:]}", f"W{str(year+1)[2:]}"

def parse_acnl_date(date_str):
    if pd.isna(date_str): return None
    try:
        clean_str = str(date_str).strip()
        for fmt in ("%d%b%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try: return datetime.strptime(clean_str, fmt).date()
            except: continue
        return pd.to_datetime(clean_str).date()
    except: return None

def format_time(dt_obj):
    return dt_obj.strftime("%H:%M") if pd.notnull(dt_obj) else "-"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_unified_data(api_key, ident, start_dt, end_dt):
    target_ident = str(ident).replace(" ", "").strip().upper()
    headers = {'x-apikey': api_key}
    start_ts = datetime.combine(start_dt, datetime.min.time()).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_ts = datetime.combine(end_dt, datetime.max.time()).strftime('%Y-%m-%dT%H:%M:%SZ')
    params = {'start': start_ts, 'end': end_ts, 'max_pages': 1}
    for endpoint in ["flights", "history/flights"]:
        try:
            url = f"https://aeroapi.flightaware.com/aeroapi/{endpoint}/{target_ident}"
            res = requests.get(url, headers=headers, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json().get('flights', [])
                if data: return data
        except: continue
    return []

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================

with st.sidebar:
    st.header("Settings")
    fa_api_key = st.text_input("FlightAware API Key", type="password")
    st.subheader("Data Management")
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

col_title, col_logo = st.columns([8, 1])
with col_title:
    st.title("Flight Operations Performance Insights")
with col_logo:
    try:
        st.image("logo-2-removebg-preview.png", use_container_width=True)
    except:
        pass

if uploaded_file:
    try:
        raw_bytes = uploaded_file.getvalue()
        content = raw_bytes.decode('utf-8', errors='ignore')
        sep = ';' if content.count(';') > content.count(',') else ','
        df = pd.read_csv(io.StringIO(content), sep=sep)
        df.columns = df.columns.str.strip()
        if not df.empty:
            df = df[~df.iloc[:, 0].astype(str).str.contains("Coord/Act Date|source:", case=False, na=False)]
        reg_cols = [c for c in df.columns if "Registration" in c or "ACReg" in c]
        if reg_cols: 
            df['Registration_Final'] = df[reg_cols].bfill(axis=1).iloc[:, 0]
        else: 
            st.error(f"Kolom 'Registration' niet gevonden.")
            st.stop()
        df_clean = df.dropna(subset=['Date', 'Registration_Final', 'Operated Flight#']).copy()
        if not df_clean.empty:
            df_clean['Label'] = df_clean['Operated Flight#'].astype(str) + " | " + df_clean['Date'].astype(str)
            selected_option = st.selectbox("Search flight rotation:", df_clean['Label'].unique())
            row_sel = df_clean[df_clean['Label'] == selected_option].iloc[0]
            search_date = parse_acnl_date(row_sel['Date'])
            registration = str(row_sel['Registration_Final']).strip()
            flight_number_clean = str(row_sel['Operated Flight#']).replace(" ", "").upper()
            flight_digits_only = "".join(filter(str.isdigit, flight_number_clean))
            flight_display_name = str(row_sel['Operated Flight#'])
            s_start, s_end, current_season_code, next_season_code = get_iata_season_bounds(search_date)

            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state['run_main'] = True

            if st.session_state.get('run_main') and fa_api_key:
                reg_flights = fetch_unified_data(fa_api_key, registration, search_date, search_date)
                target_flight_data = fetch_unified_data(fa_api_key, flight_number_clean, search_date, search_date)
                combined_flights = reg_flights + target_flight_data
                unique_flights = []
                seen_keys = set()
                for f in combined_flights:
                    if f and f.get('scheduled_out'):
                        key = f"{f.get('ident')}_{f.get('scheduled_out')}"
                        if key not in seen_keys:
                            unique_flights.append(f)
                            seen_keys.add(key)
                if unique_flights:
                    rows, prev_sta, prev_ata, prev_arr_delay = [], None, None, 0
                    sorted_flights = sorted(unique_flights, key=lambda x: x.get('scheduled_out'))
                    for f in sorted_flights:
                        std = pd.to_datetime(f.get('scheduled_out'))
                        if std.date() != search_date: continue
                        sta, atd, ata = pd.to_datetime(f.get('scheduled_in')), pd.to_datetime(f.get('actual_out')), pd.to_datetime(f.get('actual_in'))
                        atd_val = atd if atd else (pd.to_datetime(f.get('actual_off')) - timedelta(minutes=10) if f.get('actual_off') else None)
                        ata_val = ata if ata else (pd.to_datetime(f.get('actual_on')) + timedelta(minutes=5) if f.get('actual_on') else None)
                        dep_delay, arr_delay = int((atd_val - std).total_seconds() / 60) if atd_val and std else 0, int((ata_val - sta).total_seconds() / 60) if ata_val and sta else 0
                        turn_s, turn_a = int((std - prev_sta).total_seconds() / 60) if prev_sta else None, int((atd_val - prev_ata).total_seconds() / 60) if prev_ata and atd_val else None
                        sched_block, act_block = int((sta - std).total_seconds() / 60), int((ata_val - atd_val).total_seconds() / 60) if ata_val and atd_val else 0
                        orig_iata = f.get('origin', {}).get('code_iata', '-')
                        orig_icao = f.get('origin', {}).get('code_icao', '-')
                        dest_iata = f.get('destination', {}).get('code_iata', '-')
                        dest_icao = f.get('destination', {}).get('code_icao', '-')
                        rows.append({
                            'Flight': f.get('ident'), 'Origin': f"{orig_iata} ({orig_icao})" if orig_iata != '-' else orig_icao, 'Dest': f"{dest_iata} ({dest_icao})" if dest_iata != '-' else dest_icao,
                            'Slot Out (STD)': format_time(std), 'Actual Out (ATD)': format_time(atd_val), 'Dep Delay': f"{dep_delay}m", 
                            'Actual In (ATA)': format_time(ata_val), 'Slot In (STA)': format_time(sta), 'Arr Delay': f"{arr_delay}m", 
                            'Block (S/A)': f"{sched_block}/{act_block}", 'Turn (S/A)': f"{turn_s if turn_s else '-'}/{turn_a if turn_a else '-'}",
                            'p_STD': std, 'p_STA': sta, 'p_ATD': atd_val, 'p_ATA': ata_val, 'Node': f.get('ident'),
                            'Inbound_Arr_Delay': prev_arr_delay, 'Sched_Turn': turn_s, 'Act_Turn': turn_a,
                            'Sched_Block': sched_block, 'Act_Block': act_block, 'Dep_Delay_Num': dep_delay, 'Arr_Delay_Num': arr_delay
                        })
                        prev_sta, prev_ata, prev_arr_delay = sta, ata_val, arr_delay
                    df_res = pd.DataFrame(rows)
                    weekday_num = search_date.isoweekday()
                    c_t1, c_t2, c_t3, c_t4, c_t5 = st.columns(5)
                    c_t1.metric("Selected Flight", flight_display_name)
                    c_t2.metric("Aircraft", registration); c_t3.metric("Season", current_season_code); c_t4.metric("Date", search_date.strftime("%d-%m-%Y")); c_t5.metric("Week Number", f"W{search_date.isocalendar()[1]} | D{weekday_num}")

                    st.subheader("Flight Rotation Analysis")
                    with st.expander("ⓘ Info - Column definitions & Calculations"):
                        st.write("Table definitions...")
                    st.dataframe(df_res[['Flight', 'Origin', 'Dest', 'Slot Out (STD)', 'Actual Out (ATD)', 'Dep Delay', 'Actual In (ATA)', 'Slot In (STA)', 'Arr Delay', 'Block (S/A)', 'Turn (S/A)']], use_container_width=True)

                    st.subheader("Inbound vs. Outbound Delay Analysis")
                    fig_erosion = go.Figure()
                    fig_erosion.add_trace(go.Scatter(x=df_res['Node'], y=df_res['Inbound_Arr_Delay'], mode='lines+markers', name='Inbound Delay', line=dict(color='#94a3b8', dash='dash')))
                    fig_erosion.add_trace(go.Scatter(x=df_res['Node'], y=df_res['Dep_Delay_Num'], mode='lines+markers+text', name='Outbound Delay', line=dict(color='#ef4444', width=3), text=df_res['Dep_Delay_Num'], textposition="top center"))
                    st.plotly_chart(fig_erosion.update_layout(height=350), use_container_width=True)

                    # ==========================================
                    # UPDATED GANTT VIEW WITH GAP TRIANGLE
                    # ==========================================
                    st.subheader("Actual vs. Scheduled Timeline")
                    with st.expander("ⓘ Info"): 
                        st.write("Gantt view. Grey = Plan, Colored = Actual. AMS margin: 45m (Black/Red), Others: 15m (Red).")
                        st.markdown("⚠️ **Warning Triangle:** Visible between scheduled legs when a planning gap (Scheduled Turnaround) of **40 minutes or more** is detected.")
                    
                    fig_g = go.Figure()
                    for i, r in df_res.iterrows():
                        # Plot Scheduled Line
                        fig_g.add_trace(go.Scatter(x=[r['p_STD'], r['p_STA']], y=[i + 0.15, i + 0.15], mode='lines', line=dict(color='#CBD5E1', width=8), showlegend=False))
                        
                        # Plot Warning Triangle for gaps >= 40 min
                        if i > 0:
                            prev_r = df_res.iloc[i-1]
                            # Check gap between previous STA and current STD
                            gap_minutes = (r['p_STD'] - prev_r['p_STA']).total_seconds() / 60
                            if gap_minutes >= 40:
                                # Calculate midpoint for placement
                                midpoint = prev_r['p_STA'] + (r['p_STD'] - prev_r['p_STA']) / 2
                                fig_g.add_trace(go.Scatter(
                                    x=[midpoint], y=[i - 0.5], 
                                    mode='markers+text',
                                    marker=dict(symbol='triangle-up', size=15, color='#f59e0b'),
                                    text=["⚠️"], textposition="top center",
                                    name="Planning Gap Warning",
                                    showlegend=False
                                ))

                        # Plot Actual Line
                        dest_check = str(r['Dest']).upper()
                        delay = r['Arr_Delay_Num']
                        if any(x in dest_check for x in ["AMS", "EHAM"]): color = '#EF4444' if delay > 30 else ('#000000' if 10 <= delay <= 30 else '#22C55E')
                        else: color = '#EF4444' if delay > 15 else '#22C55E'
                        fig_g.add_trace(go.Scatter(x=[r['p_ATD'], r['p_ATA']], y=[i - 0.15, i - 0.15], mode='lines+markers', line=dict(color=color, width=8), marker=dict(symbol='triangle-right', size=10), showlegend=False))
                    
                    fig_g.update_layout(height=450, yaxis=dict(tickvals=list(range(len(df_res))), ticktext=df_res['Flight'].tolist()))
                    st.plotly_chart(fig_g, use_container_width=True)

                    st.divider(); c1, c2 = st.columns(2)
                    match_indices = [i for i, f_id in enumerate(df_res['Flight']) if flight_digits_only in str(f_id)]
                    df_log_subset = df_res.loc[:match_indices[0]].copy() if match_indices else df_res.copy()

                    with c1:
                        st.subheader("Planned vs. Actual Turnaround Performance")
                        turn_labels = [f"{i+1}. {r['Origin']}" for i, r in df_log_subset.iterrows() if r['Sched_Turn'] is not None]
                        sched_turns = [r['Sched_Turn'] for _, r in df_log_subset.iterrows() if r['Sched_Turn'] is not None]
                        act_turns = [r['Act_Turn'] for _, r in df_log_subset.iterrows() if r['Sched_Turn'] is not None]
                        st.plotly_chart(go.Figure([go.Bar(x=turn_labels, y=sched_turns, name='Plan', marker_color='lightgrey'), go.Bar(x=turn_labels, y=act_turns, name='Actual', marker_color='#3498db')]).update_layout(barmode='group', height=350), use_container_width=True)
                    with c2:
                        st.subheader("Planned vs. Actual Block-Time performance")
                        st.plotly_chart(go.Figure([go.Bar(x=df_log_subset['Flight'], y=df_log_subset['Sched_Block'], name='Plan', marker_color='lightgrey'), go.Bar(x=df_log_subset['Flight'], y=df_log_subset['Act_Block'], name='Actual', marker_color='#2ecc71')]).update_layout(barmode='group', height=350), use_container_width=True)

                    st.divider(); st.subheader(f"Total Turnaround Buffer (until {flight_display_name} Departure)")
                    buffer_summary = ""
                    if match_indices:
                        min_turn_day = df_res['Sched_Turn'].dropna().min() if not df_res['Sched_Turn'].dropna().empty else 0
                        valid_turns = df_log_subset['Sched_Turn'].dropna()
                        if not valid_turns.empty:
                            total_p, total_min = valid_turns.sum(), len(valid_turns) * min_turn_day
                            buffer = total_p - total_min
                            bm1, bm2, bm3 = st.columns(3)
                            bm1.metric("Min. Required (Sum)", f"{int(total_min)}m"); bm2.metric("Planned Total (Sum)", f"{int(total_p)}m"); bm3.metric("Built-in Buffer", f"{int(buffer)}m", delta=f"{int(buffer)}m")
                            buffer_summary = f"Total Operational Buffer: {int(buffer)}m."

                    target_airport = "UNK"
                    for col in row_sel.index:
                        if any(x in col for x in ["Airport", "Station", "Luchthaven"]):
                            target_airport = str(row_sel[col]).strip(); break
                    flight_type = "MOV"
                    for col in row_sel.index:
                        if any(x in col for x in ["ArrDep", "Type", "A/D"]):
                            val = str(row_sel[col]).upper()
                            if "A" in val: flight_type = "ARR"
                            elif "D" in val: flight_type = "DEP"; break

                    base_filename = f"{search_date.strftime('%Y%m%d')} {current_season_code} different time {target_airport} {flight_number_clean} {flight_type} D{weekday_num}"
                    pdf_basic = create_pdf_report(df_res, flight_display_name, search_date.strftime("%d-%m-%Y"), buffer_summary, "", current_season_code)
                    st.download_button("Download Report as PDF", pdf_basic, f"{base_filename}.pdf", "application/pdf", use_container_width=True)

                    st.divider(); st.subheader(f"{current_season_code} Strategic Trend Analysis")
                    if st.button(f"🚀 Analyze structural {current_season_code} patterns", use_container_width=True):
                        with st.spinner("Analyzing seasonal patterns..."):
                            history_rows, all_leg_data, ml_X, ml_y, sample_count = [], [], [], [], 0
                            current_dt, end_limit = s_start, min(s_end, datetime.now().date())
                            targets = ["EIN", "EHEH", "AMS", "EHAM", "RTM", "EHRD"]
                            api_ident = flight_number_clean
                            while current_dt <= end_limit:
                                if current_dt.weekday() == search_date.weekday():
                                    day_f = fetch_unified_data(fa_api_key, api_ident, current_dt, current_dt)
                                    valid_reg = None
                                    if day_f:
                                        for f in day_f:
                                            dest = f.get('destination');
                                            if dest and (dest.get('code_iata') in targets or dest.get('code_icao') in targets): 
                                                valid_reg = f.get('registration'); break
                                    if valid_reg:
                                        rotation_raw = fetch_unified_data(fa_api_key, valid_reg, current_dt, current_dt)
                                        if rotation_raw:
                                            sample_count += 1
                                            day_T, day_F, d_rows, prev_s, prev_a = 0, 0, [], None, None
                                            hist_sorted = sorted([f for f in rotation_raw if f and f.get('scheduled_out')], key=lambda x: x.get('scheduled_out'))
                                            for h in hist_sorted:
                                                h_std, h_sta = pd.to_datetime(h.get('scheduled_out')), pd.to_datetime(h.get('scheduled_in'))
                                                h_atd, h_ata = pd.to_datetime(h.get('actual_out')), pd.to_datetime(h.get('actual_in'))
                                                if h_std.date() != current_dt: continue
                                                t_s, t_a = int((h_std-prev_s).total_seconds()/60) if prev_s else 0, int((h_atd-prev_a).total_seconds()/60) if prev_a and h_atd else 0
                                                t_loss = max(0, (t_a - t_s)) if prev_s and prev_a else 0
                                                f_loss = max(0, (int((h_ata-h_atd).total_seconds()/60) - int((h_sta-h_std).total_seconds()/60))) if h_sta and h_ata and h_atd else 0
                                                day_T += t_loss; day_F += f_loss
                                                all_leg_data.append({'Origin': h.get('origin', {}).get('code_iata'), 'Dest': h.get('destination', {}).get('code_iata'), 'T_Var': t_loss, 'F_Var': f_loss, 'Ident': h.get('ident')})
                                                d_rows.append({'T_Var': t_loss, 'F_Var': f_loss, 'Origin': h.get('origin', {}).get('code_iata'), 'Ident': h.get('ident')})
                                                if flight_digits_only in str(h.get('ident')):
                                                    if h_sta and h_ata: ml_X.append([day_T, day_F]); ml_y.append(int((h_ata - h_sta).total_seconds()/60))
                                                    break
                                                prev_s, prev_a = h_sta, h_ata
                                            if d_rows:
                                                df_h = pd.DataFrame(d_rows)
                                                max_t_idx, max_f_idx = df_h.T_Var.idxmax(), df_h.F_Var.idxmax()
                                                history_rows.append({
                                                    'Date': current_dt.strftime('%d-%b'), 
                                                    'Reg': valid_reg, 
                                                    'Max Turn Loss': f"{df_h.loc[max_t_idx]['Origin']} (+{int(df_h.T_Var.max())}m)", 
                                                    'Max Flight Loss': f"{df_h.loc[max_f_idx]['Ident']} (+{int(df_h.F_Var.max())}m)"
                                                })
                                current_dt += timedelta(days=1)
                        if history_rows:
                            st.subheader("Full Seasonal Rotation Log")
                            hist_df = pd.DataFrame(history_rows); st.table(hist_df)
                            intelligence_text = ""
                            if len(ml_y) > 2:
                                pat_df = pd.DataFrame(all_leg_data); ml_X_arr, ml_y_arr = np.array(ml_X), np.array(ml_y)
                                model = LinearRegression().fit(ml_X_arr, ml_y_arr); r2 = model.score(ml_X_arr, ml_y_arr)
                                s_pred = model.predict([[pat_df['T_Var'].mean(), pat_df['F_Var'].mean()]])[0] if r2 > 0.15 else np.mean(ml_y)
                                g_i, f_i = (model.coef_[0], model.coef_[1]) if r2 > 0.15 else (1.0, 1.0)
                                col_l, col_r = st.columns(2)
                                with col_l:
                                    st.markdown('<div class="analysis-card"><h3>Intelligence Report</h3>', unsafe_allow_html=True)
                                    st.markdown("#### 🏢 Turnaround Bottlenecks")
                                    t_bots = pat_df[pat_df['T_Var'] > 5].groupby('Origin').size()
                                    for o, count in t_bots.items():
                                        if count > sample_count * 0.2:
                                            avg = pat_df[pat_df['Origin'] == o]['T_Var'].mean()
                                            msg = f"Airport {o}: Structural ground loss in {int(count/sample_count*100)}% of flights. Avg: +{avg:.1f}m."
                                            st.markdown(f'<div class="pattern-item">{msg}</div>', unsafe_allow_html=True); intelligence_text += "- " + msg + "\n"
                                    st.markdown("#### ✈️ Flight Leg Bottlenecks")
                                    f_bots = pat_df[pat_df['F_Var'] > 5].groupby(['Origin', 'Dest']).size()
                                    for (o, d), count in f_bots.items():
                                        if count > sample_count * 0.2:
                                            avg = pat_df[(pat_df.Origin == o) & (pat_df.Dest == d)]['F_Var'].mean()
                                            msg = f"Sector {o}-{d}: Structural airborne loss in {int(count/sample_count*100)}% of cycles. Avg: +{avg:.1f}m."
                                            st.markdown(f'<div class="pattern-item">{msg}</div>', unsafe_allow_html=True); intelligence_text += "- " + msg + "\n"
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col_r:
                                    st.markdown(f'''<div class="ml-card"><h3>{next_season_code} Prediction</h3><div style="font-size: 3em; font-weight: bold; color: {"#ef4444" if s_pred > 15 else "#10b981"};">+{max(0, s_pred):.1f}m</div><hr><p><b>Logic:</b> R²: {r2:.2f}, Ground Factor: {g_i:.2f}x, Air Factor: {f_i:.2f}x.</p></div>''', unsafe_allow_html=True)
                            pdf_full = create_pdf_report(df_res, flight_display_name, search_date.strftime("%d-%m-%Y"), buffer_summary, intelligence_text, current_season_code, hist_df)
                            st.download_button("Download Detailed Seasonal Report", pdf_full, f"{base_filename} Season.pdf", "application/pdf", use_container_width=True)
    except Exception as e: st.error(f"Error: {e}")
else: st.info("Upload CSV.")
