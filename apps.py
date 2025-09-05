import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Set page config for better appearance
st.set_page_config(
    page_title="TikTok Shop Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Modern CSS with dark mode support
def get_css():
    base_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    .main-header {
        font-size: 48px !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(118, 75, 162, 0.8)); }
    }

    .main-subheader {
        font-size: 20px !important;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 300;
    }

    .metric-card {
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        transition: left 0.5s;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
    }

    .metric-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
    }

    .metric-description {
        font-size: 13px;
        font-weight: 400;
        margin-top: 10px;
    }

    .section-header {
        font-size: 28px !important;
        font-weight: 700;
        margin-top: 50px;
        margin-bottom: 30px;
        padding-bottom: 15px;
        text-align: center;
    }

    .subsection-header {
        font-size: 22px !important;
        font-weight: 600;
        margin-top: 35px;
        margin-bottom: 20px;
        text-align: center;
    }

    .chart-container {
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }

    .chart-container:hover {
        transform: translateY(-5px);
    }

    .insight-card {
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
    }

    .insight-header {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 20px;
        text-align: center;
    }

    .insight-item {
        margin-bottom: 15px;
        padding: 15px;
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .insight-item:hover {
        transform: translateX(5px);
    }

    .insight-highlight {
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
    }

    .insight-warning {
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
    }

    .mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        border-radius: 50px;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .mode-toggle:hover {
        transform: scale(1.05);
    }

    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4) !important;
    }

    .footer {
        text-align: center;
        padding: 30px;
        font-size: 14px;
        margin-top: 50px;
        border-radius: 20px 20px 0 0;
    }
    """

    if st.session_state.dark_mode:
        return base_css + """
        .stApp {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%) !important;
        }

        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .main-subheader {
            color: #b3b3b3;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-card::before {
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        }

        .metric-card:hover {
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(102, 126, 234, 0.5);
        }

        .metric-label {
            color: #e0e0e0;
        }

        .metric-description {
            color: #999;
        }

        .section-header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }

        .subsection-header {
            color: #4facfe;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .chart-container:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
        }

        .insight-card {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.05) 100%);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(79, 172, 254, 0.2);
        }

        .insight-header {
            color: #4facfe;
        }

        .insight-item {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
        }

        .insight-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .insight-highlight {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(139, 195, 74, 0.1) 100%);
            color: #e8f5e8;
        }

        .insight-warning {
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 193, 7, 0.1) 100%);
            color: #fff3e0;
        }

        .mode-toggle {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }

        .mode-toggle:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .footer {
            color: #666;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
        }
        </style>
        """
    else:
        return base_css + """
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }

        .main-header {
            color: white;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }

        .main-subheader {
            color: rgba(255, 255, 255, 0.8);
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .metric-card::before {
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        }

        .metric-card:hover {
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            border-color: rgba(102, 126, 234, 0.5);
        }

        .metric-label {
            color: #555;
        }

        .metric-description {
            color: #777;
        }

        .section-header {
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        }

        .subsection-header {
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .chart-container:hover {
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        }

        .insight-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .insight-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .insight-item {
            background: rgba(102, 126, 234, 0.05);
            border: 1px solid rgba(102, 126, 234, 0.1);
        }

        .insight-item:hover {
            background: rgba(102, 126, 234, 0.1);
        }

        .insight-highlight {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(139, 195, 74, 0.05) 100%);
        }

        .insight-warning {
            background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        }

        .mode-toggle {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #333;
        }

        .mode-toggle:hover {
            background: rgba(255, 255, 255, 1);
        }

        .footer {
            color: rgba(255, 255, 255, 0.7);
            border-top: 1px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        </style>
        """

# Apply CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Dark mode toggle
col_space, col_toggle = st.columns([9, 1])
with col_toggle:
    if st.button("🌓", key="dark_mode_toggle", help="Toggle Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Function to format numbers as Rupiah
def format_rupiah(value):
    try:
        if pd.isna(value):
            return "IDR 0"
        return f"IDR {value:,.0f}".replace(",", ".")
    except:
        return value

def format_number(value):
    try:
        if pd.isna(value):
            return "0"
        return f"{value:,.0f}".replace(",", ".")
    except:
        return value

def format_percentage(value):
    try:
        if pd.isna(value):
            return "0%"
        return f"{value:.1f}%"
    except:
        return "0%"

def format_id(value):
    try:
        if pd.isna(value):
            return "0"
        return f"{int(value)}"
    except:
        return str(value)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            df['Order created time(UTC)'] = pd.to_datetime(df['Order created time(UTC)'], format='%Y/%m/%d', errors='coerce')
            df['Order settled time(UTC)'] = pd.to_datetime(df['Order settled time(UTC)'], format='%Y/%m/%d', errors='coerce')
            df['Order created time(UTC)'] = df['Order created time(UTC)'].dt.floor('ms')
            df['Order settled time(UTC)'] = df['Order settled time(UTC)'].dt.floor('ms')
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    return None

# Modern animated header
st.markdown('<h1 class="main-header">🚀 TikTok Shop Analytics Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Next-generation analytics dashboard with AI-powered insights</p>', unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    uploaded_file = st.file_uploader("📁 Upload Analytics Data", type=["xlsx"], help="Upload your TikTok Shop income data")
    
    if uploaded_file is not None:
        st.success("✅ Data loaded successfully!")
        st.info(f"📊 File: {uploaded_file.name}")
    else:
        st.warning("⚠️ Please upload your data file")

# Load and process data
df = load_data(uploaded_file)

if df is not None:
    df_display = df.copy()
    df_display['Order created time(UTC)'] = df_display['Order created time(UTC)'].dt.strftime('%Y-%m-%d')
    df_display['Order settled time(UTC)'] = df_display['Order settled time(UTC)'].dt.strftime('%Y-%m-%d')
    
    # Enhanced sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Smart Filters")
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("📅 From", value=df['Order created time(UTC)'].min().date())
        with date_col2:
            end_date = st.date_input("📅 To", value=df['Order created time(UTC)'].max().date())
        
        filtered_df = df[(df['Order created time(UTC)'].dt.date >= start_date) & (df['Order created time(UTC)'].dt.date <= end_date)]
        
        st.markdown("#### 📊 Order Type")
        order_types = ['All Orders', 'Affiliate Orders', 'Direct Store Orders']
        selected_order_type = st.selectbox("", order_types, index=0)
        
        if selected_order_type == 'Affiliate Orders':
            filtered_df = filtered_df[filtered_df['Affiliate commission'].fillna(0) < 0]
        elif selected_order_type == 'Direct Store Orders':
            filtered_df = filtered_df[filtered_df['Affiliate commission'].fillna(0) == 0]
        
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Records", format_number(len(filtered_df)))
        st.metric("Date Range", f"{(end_date - start_date).days + 1} days")
    
    # Modern KPI Section
    st.markdown('<h2 class="section-header">💎 Key Performance Indicators</h2>', unsafe_allow_html=True)

    total_settlement = filtered_df["Total settlement amount"].sum()
    total_fees = filtered_df["Total fees"].sum()
    fee_percentage = abs(total_fees / total_settlement) * 100 if total_settlement != 0 else 0

    # Enhanced metrics layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">💰 Total Revenue</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(filtered_df["Total revenue"].sum())}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Gross income from all sales</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">💳 Net Settlement</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(total_settlement)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Amount settled to account</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📦 Total Orders</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(len(filtered_df))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Number of transactions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">💵 Avg Order Value</div>', unsafe_allow_html=True)
        avg_order_value = filtered_df["Total revenue"].mean()
        st.markdown(f'<div class="metric-value">{format_rupiah(avg_order_value)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Revenue per transaction</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">💸 Total Fees</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(total_fees)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Platform & service fees</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📊 Fee Percentage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_percentage(fee_percentage)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Fees as % of settlement</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive Revenue Visualization
    st.markdown('<h2 class="section-header">📊 Interactive Revenue Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-header">📈 Advanced Revenue Trend</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not filtered_df.empty:
            revenue_over_time = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum().reset_index()
            revenue_over_time.columns = ['Date', 'Revenue']

            # Create advanced interactive chart with multiple traces
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Revenue Trend', 'Cumulative Revenue'),
                row_heights=[0.7, 0.3],
                vertical_spacing=0.1
            )

            # Main trend line with area fill
            fig.add_trace(
                go.Scatter(
                    x=revenue_over_time['Date'],
                    y=revenue_over_time['Revenue'],
                    mode='lines+markers',
                    name='Daily Revenue',
                    line=dict(color='#667eea', width=3, shape='spline'),
                    marker=dict(size=8, color='#667eea', line=dict(width=2, color='white')),
                    fill='tonexty',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> IDR %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Moving average
            if len(revenue_over_time) > 3:
                revenue_over_time['MA7'] = revenue_over_time['Revenue'].rolling(window=min(7, len(revenue_over_time))).mean()
                fig.add_trace(
                    go.Scatter(
                        x=revenue_over_time['Date'],
                        y=revenue_over_time['MA7'],
                        mode='lines',
                        name='Moving Average',
                        line=dict(color='#ff6b6b', width=2, dash='dash'),
                        hovertemplate='<b>Date:</b> %{x}<br><b>MA:</b> IDR %{y:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )

            # Cumulative revenue
            revenue_over_time['Cumulative'] = revenue_over_time['Revenue'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=revenue_over_time['Date'],
                    y=revenue_over_time['Cumulative'],
                    mode='lines',
                    name='Cumulative Revenue',
                    line=dict(color='#4ecdc4', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(78, 205, 196, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative:</b> IDR %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=700,
                title=dict(text="🚀 Advanced Revenue Analytics", font=dict(size=24, color='#667eea'), x=0.5),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )

            # Update axes
            fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title_text="Date")
            fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', title_text="Revenue (IDR)", tickformat=',.0f')

            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Order Analysis
    st.markdown('<h2 class="section-header">🎯 Smart Order Analytics</h2>', unsafe_allow_html=True)

    affiliate_orders = filtered_df[filtered_df['Affiliate commission'].fillna(0) < 0]
    direct_orders = filtered_df[filtered_df['Affiliate commission'].fillna(0) == 0]

    # Interactive pie chart for order distribution
    st.markdown('<h3 class="subsection-header">🥧 Order Type Distribution</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        order_data = {
            'Type': ['Affiliate Orders', 'Direct Orders'],
            'Count': [len(affiliate_orders), len(direct_orders)],
            'Revenue': [affiliate_orders['Total revenue'].sum(), direct_orders['Total revenue'].sum()]
        }
        
        # Create subplots for count and revenue
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=('Order Count Distribution', 'Revenue Distribution')
        )

        # Order count pie
        fig.add_trace(
            go.Pie(
                labels=order_data['Type'],
                values=order_data['Count'],
                name="Order Count",
                marker=dict(colors=['#ff6b6b', '#4ecdc4']),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )

        # Revenue pie
        fig.add_trace(
            go.Pie(
                labels=order_data['Type'],
                values=order_data['Revenue'],
                name="Revenue",
                marker=dict(colors=['#ff6b6b', '#4ecdc4']),
                hovertemplate='<b>%{label}</b><br>Revenue: IDR %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=dict(text="🎯 Order Type Analysis", font=dict(size=20, color='#667eea'), x=0.5),
            showlegend=True,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced 3D Visualization
    st.markdown('<h2 class="section-header">🌟 Advanced Visualizations</h2>', unsafe_allow_html=True)
    
    # 3D Revenue Analysis
    st.markdown('<h3 class="subsection-header">🎲 3D Revenue Analysis</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if not filtered_df.empty and len(filtered_df) > 10:
            sample_data = filtered_df.sample(min(100, len(filtered_df)))
            
            fig = px.scatter_3d(
                sample_data,
                x='Total revenue',
                y='Total fees',
                z=sample_data['Order created time(UTC)'].dt.dayofyear,
                color='Total revenue',
                color_continuous_scale='Viridis',
                title='🎲 3D Revenue, Fees & Time Analysis',
                labels={
                    'Total revenue': 'Revenue (IDR)',
                    'Total fees': 'Fees (IDR)',
                    'z': 'Day of Year'
                },
                hover_data=['Total revenue', 'Total fees']
            )
            
            fig.update_layout(
                height=600,
                scene=dict(
                    xaxis_title='Revenue',
                    yaxis_title='Fees',
                    zaxis_title='Time (Day of Year)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Need more data points for 3D visualization")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI-Powered Analytics
    st.markdown('<h2 class="section-header">🤖 AI Analytics</h2>', unsafe_allow_html=True)
    
    # Revenue Forecasting
    st.markdown('<h3 class="subsection-header">🔮 Revenue Forecast</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        
        if not filtered_df.empty and len(filtered_df) > 3:
            daily_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
            dates = pd.to_datetime(daily_revenue.index)
            revenues = daily_revenue.values
            
            days_since_start = [(date - dates.min()).days for date in dates]
            X = np.array(days_since_start).reshape(-1, 1)
            y = revenues
            
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            
            last_day = max(days_since_start)
            future_days = np.array([last_day + i for i in range(1, 8)]).reshape(-1, 1)
            future_days_poly = poly_features.transform(future_days)
            future_revenues = model.predict(future_days_poly)
            
            future_dates = [dates.max() + timedelta(days=i) for i in range(1, 8)]
            
            # Create forecast visualization
            fig, ax = plt.subplots(figsize=(14, 8))
            
            ax.plot(dates, revenues, marker='o', linewidth=3, markersize=8, color='#667eea', label='Historical Revenue', alpha=0.8)
            ax.plot(future_dates, future_revenues, marker='s', linewidth=3, markersize=8, color='#ff6b6b', linestyle='--', label='AI Forecast', alpha=0.9)
            
            std_dev = np.std(revenues)
            upper_bound = future_revenues + std_dev
            lower_bound = future_revenues - std_dev
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#ff6b6b', alpha=0.2, label='Confidence Interval')
            
            ax.set_xlabel("Date", fontsize=12, fontweight='bold')
            ax.set_ylabel("Revenue (IDR)", fontsize=12, fontweight='bold')
            ax.set_title("🤖 AI Revenue Forecast with Confidence Intervals", fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'IDR {x:,.0f}'.replace(',', '.')))
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Forecast insights
            avg_forecast = np.mean(future_revenues)
            total_forecast = np.sum(future_revenues)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Expected Daily Revenue", format_rupiah(avg_forecast))
            with col2:
                st.metric("💰 7-Day Forecast", format_rupiah(total_forecast))
        else:
            st.info("📈 Need more data for AI forecasting")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation Analysis
    st.markdown('<h3 class="subsection-header">🔗 Smart Correlation Analysis</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            relevant_cols = [col for col in numeric_cols if col in [
                'Total revenue', 'Total settlement amount', 'Total fees', 
                'TikTok Shop commission fee', 'Shipping cost', 'Affiliate commission'
            ]]
            
            if len(relevant_cols) >= 2:
                corr_matrix = filtered_df[relevant_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="🔗 Interactive Correlation Matrix"
                )
                
                fig.update_layout(
                    height=600,
                    coloraxis_colorbar=dict(title="Correlation"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                fig.update_traces(
                    hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key insights
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if corr_pairs:
                    st.markdown("#### Key Correlation Insights:")
                    for pair in corr_pairs:
                        strength = "Strong" if abs(pair['Correlation']) > 0.7 else "Moderate"
                        direction = "Positive" if pair['Correlation'] > 0 else "Negative"
                        st.markdown(f"• **{strength} {direction}**: {pair['Variable 1']} ↔ {pair['Variable 2']} ({format_percentage(abs(pair['Correlation'])*100)})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fee Analysis Section
    st.markdown('<h2 class="section-header">💸 Fee Analysis</h2>', unsafe_allow_html=True)

    # Fee overview metrics
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">💸 Total Fees</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(total_fees)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">All platform and service fees combined</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">📊 Fee Efficiency</div>', unsafe_allow_html=True)
        total_revenue = filtered_df['Total revenue'].sum()
        # Calculate fee percentage as percentage of settlement (like original logic)
        fee_percentage = abs(total_fees / total_settlement) * 100 if total_settlement != 0 else 0
        # Calculate fee efficiency as percentage of revenue
        fee_efficiency = (abs(total_fees) / total_revenue) * 100 if total_revenue != 0 else 0
        st.markdown(f'<div class="metric-value">{format_percentage(fee_efficiency)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Fees as % of total revenue</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fee breakdown
    fee_breakdown_cols = [
        'TikTok Shop commission fee',
        'Affiliate commission',
        'Fix Infrastructure Fee',
        'Dynamic Commission',
        'Voucher Xtra Service Fee',
        'Shipping cost'
    ]

    existing_fee_breakdown_cols = [col for col in fee_breakdown_cols if col in filtered_df.columns]

    if existing_fee_breakdown_cols:
        fee_totals = filtered_df[existing_fee_breakdown_cols].sum()

        # Fee breakdown chart - Interactive version
        st.markdown('<h3 class="subsection-header">📊 Fee Composition Breakdown</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            if len(existing_fee_breakdown_cols) > 1:
                # Create interactive pie chart for fee breakdown
                fig = go.Figure()

                fee_values = [abs(fee_totals[col]) for col in existing_fee_breakdown_cols]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                fig.add_trace(go.Pie(
                    labels=existing_fee_breakdown_cols,
                    values=fee_values,
                    marker=dict(colors=colors),
                    hovertemplate='<b>%{label}</b><br>Amount: IDR %{value:,.0f}<br>Percentage: %{percent}<extra></extra>',
                    textinfo='percent+label',
                    textposition='inside'
                ))

                fig.update_layout(
                    title=dict(text="💸 Fee Distribution by Type", font=dict(size=20, color='#667eea'), x=0.5),
                    height=600,
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )

                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Fee breakdown table
        st.markdown('<h3 class="subsection-header">Fee Details</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fee_df = pd.DataFrame({
                'Fee Type': existing_fee_breakdown_cols,
                'Total Amount': [fee_totals[col] for col in existing_fee_breakdown_cols]
            })
            fee_df['Total Amount (IDR)'] = fee_df['Total Amount'].apply(format_rupiah)
            st.dataframe(fee_df[['Fee Type', 'Total Amount (IDR)']], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No specific fee data available in the current dataset.")

    # Advanced Insights Section - Redesigned for Better Visual Appeal
    st.markdown('<h2 class="section-header">💡 Advanced Insights</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 16px; color: #666; margin-bottom: 30px; text-align: center;">AI-powered analysis and strategic recommendations for your business</p>', unsafe_allow_html=True)

    if not filtered_df.empty:
        # Create organized insight cards in a grid layout
        col1, col2 = st.columns(2)

        # Revenue Performance Card
        with col1:
            with st.container():
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-header">💰 Revenue Performance</div>', unsafe_allow_html=True)

                total_revenue = filtered_df['Total revenue'].sum()
                avg_revenue = filtered_df['Total revenue'].mean()
                max_revenue = filtered_df['Total revenue'].max()

                # Key metrics in a clean layout
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Total Revenue", format_rupiah(total_revenue))
                    st.metric("Average Order", format_rupiah(avg_revenue))
                with metric_col2:
                    st.metric("Highest Order", format_rupiah(max_revenue))
                    st.metric("Total Orders", format_number(len(filtered_df)))

                # Revenue trend indicator
                if len(filtered_df) > 1:
                    revenue_trend = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
                    revenue_growth = ((revenue_trend.iloc[-1] - revenue_trend.iloc[0]) / revenue_trend.iloc[0]) * 100 if revenue_trend.iloc[0] != 0 else 0

                    if revenue_growth > 0:
                        st.markdown('<div class="insight-highlight" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
                        st.markdown(f"### 🚀 Revenue Growth\n**+{format_percentage(revenue_growth)}** over the period")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif revenue_growth < 0:
                        st.markdown('<div class="insight-warning" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ Revenue Decline\n**{format_percentage(revenue_growth)}** over the period")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # Fee Analysis Card
        with col2:
            with st.container():
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-header">💸 Fee Analysis</div>', unsafe_allow_html=True)

                total_revenue = filtered_df['Total revenue'].sum()
                fee_efficiency = (abs(total_fees) / total_revenue) * 100 if total_revenue != 0 else 0

                # Fee metrics
                fee_col1, fee_col2 = st.columns(2)
                with fee_col1:
                    st.metric("Total Fees", format_rupiah(total_fees))
                    st.metric("Fee Percentage", format_percentage(fee_percentage))
                with fee_col2:
                    if existing_fee_breakdown_cols:
                        # Use absolute values for fee comparison (since some fees might be negative)
                        abs_fee_totals = fee_totals.abs()
                        highest_fee_type = abs_fee_totals.idxmax()
                        highest_fee_value = fee_totals[highest_fee_type]  # Keep original sign
                        st.metric("Highest Fee", highest_fee_type)
                        st.metric("Fee Amount", format_rupiah(highest_fee_value))
                    else:
                        st.metric("Highest Fee", "No data")
                        st.metric("Fee Amount", format_rupiah(0))

                # Fee efficiency indicator
                if fee_efficiency > 20:
                    st.markdown('<div class="insight-warning" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ High Fee Burden\nConsider optimizing pricing strategy")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif fee_efficiency < 10:
                    st.markdown('<div class="insight-highlight" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
                    st.markdown("### ✅ Low Fee Burden\nEfficient cost management")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        # Order Analysis Card - Full Width
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-header">📦 Order Analysis</div>', unsafe_allow_html=True)

            # Order metrics in columns
            order_col1, order_col2, order_col3, order_col4 = st.columns(4)

            with order_col1:
                st.metric("Total Orders", format_number(len(filtered_df)))

            with order_col2:
                affiliate_ratio = len(affiliate_orders) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
                st.metric("Affiliate Orders", format_percentage(affiliate_ratio))

            with order_col3:
                direct_ratio = len(direct_orders) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
                st.metric("Direct Orders", format_percentage(direct_ratio))

            with order_col4:
                avg_order_value = filtered_df["Total revenue"].mean()
                st.metric("Avg Order Value", format_rupiah(avg_order_value))

            # Order type insights
            if len(filtered_df) > 0:
                insight_col1, insight_col2 = st.columns(2)

                with insight_col1:
                    if affiliate_ratio > 50:
                        st.markdown('<div class="insight-highlight" style="text-align: center;">', unsafe_allow_html=True)
                        st.markdown("### 📈 Strong Affiliate\nConsider expanding affiliate marketing")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif direct_ratio > 70:
                        st.markdown('<div class="insight-highlight" style="text-align: center;">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Strong Direct Sales\nHigh brand recognition")
                        st.markdown('</div>', unsafe_allow_html=True)

                with insight_col2:
                    avg_affiliate_value = affiliate_orders['Total revenue'].mean() if not affiliate_orders.empty else 0
                    avg_direct_value = direct_orders['Total revenue'].mean() if not direct_orders.empty else 0

                    st.markdown("#### 💰 Average Order Values")
                    st.write(f"**Affiliate:** {format_rupiah(avg_affiliate_value)}")
                    st.write(f"**Direct:** {format_rupiah(avg_direct_value)}")

            st.markdown('</div>', unsafe_allow_html=True)

        # Time-Based & Strategic Insights - Side by Side
        st.markdown("<br>", unsafe_allow_html=True)
        time_col, strategy_col = st.columns(2)

        # Time-Based Insights
        with time_col:
            with st.container():
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-header">📅 Time-Based Insights</div>', unsafe_allow_html=True)

                if len(filtered_df) > 1:
                    daily_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
                    best_day = daily_revenue.idxmax()
                    best_day_revenue = daily_revenue.max()

                    st.markdown("#### 🌟 Best Performance")
                    st.metric("Top Sales Day", str(best_day))
                    st.metric("Revenue", format_rupiah(best_day_revenue))

                    if len(daily_revenue) >= 7:
                        filtered_df['Weekday'] = filtered_df['Order created time(UTC)'].dt.day_name()
                        weekday_revenue = filtered_df.groupby('Weekday')['Total revenue'].sum()
                        best_weekday = weekday_revenue.idxmax()

                        st.markdown("#### 📊 Weekly Pattern")
                        st.metric("Best Day", best_weekday)
                        st.metric("Revenue", format_rupiah(weekday_revenue.max()))

                st.markdown('</div>', unsafe_allow_html=True)

        # Strategic Recommendations
        with strategy_col:
            with st.container():
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-header">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)

                recommendations = []

                if total_revenue != 0:
                    if fee_efficiency > 20:
                        recommendations.append(("💰 High Fee Burden", "Review pricing strategy to maintain margins"))
                    elif fee_efficiency < 10:
                        recommendations.append(("✅ Excellent Efficiency", "Cost management is performing well"))

                if len(filtered_df) > 0:
                    if affiliate_ratio > 50:
                        recommendations.append(("📈 Strong Affiliate", "Expand affiliate marketing program"))
                    elif affiliate_ratio < 20:
                        recommendations.append(("👥 Low Affiliate", "Explore new affiliate partnerships"))

                if len(filtered_df) > 1:
                    revenue_trend = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
                    revenue_growth = ((revenue_trend.iloc[-1] - revenue_trend.iloc[0]) / revenue_trend.iloc[0]) * 100 if revenue_trend.iloc[0] != 0 else 0

                    if revenue_growth > 10:
                        recommendations.append(("🚀 Strong Growth", "Scale successful strategies"))
                    elif revenue_growth < -5:
                        recommendations.append(("📉 Revenue Decline", "Investigate causes and implement fixes"))

                if recommendations:
                    for title, desc in recommendations:
                        st.markdown('<div class="insight-item" style="margin-bottom: 10px;">', unsafe_allow_html=True)
                        st.markdown(f"**{title}**")
                        st.write(desc)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="insight-highlight" style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown("### ✅ Excellent Performance\nYour store is performing well!")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    # Anomaly Detection
    st.markdown('<h3 class="subsection-header">🔍 Anomaly Detection</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        if not filtered_df.empty:
            revenue_data = filtered_df['Total revenue']
            Q1 = revenue_data.quantile(0.25)
            Q3 = revenue_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            anomalies = filtered_df[(revenue_data < lower_bound) | (revenue_data > upper_bound)]

            if not anomalies.empty:
                st.markdown('<div class="insight-warning">', unsafe_allow_html=True)
                st.write(f"⚠️ Detected {len(anomalies)} anomalous orders that deviate significantly from normal revenue patterns:")
                st.markdown('</div>', unsafe_allow_html=True)

                anomaly_display = anomalies[['Order created time(UTC)', 'Total revenue']].copy()
                anomaly_display['Order created time(UTC)'] = anomaly_display['Order created time(UTC)'].dt.strftime('%Y-%m-%d')
                anomaly_display['Total revenue'] = anomaly_display['Total revenue'].apply(format_rupiah)
                st.dataframe(anomaly_display, use_container_width=True)

                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write("These orders may require investigation for potential issues or opportunities.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                st.write("✅ No significant anomalies detected in revenue data. Revenue patterns are consistent.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No data available for anomaly detection.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Customer Segmentation
    st.markdown('<h3 class="subsection-header">👥 Customer Segmentation</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        if not filtered_df.empty and len(filtered_df) > 5:
            try:
                quantiles = filtered_df['Total revenue'].quantile([0, 0.33, 0.67, 1]).values
                unique_bins = pd.Series(quantiles).drop_duplicates().values
                if len(unique_bins) >= 3:
                    filtered_df['Revenue_Segment'] = pd.cut(
                        filtered_df['Total revenue'],
                        bins=unique_bins,
                        labels=['Low Value', 'Medium Value', 'High Value'][:len(unique_bins)-1],
                        include_lowest=True
                    )
                else:
                    median_val = filtered_df['Total revenue'].median()
                    filtered_df['Revenue_Segment'] = pd.cut(
                        filtered_df['Total revenue'],
                        bins=[filtered_df['Total revenue'].min(), median_val, filtered_df['Total revenue'].max()],
                        labels=['Low Value', 'High Value'],
                        include_lowest=True
                    )
            except Exception:
                median_val = filtered_df['Total revenue'].median()
                filtered_df['Revenue_Segment'] = pd.cut(
                    filtered_df['Total revenue'],
                    bins=[filtered_df['Total revenue'].min(), median_val, filtered_df['Total revenue'].max()],
                    labels=['Low Value', 'High Value'],
                    include_lowest=True
                )

            segment_stats = filtered_df.groupby('Revenue_Segment', observed=True).agg({
                'Total revenue': ['count', 'mean', 'sum']
            }).round(2)

            segment_stats.columns = ['Order Count', 'Avg Revenue', 'Total Revenue']
            segment_stats['Avg Revenue'] = segment_stats['Avg Revenue'].apply(format_rupiah)
            segment_stats['Total Revenue'] = segment_stats['Total Revenue'].apply(format_rupiah)

            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write("Customer segmentation based on order value:")
            st.markdown('</div>', unsafe_allow_html=True)

            st.dataframe(segment_stats, use_container_width=True)

            # Segment visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts = filtered_df['Revenue_Segment'].value_counts()
            bars = ax.bar(segment_counts.index, segment_counts.values, color=['#1f77b4', '#2ca02c', '#d62728'])
            ax.set_xlabel("Customer Segment")
            ax.set_ylabel("Number of Orders")
            ax.set_title("Customer Segmentation by Order Value")

            for bar in bars:
                height = bar.get_height()
                ax.annotate(format_number(height),
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)

            # Segment insights
            available_segments = segment_counts.index.tolist()

            if 'High Value' in available_segments:
                high_value_pct = (segment_counts['High Value'] / len(filtered_df)) * 100
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"🏆 High-value customers make up {format_percentage(high_value_pct)} of orders")
                st.markdown('</div>', unsafe_allow_html=True)

                if high_value_pct > 20:
                    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                    st.write("🎯 Strong high-value customer base. Consider VIP programs to retain these customers.")
                    st.markdown('</div>', unsafe_allow_html=True)

            if 'Low Value' in available_segments:
                low_value_pct = (segment_counts['Low Value'] / len(filtered_df)) * 100
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📊 Low-value customers make up {format_percentage(low_value_pct)} of orders")
                st.markdown('</div>', unsafe_allow_html=True)

            if 'Medium Value' in available_segments:
                medium_value_pct = (segment_counts['Medium Value'] / len(filtered_df)) * 100
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📈 Medium-value customers make up {format_percentage(medium_value_pct)} of orders")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Insufficient data for customer segmentation. Need at least 6 orders for meaningful analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Data Tables with Modern Design
    st.markdown('<h2 class="section-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📋 Raw Data", "🔍 Insights"])
    
    with tab1:
        st.markdown('<h3 class="subsection-header">Statistical Overview</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            summary_stats = filtered_df[numeric_columns].describe()
            
            for col in summary_stats.columns:
                if 'revenue' in col.lower() or 'fee' in col.lower() or 'amount' in col.lower():
                    summary_stats[col] = summary_stats[col].apply(format_rupiah)
                elif 'id' in col.lower():
                    summary_stats[col] = summary_stats[col].apply(format_id)
                else:
                    summary_stats[col] = summary_stats[col].apply(format_number)
            
            st.dataframe(summary_stats, use_container_width=True, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h3 class="subsection-header">Complete Dataset</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.dataframe(df_display, use_container_width=True, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h3 class="subsection-header">Smart Insights</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📅 Date Range: {filtered_df['Order created time(UTC)'].min().date()} to {filtered_df['Order created time(UTC)'].max().date()}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"🔢 Total Records: {format_number(len(filtered_df))}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"💰 Highest Order: {format_rupiah(filtered_df['Total revenue'].max())}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📉 Lowest Order: {format_rupiah(filtered_df['Total revenue'].min())}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Please upload your sample_income.xlsx file to begin analysis")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
             caption="Upload your TikTok Shop data to unlock advanced analytics", use_container_width=True)

# AI-Powered Predictions & Recommendations Section
st.markdown('<h2 class="section-header">🤖 AI Prediksi & Rekomendasi Bisnis</h2>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 16px; color: #666; margin-bottom: 30px; text-align: center;">Analisis cerdas berbasis AI untuk strategi bisnis TikTok Shop Anda</p>', unsafe_allow_html=True)

if not filtered_df.empty:
    # AI Predictions Container
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">🔮 Prediksi Tren Penjualan</div>', unsafe_allow_html=True)

        # Revenue trend analysis for predictions
        if len(filtered_df) > 3:
            daily_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
            dates = pd.to_datetime(daily_revenue.index)
            revenues = daily_revenue.values

            # Simple trend analysis
            if len(revenues) > 1:
                recent_trend = (revenues[-1] - revenues[0]) / len(revenues)
                trend_direction = "meningkat" if recent_trend > 0 else "menurun"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tren Penjualan", f"Rp {recent_trend:,.0f}/hari")
                    st.metric("Arah Tren", trend_direction.title())

                with col2:
                    avg_daily = np.mean(revenues)
                    predicted_monthly = avg_daily * 30
                    st.metric("Rata-rata Harian", format_rupiah(avg_daily))
                    st.metric("Prediksi Bulanan", format_rupiah(predicted_monthly))

        st.markdown('</div>', unsafe_allow_html=True)

    # AI Recommendations Container
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">🎯 Rekomendasi Strategi Bisnis</div>', unsafe_allow_html=True)

        recommendations = []

        # Revenue-based recommendations
        total_revenue = filtered_df['Total revenue'].sum()
        avg_order_value = filtered_df['Total revenue'].mean()

        if total_revenue > 10000000:  # Over 10 million IDR
            recommendations.append(("🚀 Pertumbuhan Kuat", "Bisnis Anda menunjukkan performa yang sangat baik. Pertimbangkan untuk meningkatkan investasi pemasaran untuk mempercepat pertumbuhan."))
        elif total_revenue > 5000000:  # Over 5 million IDR
            recommendations.append(("📈 Potensi Besar", "Performa bisnis Anda cukup baik. Fokus pada optimalisasi produk dan pemasaran untuk mencapai target yang lebih tinggi."))
        else:
            recommendations.append(("🎯 Optimasi Diperlukan", "Tingkatkan strategi pemasaran dan optimalkan produk untuk meningkatkan penjualan."))

        # Fee analysis recommendations
        if total_settlement != 0:
            fee_percentage = abs(total_fees / total_settlement) * 100
            if fee_percentage > 20:
                recommendations.append(("💰 Biaya Platform Tinggi", "Biaya platform cukup tinggi. Pertimbangkan strategi harga yang lebih kompetitif atau cari alternatif platform dengan biaya lebih rendah."))
            elif fee_percentage < 10:
                recommendations.append(("✅ Efisiensi Biaya Baik", "Biaya platform Anda cukup efisien. Pertahankan strategi saat ini dan fokus pada peningkatan penjualan."))

        # Order type recommendations
        if len(filtered_df) > 0:
            affiliate_ratio = len(affiliate_orders) / len(filtered_df) * 100
            if affiliate_ratio > 50:
                recommendations.append(("🤝 Affiliate Marketing Kuat", "Program affiliate Anda sangat efektif. Tingkatkan kolaborasi dengan influencer dan perluas jaringan affiliate."))
            elif affiliate_ratio < 20:
                recommendations.append(("👥 Perluas Affiliate", "Tingkatkan program affiliate dengan mencari influencer baru dan membuat konten yang lebih menarik."))

        # Average order value recommendations
        if avg_order_value < 100000:  # Under 100k IDR
            recommendations.append(("🛒 Tingkatkan Nilai Pesanan", "Rata-rata nilai pesanan masih rendah. Pertimbangkan bundling produk atau upsell untuk meningkatkan nilai transaksi."))
        elif avg_order_value > 500000:  # Over 500k IDR
            recommendations.append(("💎 Pelanggan Premium", "Rata-rata nilai pesanan sangat baik. Fokus pada retensi pelanggan dan program loyalitas."))

        # Time-based recommendations
        if len(filtered_df) > 7:
            weekday_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.day_name())['Total revenue'].sum()
            best_weekday = weekday_revenue.idxmax()
            recommendations.append(("📅 Optimasi Waktu", f"Hari {best_weekday} adalah hari tersibuk. Tingkatkan promosi pada hari tersebut untuk hasil maksimal."))

        # Display recommendations
        if recommendations:
            for i, (title, desc) in enumerate(recommendations, 1):
                st.markdown('<div class="insight-item" style="margin-bottom: 15px;">', unsafe_allow_html=True)
                st.markdown(f"**{i}. {title}**")
                st.write(desc)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-highlight" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("### ✅ Analisis Selesai\nSemua indikator bisnis Anda dalam kondisi baik!")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Future Business Strategy Container
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">📋 Strategi Bisnis Jangka Panjang</div>', unsafe_allow_html=True)

        strategy_col1, strategy_col2 = st.columns(2)

        with strategy_col1:
            st.markdown("#### 🎯 Target Jangka Pendek (1-3 bulan)")
            st.markdown("""
            • Optimalkan konten produk dengan foto/video berkualitas tinggi
            • Tingkatkan engagement dengan pelanggan melalui live streaming
            • Analisis produk terlaris dan tingkatkan stok
            • Implementasikan program loyalitas pelanggan
            """)

            st.markdown("#### 📈 Target Jangka Menengah (3-6 bulan)")
            st.markdown("""
            • Kembangkan variasi produk berdasarkan tren pasar
            • Bangun brand awareness melalui kolaborasi influencer
            • Optimalkan harga dan promosi berdasarkan data penjualan
            • Tingkatkan konversi dengan A/B testing
            """)

        with strategy_col2:
            st.markdown("#### 🚀 Target Jangka Panjang (6-12 bulan)")
            st.markdown("""
            • Diversifikasi channel penjualan (marketplace lain)
            • Kembangkan produk private label
            • Implementasikan automation marketing
            • Bangun komunitas loyal pelanggan
            """)

            st.markdown("#### ⚠️ Peringatan & Monitoring")
            st.markdown("""
            • Pantau terus biaya platform dan margin keuntungan
            • Analisis tren kompetitor secara berkala
            • Update strategi berdasarkan feedback pelanggan
            • Siapkan rencana contingency untuk perubahan algoritma
            """)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Silakan upload file data TikTok Shop Anda untuk mendapatkan analisis AI dan rekomendasi bisnis")

# Modern footer
st.markdown('<div class="footer">🚀 TikTok Shop Analytics Pro | Didukung AI & Data Science Modern</div>', unsafe_allow_html=True)
