import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Set page config for better appearance
st.set_page_config(
    page_title="TikTok Shop Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .main-subheader {
        font-size: 18px !important;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        border: 1px solid #e1e8f0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 18px;
        font-weight: 600;
        color: #555;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 5px;
    }
    .metric-description {
        font-size: 14px;
        color: #777;
        margin-top: 10px;
    }
    .section-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 40px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e1e8f0;
    }
    .subsection-header {
        font-size: 20px !important;
        font-weight: 600;
        color: #ff7f0e;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #777;
        font-size: 14px;
        margin-top: 40px;
        border-top: 1px solid #e1e8f0;
    }
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .insight-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #d1e7ff;
    }
    .insight-header {
        font-size: 18px;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 15px;
    }
    .insight-item {
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 8px;
        background-color: rgba(31, 119, 180, 0.05);
    }
    .insight-highlight {
        background-color: rgba(44, 160, 44, 0.1);
        border-left: 4px solid #2ca02c;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .insight-warning {
        background-color: rgba(255, 127, 14, 0.1);
        border-left: 4px solid #ff7f0e;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .ai-header {
        font-size: 24px !important;
        font-weight: bold;
        color: #9467bd;
        margin-top: 40px;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e1e8f0;
        text-align: center;
    }
    .ai-subheader {
        font-size: 20px !important;
        font-weight: 600;
        color: #d62728;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to format numbers as Rupiah
def format_rupiah(value):
    try:
        if pd.isna(value):
            return "IDR 0"
        return f"IDR {value:,.0f}".replace(",", ".")
    except:
        return value

# Function to format numbers with commas (for quantity, etc.)
def format_number(value):
    try:
        if pd.isna(value):
            return "0"
        return f"{value:,.0f}".replace(",", ".")
    except:
        return value

# Function to format percentages
def format_percentage(value):
    try:
        if pd.isna(value):
            return "0%"
        return f"{value:.1f}%"
    except:
        return "0%"

# Function to format IDs (no thousand separators)
def format_id(value):
    try:
        if pd.isna(value):
            return "0"
        # Format as integer without thousand separators
        return f"{int(value)}"
    except:
        return str(value)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Strip column names to avoid trailing spaces issues
            df.columns = df.columns.str.strip()
            # Convert date columns to datetime from string format yyyy/mm/dd
            df['Order created time(UTC)'] = pd.to_datetime(df['Order created time(UTC)'], format='%Y/%m/%d', errors='coerce')
            df['Order settled time(UTC)'] = pd.to_datetime(df['Order settled time(UTC)'], format='%Y/%m/%d', errors='coerce')
            # Convert to datetime64[ns] to avoid pyarrow conversion issues
            df['Order created time(UTC)'] = df['Order created time(UTC)'].dt.floor('ms')
            df['Order settled time(UTC)'] = df['Order settled time(UTC)'].dt.floor('ms')
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    else:
        return None

# Main app
st.markdown('<h1 class="main-header">📊 TikTok Shop Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subheader">Comprehensive insights for your TikTok Shop business performance</p>', unsafe_allow_html=True)

# Sidebar for file upload and filters
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your sample_income.xlsx file", type=["xlsx"])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
    else:
        st.info("Please upload your Excel file to begin analysis")

# Load data
df = load_data(uploaded_file)

if df is not None:
    # Create display version with datetime as string to avoid pyarrow issues
    df_display = df.copy()
    df_display['Order created time(UTC)'] = df_display['Order created time(UTC)'].dt.strftime('%Y-%m-%d')
    df_display['Order settled time(UTC)'] = df_display['Order settled time(UTC)'].dt.strftime('%Y-%m-%d')
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        # Date range filter
        start_date = st.date_input("Start Date", value=df['Order created time(UTC)'].min().date())
        end_date = st.date_input("End Date", value=df['Order created time(UTC)'].max().date())
        
        # Filter data based on date range
        filtered_df = df[(df['Order created time(UTC)'].dt.date >= start_date) & (df['Order created time(UTC)'].dt.date <= end_date)]
        
        # Order type filter
        order_types = ['All', 'Affiliate Orders', 'Direct Store Orders']
        selected_order_type = st.selectbox("Order Type", order_types)
        
        if selected_order_type == 'Affiliate Orders':
            filtered_df = filtered_df[filtered_df['Affiliate commission'].fillna(0) < 0]
        elif selected_order_type == 'Direct Store Orders':
            filtered_df = filtered_df[filtered_df['Affiliate commission'].fillna(0) == 0]
    
    # Main dashboard
    # Key metrics
    st.markdown('<h2 class="section-header">📈 Key Performance Metrics</h2>', unsafe_allow_html=True)
    
    # Calculate fee percentage
    total_settlement = filtered_df["Total settlement amount"].sum()
    total_fees = filtered_df["Total fees"].sum()
    fee_percentage = 0
    if total_settlement != 0:
        fee_percentage = abs(total_fees / total_settlement) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Revenue</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(filtered_df["Total revenue"].sum())}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Gross income from sales</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Settlement</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(total_settlement)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Net amount settled to account</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Fees</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_rupiah(total_fees)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">All platform and service fees</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Fee Percentage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_percentage(fee_percentage)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Fees as % of settlement</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Orders</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(len(filtered_df))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Number of transactions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Revenue breakdown
    st.markdown('<h2 class="section-header">💰 Revenue Analysis</h2>', unsafe_allow_html=True)
    
    # Revenue trend chart
    st.markdown('<h3 class="subsection-header">Revenue Trend Over Time</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not filtered_df.empty:
            revenue_over_time = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(revenue_over_time.index, revenue_over_time.values, marker='o', linewidth=2, markersize=6, color='#1f77b4')
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue (IDR)")
            ax.grid(True, alpha=0.3)
            # Format y-axis as Rupiah
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'IDR {x:,.0f}'.replace(',', '.')))
            plt.xticks(rotation=45)
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Order breakdown
    st.markdown('<h2 class="section-header">📦 Order Analysis</h2>', unsafe_allow_html=True)
    
    # Adjust affiliate_orders and direct_orders based on Affiliate commission logic
    affiliate_orders = filtered_df[filtered_df['Affiliate commission'].fillna(0) < 0]
    direct_orders = filtered_df[filtered_df['Affiliate commission'].fillna(0) == 0]
    
    # Order metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Affiliate Orders</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(len(affiliate_orders))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Orders through affiliate links</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Direct Store Orders</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(len(direct_orders))}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Direct customer purchases</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Affiliate Ratio</div>', unsafe_allow_html=True)
        if len(filtered_df) > 0:
            affiliate_ratio = len(affiliate_orders) / len(filtered_df) * 100
            st.markdown(f'<div class="metric-value">{format_percentage(affiliate_ratio)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">0%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-description">Percentage of affiliate orders</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Order type revenue comparison
    st.markdown('<h3 class="subsection-header">Revenue by Order Type</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if not affiliate_orders.empty or not direct_orders.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            order_types = ['Affiliate Orders', 'Direct Store Orders']
            revenues = [affiliate_orders['Total revenue'].sum(), direct_orders['Total revenue'].sum()]
            bars = ax.bar(order_types, revenues, color=['#d62728', '#2ca02c'])
            ax.set_ylabel("Revenue (IDR)")
            ax.set_title("Revenue by Order Type")
            # Format y-axis as Rupiah
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'IDR {x:,.0f}'.replace(',', '.')))
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(format_rupiah(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fee breakdown
    st.markdown('<h2 class="section-header">💸 Fee Breakdown</h2>', unsafe_allow_html=True)
    
    # Fee metrics
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Fees</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{format_rupiah(total_fees)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-description">All platform and service fees combined</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Specific fee breakdown
    fee_breakdown_cols = [
        'TikTok Shop commission fee',
        'Affiliate commission',
        'Fix Infrastructure Fee',
        'Dynamic Commission',
        'Voucher Xtra Service Fee',
        'Shipping cost'
    ]
    
    # Filter only existing columns
    existing_fee_breakdown_cols = [col for col in fee_breakdown_cols if col in filtered_df.columns]
    
    if existing_fee_breakdown_cols:
        fee_totals = filtered_df[existing_fee_breakdown_cols].sum()
        
        # Create a DataFrame for better display
        fee_df = pd.DataFrame({
            'Fee Type': existing_fee_breakdown_cols,
            'Total Amount': [fee_totals[col] for col in existing_fee_breakdown_cols]
        })
        
        # Format the amounts as Rupiah
        fee_df['Total Amount (IDR)'] = fee_df['Total Amount'].apply(format_rupiah)
        
        # Fee breakdown chart
        st.markdown('<h3 class="subsection-header">Fee Composition</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Create a pie chart for fee breakdown
            if len(existing_fee_breakdown_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                # Take absolute values for the pie chart
                fee_values = [abs(fee_totals[col]) for col in existing_fee_breakdown_cols]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                wedges, texts, autotexts = ax.pie(fee_values, labels=existing_fee_breakdown_cols, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                # Improve label styling
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fee breakdown table
        st.markdown('<h3 class="subsection-header">Fee Details</h3>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.dataframe(fee_df[['Fee Type', 'Total Amount (IDR)']], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No specific fee data available in the current dataset.")
    
    # Data tables
    st.markdown('<h2 class="section-header">📋 Detailed Data</h2>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown('<h3 class="subsection-header">Summary Statistics</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        summary_stats = filtered_df[numeric_columns].describe()
        
        # Format the summary statistics with Rupiah for monetary values
        for col in summary_stats.columns:
            if 'revenue' in col.lower() or 'fee' in col.lower() or 'amount' in col.lower() or 'payment' in col.lower():
                summary_stats[col] = summary_stats[col].apply(format_rupiah)
            elif 'order/adjustment id' in col.lower() or 'id' in col.lower():
                # Format IDs without thousand separators
                summary_stats[col] = summary_stats[col].apply(format_id)
            else:
                summary_stats[col] = summary_stats[col].apply(format_number)
        
        st.dataframe(summary_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Raw data
    st.markdown('<h3 class="subsection-header">Raw Data</h3>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.dataframe(df_display, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced insights
    st.markdown('<h2 class="section-header">💡 Advanced Insights</h2>', unsafe_allow_html=True)
    
    # Revenue insights
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">💰 Revenue Insights</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            total_revenue = filtered_df['Total revenue'].sum()
            avg_revenue = filtered_df['Total revenue'].mean()
            max_revenue = filtered_df['Total revenue'].max()
            min_revenue = filtered_df['Total revenue'].min()
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"📈 Total Revenue: {format_rupiah(total_revenue)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"📊 Average Order Value: {format_rupiah(avg_revenue)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"🏆 Highest Order Value: {format_rupiah(max_revenue)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"📉 Lowest Order Value: {format_rupiah(min_revenue)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Revenue trend analysis
            if len(filtered_df) > 1:
                revenue_trend = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
                revenue_growth = ((revenue_trend.iloc[-1] - revenue_trend.iloc[0]) / revenue_trend.iloc[0]) * 100 if revenue_trend.iloc[0] != 0 else 0
                
                if revenue_growth > 0:
                    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                    st.write(f"🚀 Revenue Growth: +{format_percentage(revenue_growth)} over the period")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif revenue_growth < 0:
                    st.markdown('<div class="insight-warning">', unsafe_allow_html=True)
                    st.write(f"⚠️ Revenue Decline: {format_percentage(revenue_growth)} over the period")
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fee analysis
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">💸 Fee Analysis</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            # Fee breakdown analysis
            if existing_fee_breakdown_cols:
                highest_fee_type = fee_totals.idxmax()
                highest_fee_value = fee_totals.max()
                lowest_fee_type = fee_totals.idxmin()
                lowest_fee_value = fee_totals.min()
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"🔝 Highest Fee Type: {highest_fee_type} ({format_rupiah(highest_fee_value)})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📊 Lowest Fee Type: {lowest_fee_type} ({format_rupiah(lowest_fee_value)})")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Fee efficiency
            total_revenue = filtered_df['Total revenue'].sum()
            if total_revenue != 0:
                fee_efficiency = (abs(total_fees) / total_revenue) * 100
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"💼 Fee Efficiency: {format_percentage(fee_efficiency)} of revenue goes to fees")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if fee_efficiency > 20:
                    st.markdown('<div class="insight-warning">', unsafe_allow_html=True)
                    st.write("⚠️ High fee burden - consider optimizing your pricing strategy")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif fee_efficiency < 10:
                    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                    st.write("✅ Low fee burden - efficient cost management")
                    st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Order analysis
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">📦 Order Analysis</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            # Order volume insights
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"📦 Total Orders: {format_number(len(filtered_df))}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Order type insights
            if len(filtered_df) > 0:
                affiliate_ratio = len(affiliate_orders) / len(filtered_df) * 100
                direct_ratio = len(direct_orders) / len(filtered_df) * 100
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"👥 Affiliate Orders: {format_number(len(affiliate_orders))} ({format_percentage(affiliate_ratio)})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"🏪 Direct Orders: {format_number(len(direct_orders))} ({format_percentage(direct_ratio)})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if affiliate_ratio > 50:
                    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                    st.write("📈 Strong affiliate performance - consider investing more in affiliate marketing")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif direct_ratio > 70:
                    st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                    st.write("🎯 Strong direct sales - brand recognition is high")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Order value insights
            avg_affiliate_value = affiliate_orders['Total revenue'].mean() if not affiliate_orders.empty else 0
            avg_direct_value = direct_orders['Total revenue'].mean() if not direct_orders.empty else 0
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"💰 Avg Affiliate Order Value: {format_rupiah(avg_affiliate_value)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"💵 Avg Direct Order Value: {format_rupiah(avg_direct_value)}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Time-based insights
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">📅 Time-Based Insights</div>', unsafe_allow_html=True)
        if not filtered_df.empty and len(filtered_df) > 1:
            # Best performing day
            daily_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
            best_day = daily_revenue.idxmax()
            best_day_revenue = daily_revenue.max()
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"🌟 Best Sales Day: {best_day} ({format_rupiah(best_day_revenue)})")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Weekly pattern (if enough data)
            if len(daily_revenue) >= 7:
                filtered_df['Weekday'] = filtered_df['Order created time(UTC)'].dt.day_name()
                weekday_revenue = filtered_df.groupby('Weekday')['Total revenue'].sum()
                best_weekday = weekday_revenue.idxmax()
                worst_weekday = weekday_revenue.idxmin()
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📅 Best Sales Day of Week: {best_weekday} ({format_rupiah(weekday_revenue.max())})")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(f"📆 Worst Sales Day of Week: {worst_weekday} ({format_rupiah(weekday_revenue.min())})")
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-header">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)
        
        recommendations = []
        
        # Fee recommendations
        if not filtered_df.empty:
            total_revenue = filtered_df['Total revenue'].sum()
            if total_revenue != 0:
                fee_efficiency = (abs(total_fees) / total_revenue) * 100
                if fee_efficiency > 20:
                    recommendations.append("💰 High fee burden detected. Review your pricing strategy to maintain margins.")
                elif fee_efficiency < 10:
                    recommendations.append("✅ Excellent fee efficiency. Your cost management is performing well.")
            
            # Order type recommendations
            if len(filtered_df) > 0:
                affiliate_ratio = len(affiliate_orders) / len(filtered_df) * 100
                if affiliate_ratio > 50:
                    recommendations.append("📈 Strong affiliate performance. Consider expanding your affiliate marketing program.")
                elif affiliate_ratio < 20:
                    recommendations.append("👥 Low affiliate performance. Explore new affiliate partnerships to expand reach.")
            
            # Revenue trend recommendations
            if len(filtered_df) > 1:
                revenue_trend = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
                revenue_growth = ((revenue_trend.iloc[-1] - revenue_trend.iloc[0]) / revenue_trend.iloc[0]) * 100 if revenue_trend.iloc[0] != 0 else 0
                
                if revenue_growth > 10:
                    recommendations.append("🚀 Strong revenue growth. Consider scaling successful strategies.")
                elif revenue_growth < -5:
                    recommendations.append("📉 Revenue decline detected. Investigate causes and implement corrective measures.")
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write(rec)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write("✅ Your store is performing well. Keep up the good work!")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Analytics section
    st.markdown('<h2 class="ai-header">🤖 AI-Powered Analytics</h2>', unsafe_allow_html=True)
    
    # Predictive analytics
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="ai-subheader">🔮 Revenue Forecast</div>', unsafe_allow_html=True)
        if not filtered_df.empty and len(filtered_df) > 3:
            # Prepare data for forecasting
            daily_revenue = filtered_df.groupby(filtered_df['Order created time(UTC)'].dt.date)['Total revenue'].sum()
            dates = pd.to_datetime(daily_revenue.index)
            revenues = daily_revenue.values
            
            # Create features (days since start)
            days_since_start = [(date - dates.min()).days for date in dates]
            X = np.array(days_since_start).reshape(-1, 1)
            y = revenues
            
            # Train a polynomial regression model
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Predict next 7 days
            last_day = max(days_since_start)
            future_days = np.array([last_day + i for i in range(1, 8)]).reshape(-1, 1)
            future_days_poly = poly_features.transform(future_days)
            future_revenues = model.predict(future_days_poly)
            
            # Create forecast dataframe
            future_dates = [dates.max() + timedelta(days=i) for i in range(1, 8)]
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Revenue': future_revenues
            })
            
            # Display forecast
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write("Based on historical trends, here's a 7-day revenue forecast:")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create forecast chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(dates, revenues, marker='o', linewidth=2, markersize=6, color='#1f77b4', label='Historical Revenue')
            
            # Plot forecast
            ax.plot(future_dates, future_revenues, marker='s', linewidth=2, markersize=6, color='#d62728', linestyle='--', label='Forecasted Revenue')
            
            # Add confidence interval (simplified)
            std_dev = np.std(revenues)
            upper_bound = future_revenues + std_dev
            lower_bound = future_revenues - std_dev
            ax.fill_between(future_dates, lower_bound, upper_bound, color='#d62728', alpha=0.2)
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Revenue (IDR)")
            ax.set_title("Revenue Forecast with Confidence Interval")
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Format y-axis as Rupiah
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'IDR {x:,.0f}'.replace(',', '.')))
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Display forecast table
            forecast_df['Predicted Revenue'] = forecast_df['Predicted Revenue'].apply(format_rupiah)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Forecast insights
            avg_forecast = np.mean(future_revenues)
            total_forecast = np.sum(future_revenues)
            
            st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
            st.write(f"📊 Expected average daily revenue: {format_rupiah(avg_forecast)}")
            st.write(f"💰 Projected revenue for next 7 days: {format_rupiah(total_forecast)}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Insufficient data for forecasting. Need at least 4 days of data for accurate predictions.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation analysis
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="ai-subheader">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            # Select numeric columns for correlation analysis
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter for relevant columns
            relevant_cols = [col for col in numeric_cols if col in [
                'Total revenue', 'Total settlement amount', 'Total fees', 
                'TikTok Shop commission fee', 'Shipping cost', 'Affiliate commission'
            ]]
            
            if len(relevant_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = filtered_df[relevant_cols].corr()
                
                # Create correlation heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                ax.set_title("Correlation Matrix of Key Metrics")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig)
                
                # Key correlation insights
                st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                st.write("Key Correlation Insights:")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:  # Only show strong correlations
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if corr_pairs:
                    for pair in corr_pairs:
                        st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                        if pair['Correlation'] > 0.7:
                            st.write(f"🚀 Strong positive correlation ({format_percentage(abs(pair['Correlation'])*100)}) between {pair['Variable 1']} and {pair['Variable 2']}")
                        elif pair['Correlation'] > 0.5:
                            st.write(f"📈 Moderate positive correlation ({format_percentage(abs(pair['Correlation'])*100)}) between {pair['Variable 1']} and {pair['Variable 2']}")
                        elif pair['Correlation'] < -0.7:
                            st.write(f"📉 Strong negative correlation ({format_percentage(abs(pair['Correlation'])*100)}) between {pair['Variable 1']} and {pair['Variable 2']}")
                        elif pair['Correlation'] < -0.5:
                            st.write(f"🔻 Moderate negative correlation ({format_percentage(abs(pair['Correlation'])*100)}) between {pair['Variable 1']} and {pair['Variable 2']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="insight-item">', unsafe_allow_html=True)
                    st.write("No strong correlations found between key metrics.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Not enough relevant numeric columns for correlation analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Anomaly detection
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="ai-subheader">🔍 Anomaly Detection</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            # Detect anomalies in revenue using IQR method
            revenue_data = filtered_df['Total revenue']
            Q1 = revenue_data.quantile(0.25)
            Q3 = revenue_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify anomalies
            anomalies = filtered_df[(revenue_data < lower_bound) | (revenue_data > upper_bound)]
            
            if not anomalies.empty:
                st.markdown('<div class="insight-warning">', unsafe_allow_html=True)
                st.write(f"⚠️ Detected {len(anomalies)} anomalous orders that deviate significantly from normal revenue patterns:")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display anomalies
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
    
    # Cluster analysis
    with st.container():
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="ai-subheader">蔟 Customer Segmentation</div>', unsafe_allow_html=True)
        if not filtered_df.empty and len(filtered_df) > 5:
            # Create customer segments based on revenue
            filtered_df['Revenue_Segment'] = pd.qcut(
                filtered_df['Total revenue'], 
                q=3, 
                labels=['Low Value', 'Medium Value', 'High Value']
            )
            
            # Calculate segment statistics
            segment_stats = filtered_df.groupby('Revenue_Segment').agg({
                'Total revenue': ['count', 'mean', 'sum']
            }).round(2)
            
            # Flatten column names
            segment_stats.columns = ['Order Count', 'Avg Revenue', 'Total Revenue']
            
            # Format values
            segment_stats['Avg Revenue'] = segment_stats['Avg Revenue'].apply(format_rupiah)
            segment_stats['Total Revenue'] = segment_stats['Total Revenue'].apply(format_rupiah)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write("Customer segmentation based on order value:")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.dataframe(segment_stats, use_container_width=True)
            
            # Create segment visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts = filtered_df['Revenue_Segment'].value_counts()
            bars = ax.bar(segment_counts.index, segment_counts.values, color=['#1f77b4', '#2ca02c', '#d62728'])
            ax.set_xlabel("Customer Segment")
            ax.set_ylabel("Number of Orders")
            ax.set_title("Customer Segmentation by Order Value")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(format_number(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
            
            # Segment insights
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write("Segment Insights:")
            st.markdown('</div>', unsafe_allow_html=True)
            
            high_value_pct = (segment_counts['High Value'] / len(filtered_df)) * 100
            low_value_pct = (segment_counts['Low Value'] / len(filtered_df)) * 100
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"🏆 High-value customers make up {format_percentage(high_value_pct)} of orders")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="insight-item">', unsafe_allow_html=True)
            st.write(f"📊 Low-value customers make up {format_percentage(low_value_pct)} of orders")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if high_value_pct > 20:
                st.markdown('<div class="insight-highlight">', unsafe_allow_html=True)
                st.write("🎯 Strong high-value customer base. Consider VIP programs to retain these customers.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Insufficient data for customer segmentation. Need at least 6 orders for meaningful analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Info when no data is uploaded
    st.info("👆 Please upload your sample_income.xlsx file using the sidebar to begin analysis")
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", 
             caption="Upload your TikTok Shop data to get started", use_container_width=True)

# Footer
st.markdown('<div class="footer">TikTok Shop Analytics Dashboard | Data Insights for Better Business Decisions</div>', unsafe_allow_html=True)
