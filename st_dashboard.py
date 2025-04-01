#!/bin/bash

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import io
import random

# Set page config
st.set_page_config(
    page_title="Financial Data Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to create sample data
def create_sample_data():
    # Create a sample dataframe with the same structure as the expected data
    months = list(range(1, 13))
    years = [2024, 2025]
    codes = ["Revenue", "Cogs", "OPEX"]
    
    data = []
    
    for year in years:
        for month in months:
            # Revenue: Random between 70-100
            revenue = random.uniform(70, 100)
            # COGS: 30-50% of revenue
            cogs = revenue * random.uniform(0.3, 0.5)
            # OPEX: 10-30% of revenue
            opex = revenue * random.uniform(0.1, 0.3)
            
            data.append({"month": month, "year": year, "code": "Revenue", "amount": revenue})
            data.append({"month": month, "year": year, "code": "Cogs", "amount": cogs})
            data.append({"month": month, "year": year, "code": "OPEX", "amount": opex})
    
    return pd.DataFrame(data)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 5.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a data file to continue.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def format_number(number, prefix="$"):
    """Format numbers with K for thousands, M for millions, etc"""
    if number >= 1_000_000_000:
        return f"{prefix}{number/1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        return f"{prefix}{number/1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{prefix}{number/1_000:.2f}K"
    else:
        return f"{prefix}{number:.2f}"

def main():
    # Header
    st.markdown('<p class="main-header">Financial Performance Dashboard</p>', unsafe_allow_html=True)
    
    # File uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file", 
        type=["xlsx", "xls"],
        help="Upload an Excel file with columns for month, year, code, and amount"
    )
    
    st.sidebar.markdown("""
    ### Expected Data Format:
    - **month**: Month number (1-12)
    - **year**: Year (e.g., 2024)
    - **code**: Category (e.g., Revenue, Cogs, OPEX)
    - **amount**: Numeric values
    """)
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use sample data instead", value=False)
    
    if use_sample_data:
        # Create sample data
        sample_data = create_sample_data()
        df = sample_data
    else:
        # Load data from uploaded file
        df = load_data(uploaded_file)
    
    if df is None:
        st.info("👆 Please upload your Excel file using the sidebar to get started.")
        
        # Show template download option
        st.markdown("### Data Template")
        st.markdown("""
        You can download a template file to see the expected format:
        """)
        
        template_df = create_sample_data()
        
        # Convert to Excel for download
        buffer = io.BytesIO()
        template_df.to_excel(buffer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="Download Template Excel File",
            data=buffer,
            file_name="financial_data_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        return
    
    # Add month name column
    df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])
    
    # Convert amount to numeric if it's not already
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Year filter
    available_years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years", 
        options=available_years,
        default=available_years
    )
    
    # Category filter
    categories = sorted(df['code'].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Categories/ Codes",
        options=categories,
        default=categories
    )
    
    # Month range filter
    months = list(range(1, 13))
    month_range = st.sidebar.slider(
        "Month Range",
        min_value=min(months),
        max_value=max(months),
        value=(min(months), max(months))
    )
    
    # Apply filters
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['code'].isin(selected_categories)) &
        (df['month'] >= month_range[0]) &
        (df['month'] <= month_range[1])
    ]
    
    # Check if we have data after filtering
    if filtered_df.empty:
        st.warning("No data available with the selected filters.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Comparison", "Data", "DOC"])
    
    with tab1:
        # KPI metrics in the first row
        col1, col2, col3 = st.columns(3)
        
        # Total Revenue
        revenue_df = filtered_df[filtered_df['code'] == 'Revenue']
        total_revenue = revenue_df['amount'].sum()
        
        # Total COGS
        cogs_df = filtered_df[filtered_df['code'] == 'Cogs']
        total_cogs = cogs_df['amount'].sum()
        
        # Total OPEX
        opex_df = filtered_df[filtered_df['code'] == 'OPEX']
        total_opex = opex_df['amount'].sum()
        
        # Gross Profit
        gross_profit = total_revenue - total_cogs
        
        # Net Profit
        net_profit = gross_profit - total_opex
        
        # Gross Margin %
        gross_margin_pct = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Net Margin %
        net_margin_pct = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        with col1:
            st.markdown('<p class="sub-header">Revenue</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>{format_number(total_revenue)}</h1></div>', unsafe_allow_html=True)
            
            st.markdown('<p class="sub-header">Gross Profit</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>{format_number(gross_profit)}</h1></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p class="sub-header">COGS</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>{format_number(total_cogs)}</h1></div>', unsafe_allow_html=True)
            
            st.markdown('<p class="sub-header">Net Profit</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>Net Profit: {format_number(net_profit)}</h1>%<br>Net: {net_margin_pct:.2f}%</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<p class="sub-header">OPEX</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>{format_number(total_opex)}</h1></div>', unsafe_allow_html=True)
            
            st.markdown('<p class="sub-header">Margins</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h1>Gross: {gross_margin_pct:.2f}</h1></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stacked area chart by category
        pivot_df = filtered_df.pivot_table(
            index=['year', 'month', 'month_name'],
            columns='code',
            values='amount',
            aggfunc='sum'
        ).reset_index()
        
        # Sort by year and month
        pivot_df['month_year'] = pivot_df['year'].astype(str) + '-' + pivot_df['month'].astype(str).str.zfill(2)
        pivot_df = pivot_df.sort_values(['year', 'month'])
        
        # Create labels for x-axis
        pivot_df['label'] = pivot_df['month_name'].str[:3] + ' ' + pivot_df['year'].astype(str)
        
        fig = go.Figure()
        
        if 'Revenue' in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['label'],
                y=pivot_df['Revenue'],
                name='Revenue',
                mode='lines',
                line=dict(width=3, color='#4CAF50'),
                fill='none'
            ))
        
        if 'Cogs' in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['label'],
                y=pivot_df['Cogs'],
                name='COGS',
                mode='lines',
                line=dict(width=3, color='#F44336'),
                fill='none'
            ))
        
        if 'OPEX' in pivot_df.columns:
            fig.add_trace(go.Scatter(
                x=pivot_df['label'],
                y=pivot_df['OPEX'],
                name='OPEX',
                mode='lines',
                line=dict(width=3, color='#2196F3'),
                fill='none'
            ))
        
        # Calculate profit if we have revenue and cogs
        if 'Revenue' in pivot_df.columns and 'Cogs' in pivot_df.columns:
            pivot_df['Gross Profit'] = pivot_df['Revenue'] - pivot_df['Cogs']
            
            fig.add_trace(go.Scatter(
                x=pivot_df['label'],
                y=pivot_df['Gross Profit'],
                name='Gross Profit',
                mode='lines',
                line=dict(width=3, color='#9C27B0'),
                fill='none'
            ))
        
        fig.update_layout(
            title='Financial Performance Over Time',
            xaxis_title='Month',
            yaxis_title='Amount',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart for category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            category_totals = filtered_df.groupby('code')['amount'].sum().reset_index()
            fig_pie = px.pie(
                category_totals, 
                values='amount', 
                names='code',
                title='Distribution by Category',
                color='code',
                color_discrete_map={
                    'Revenue': '#4CAF50',
                    'Cogs': '#F44336',
                    'OPEX': '#2196F3'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Year comparison
            if len(selected_years) > 1:
                year_comparison = filtered_df.groupby(['year', 'code'])['amount'].sum().reset_index()
                fig_bar = px.bar(
                    year_comparison,
                    x='code',
                    y='amount',
                    color='year',
                    barmode='group',
                    title='Yearly Comparison by Category'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Monthly distribution for a single year
                monthly_data = filtered_df.groupby(['month_name', 'code'])['amount'].sum().reset_index()
                # Ensure proper month order
                month_order = [calendar.month_name[i] for i in range(1, 13)]
                monthly_data['month_name'] = pd.Categorical(monthly_data['month_name'], categories=month_order, ordered=True)
                monthly_data = monthly_data.sort_values('month_name')
                
                fig_monthly = px.bar(
                    monthly_data,
                    x='month_name',
                    y='amount',
                    color='code',
                    title=f'Monthly Distribution for {selected_years[0]}',
                    color_discrete_map={
                        'Revenue': '#4CAF50',
                        'Cogs': '#F44336',
                        'OPEX': '#2196F3'
                    }
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab2:
        st.markdown('<p class="sub-header">Monthly Trends</p>', unsafe_allow_html=True)
        
        # Line charts for trends
        trend_df = filtered_df.pivot_table(
            index=['year', 'month', 'month_name'],
            columns='code',
            values='amount',
            aggfunc='sum'
        ).reset_index()
        
        # Sort by year and month
        trend_df['month_year'] = trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str).str.zfill(2)
        trend_df = trend_df.sort_values(['year', 'month'])
        
        # Create labels for x-axis
        trend_df['label'] = trend_df['month_name'].str[:3] + ' ' + trend_df['year'].astype(str)
        
        # Create subplots
        fig = make_subplots(
            rows=len(categories), 
            cols=1,
            subplot_titles=[f"{cat} Trend" for cat in categories if cat in trend_df.columns],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        colors = {'Revenue': '#4CAF50', 'Cogs': '#F44336', 'OPEX': '#2196F3'}
        
        row = 1
        for cat in categories:
            if cat in trend_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df['label'],
                        y=trend_df[cat],
                        mode='lines+markers',
                        name=cat,
                        line=dict(color=colors.get(cat, '#000000'), width=2),
                        marker=dict(size=8)
                    ),
                    row=row, col=1
                )
                row += 1
        
        fig.update_layout(
            height=300 * len(categories),
            showlegend=False,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-Year comparison
        if len(selected_years) > 1:
            st.markdown('<p class="sub-header">Year-over-Year Comparison</p>', unsafe_allow_html=True)
            
            yoy_df = filtered_df.pivot_table(
                index=['month', 'month_name'],
                columns=['year', 'code'],
                values='amount',
                aggfunc='sum'
            ).reset_index()
            
            # Sort by month
            yoy_df = yoy_df.sort_values('month')
            
            # Create YoY comparison charts
            for cat in categories:
                cols = [col for col in yoy_df.columns if cat in str(col)]
                if len(cols) > 0:
                    fig = go.Figure()
                    
                    for year in selected_years:
                        col_name = (year, cat)
                        if col_name in yoy_df.columns:
                            fig.add_trace(go.Bar(
                                x=yoy_df['month_name'],
                                y=yoy_df[col_name],
                                name=f"{year} {cat}"
                            ))
                    
                    fig.update_layout(
                        title=f"{cat} - Year-over-Year Comparison",
                        xaxis_title="Month",
                        yaxis_title="Amount",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<p class="sub-header">Category Comparison</p>', unsafe_allow_html=True)
        
        # Create a stacked bar chart comparing all categories by month
        monthly_stacked = filtered_df.pivot_table(
            index=['month_name', 'month'],
            columns='code',
            values='amount',
            aggfunc='sum'
        ).reset_index()
        
        # Sort by month
        month_order = {calendar.month_name[i]: i for i in range(1, 13)}
        monthly_stacked['month_idx'] = monthly_stacked['month_name'].map(month_order)
        monthly_stacked = monthly_stacked.sort_values('month_idx')
        
        # Create the stacked bar chart
        fig = go.Figure()
        
        for cat in categories:
            if cat in monthly_stacked.columns:
                fig.add_trace(go.Bar(
                    x=monthly_stacked['month_name'],
                    y=monthly_stacked[cat],
                    name=cat,
                    marker_color=colors.get(cat, '#000000')
                ))
        
        fig.update_layout(
            title='Monthly Category Comparison',
            xaxis_title='Month',
            yaxis_title='Amount',
            barmode='group',
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add ratio analysis
        if 'Revenue' in filtered_df['code'].values and 'Cogs' in filtered_df['code'].values:
            st.markdown('<p class="sub-header">Profitability Analysis</p>', unsafe_allow_html=True)
            
            # Calculate monthly ratios
            ratio_df = filtered_df.pivot_table(
                index=['year', 'month', 'month_name'],
                columns='code',
                values='amount',
                aggfunc='sum'
            ).reset_index()
            
            if 'Revenue' in ratio_df.columns and 'Cogs' in ratio_df.columns:
                ratio_df['Gross Profit'] = ratio_df['Revenue'] - ratio_df['Cogs']
                ratio_df['Gross Margin %'] = (ratio_df['Gross Profit'] / ratio_df['Revenue']) * 100
                
                if 'OPEX' in ratio_df.columns:
                    ratio_df['Net Profit'] = ratio_df['Gross Profit'] - ratio_df['OPEX']
                    ratio_df['Net Margin %'] = (ratio_df['Net Profit'] / ratio_df['Revenue']) * 100
                    ratio_df['OPEX to Revenue %'] = (ratio_df['OPEX'] / ratio_df['Revenue']) * 100
                
                # Sort by year and month
                ratio_df['month_year'] = ratio_df['year'].astype(str) + '-' + ratio_df['month'].astype(str).str.zfill(2)
                ratio_df = ratio_df.sort_values(['year', 'month'])
                
                # Create labels for x-axis
                ratio_df['label'] = ratio_df['month_name'].str[:3] + ' ' + ratio_df['year'].astype(str)
                
                # Create margin trend chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=ratio_df['label'],
                    y=ratio_df['Gross Margin %'],
                    mode='lines+markers',
                    name='Gross Margin %',
                    line=dict(width=2, color='#4CAF50')
                ))
                
                if 'Net Margin %' in ratio_df.columns:
                    fig.add_trace(go.Scatter(
                        x=ratio_df['label'],
                        y=ratio_df['Net Margin %'],
                        mode='lines+markers',
                        name='Net Margin %',
                        line=dict(width=2, color='#9C27B0')
                    ))
                
                if 'OPEX to Revenue %' in ratio_df.columns:
                    fig.add_trace(go.Scatter(
                        x=ratio_df['label'],
                        y=ratio_df['OPEX to Revenue %'],
                        mode='lines+markers',
                        name='OPEX to Revenue %',
                        line=dict(width=2, color='#2196F3')
                    ))
                
                fig.update_layout(
                    title='Margin Trends',
                    xaxis_title='Month',
                    yaxis_title='Percentage (%)',
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # COGS to Revenue ratio
        if 'Revenue' in filtered_df['code'].values and 'Cogs' in filtered_df['code'].values:
            ratio_by_month = filtered_df.pivot_table(
                index=['month_name', 'month'],
                columns='code',
                values='amount',
                aggfunc='sum'
            ).reset_index()
            
            # Sort by month
            ratio_by_month['month_idx'] = ratio_by_month['month_name'].map(month_order)
            ratio_by_month = ratio_by_month.sort_values('month_idx')
            
            ratio_by_month['COGS to Revenue %'] = (ratio_by_month['Cogs'] / ratio_by_month['Revenue']) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=ratio_by_month['month_name'],
                y=ratio_by_month['COGS to Revenue %'],
                name='COGS to Revenue %',
                marker_color='#F44336'
            ))
            
            fig.update_layout(
                title='COGS to Revenue Ratio by Month',
                xaxis_title='Month',
                yaxis_title='Percentage (%)',
                height=400,
                hovermode="x"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<p class="sub-header">Raw Data</p>', unsafe_allow_html=True)
        
        # Add a search filter
        search = st.text_input("Search in data")
        
        if search:
            search_results = filtered_df[
                filtered_df.astype(str).apply(
                    lambda row: row.str.contains(search, case=False).any(), 
                    axis=1
                )
            ]
            st.dataframe(search_results, use_container_width=True)
        else:
            st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="filtered_financial_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()