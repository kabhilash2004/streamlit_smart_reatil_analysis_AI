import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlxtend.preprocessing import TransactionEncoder  # Added this import
from mlxtend.frequent_patterns import apriori, association_rules

# Set page config
st.set_page_config(page_title="Smart retails Insights", layout="wide")

# Title
st.title("Smart retails Insights Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display original data
    st.subheader("Original Data")
    st.write(f"Original shape: {df.shape}")
    st.dataframe(df.head())
    
    # Data cleaning
    st.subheader("Data Cleaning")
    
    # Remove completely empty columns
    df_clean = df.dropna(axis=1, how='all')
    
    # Remove rows with all NaN values
    df_clean = df_clean.dropna(how='all')
    
    # Fill missing values for specific columns if needed
    for col in ['Itemname', 'Country']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Convert Date to datetime if it exists
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
    
    # Ensure numeric columns are properly formatted
    for col in ['Quantity', 'Price']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(0)
    
    # Add calculated columns
    if all(col in df_clean.columns for col in ['Quantity', 'Price']):
        df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['Price']
    
    if 'Date' in df_clean.columns:
        df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
        df_clean['Hour'] = df_clean['Date'].dt.hour
        df_clean['Month'] = df_clean['Date'].dt.month_name()
    
    st.write(f"Cleaned shape: {df_clean.shape}")
    st.dataframe(df_clean.head())
    
    # Analysis section
    st.subheader("Data Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Sales Overview", 
        "Product Analysis", 
        "Time Analysis", 
        "Geographical Analysis",
        "Customer Analysis",
        "Advanced Insights"
    ])
    
    with tab1:
        st.subheader("Sales Overview")
        if 'TotalSales' in df_clean.columns:
            total_sales = df_clean['TotalSales'].sum()
            avg_order_value = df_clean['TotalSales'].mean()
            total_orders = df_clean['BillNo'].nunique()
            avg_items_per_order = df_clean.groupby('BillNo')['Quantity'].sum().mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Sales", f"${total_sales:,.2f}")
            col2.metric("Average Order Value", f"${avg_order_value:,.2f}")
            col3.metric("Total Orders", f"{total_orders:,}")
            col4.metric("Avg Items per Order", f"{avg_items_per_order:.1f}")
            
            # Sales over time
            if 'Date' in df_clean.columns:
                st.subheader("Sales Over Time")
                sales_over_time = df_clean.groupby(df_clean['Date'].dt.to_period('M'))['TotalSales'].sum().reset_index()
                sales_over_time['Date'] = sales_over_time['Date'].astype(str)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=sales_over_time, x='Date', y='TotalSales', marker='o', ax=ax)
                plt.xticks(rotation=45)
                plt.title("Monthly Sales Trend")
                plt.ylabel("Total Sales ($)")
                st.pyplot(fig)
    
    with tab2:
        st.subheader("Product Analysis")
        if 'Itemname' in df_clean.columns:
            # Most sold products
            top_products = df_clean.groupby('Itemname')['Quantity'].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_products.plot(kind='bar', ax=ax)
            plt.title("Top 10 Best Selling Products")
            plt.ylabel("Quantity Sold")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Highest revenue products
            if 'TotalSales' in df_clean.columns:
                top_revenue_products = df_clean.groupby('Itemname')['TotalSales'].sum().sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_revenue_products.plot(kind='bar', ax=ax)
                plt.title("Top 10 Products by Revenue")
                plt.ylabel("Total Revenue ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Time Analysis")
        if 'DayOfWeek' in df_clean.columns and 'TotalSales' in df_clean.columns:
            # Sales by day of week
            sales_by_day = df_clean.groupby('DayOfWeek')['TotalSales'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sales_by_day.plot(kind='bar', ax=ax)
            plt.title("Sales by Day of Week")
            plt.ylabel("Total Sales ($)")
            st.pyplot(fig)
            
            # Best selling product each day
            best_selling_by_day = df_clean.groupby(['DayOfWeek', 'Itemname'])['Quantity'].sum().reset_index()
            best_selling_by_day = best_selling_by_day.loc[best_selling_by_day.groupby('DayOfWeek')['Quantity'].idxmax()]
            
            st.subheader("Best Selling Product Each Day")
            st.dataframe(best_selling_by_day)
            
            # Sales by hour
            if 'Hour' in df_clean.columns:
                sales_by_hour = df_clean.groupby('Hour')['TotalSales'].sum()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sales_by_hour.plot(kind='bar', ax=ax)
                plt.title("Sales by Hour of Day")
                plt.ylabel("Total Sales ($)")
                st.pyplot(fig)
    
    with tab4:
        st.subheader("Geographical Analysis")
        if 'Country' in df_clean.columns:
            # Sales by country
            sales_by_country = df_clean.groupby('Country')['TotalSales'].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sales_by_country.head(10).plot(kind='bar', ax=ax)
            plt.title("Top 10 Countries by Sales")
            plt.ylabel("Total Sales ($)")
            st.pyplot(fig)
            
            # Map of sales (simple version)
            st.subheader("Sales by Country")
            st.dataframe(sales_by_country.reset_index())
    
    with tab5:
        st.subheader("Customer Analysis")
        if 'CustomerID' in df_clean.columns:
            # Top customers
            top_customers = df_clean.groupby('CustomerID')['TotalSales'].sum().sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            top_customers.plot(kind='bar', ax=ax)
            plt.title("Top 10 Customers by Spending")
            plt.ylabel("Total Spending ($)")
            st.pyplot(fig)
            
            # Customer frequency
            customer_freq = df_clean['CustomerID'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            customer_freq.plot(kind='bar', ax=ax)
            plt.title("Top 10 Frequent Customers")
            plt.ylabel("Number of Orders")
            st.pyplot(fig)
    
    with tab6:
        st.subheader("Advanced Insights")
        
        if all(col in df_clean.columns for col in ['Itemname', 'TotalSales', 'Quantity']):
            # Price elasticity analysis
            st.subheader("Price vs. Quantity Sold")
            price_quantity = df_clean.groupby('Itemname').agg({'Price':'mean', 'Quantity':'sum'}).reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=price_quantity, x='Price', y='Quantity', size='Quantity', sizes=(20, 200), ax=ax)
            plt.title("Price vs. Quantity Sold")
            st.pyplot(fig)
            
            # Market basket analysis
            st.subheader("Frequently Bought Together")
            if all(col in df_clean.columns for col in ['BillNo', 'Itemname']):
                try:
                    # Create a transaction list
                    transaction_list = df_clean.groupby('BillNo')['Itemname'].apply(list).values.tolist()
                    
                    # One-hot encode transactions
                    te = TransactionEncoder()
                    te_ary = te.fit(transaction_list).transform(transaction_list)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    # Find frequent itemsets
                    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
                    
                    if not frequent_itemsets.empty:
                        # Generate association rules
                        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                        rules = rules.sort_values('confidence', ascending=False)
                        
                        st.write("Association Rules (Frequently Bought Together)")
                        st.dataframe(rules.head(10))
                        
                        # Visualize the rules
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=rules.head(20), x='support', y='confidence', size='lift', hue='lift', ax=ax)
                        plt.title("Association Rules (Support vs Confidence)")
                        st.pyplot(fig)
                    else:
                        st.warning("No frequent itemsets found with current settings. Try lowering the min_support parameter.")
                except Exception as e:
                    st.error(f"Market basket analysis failed: {str(e)}")
            else:
                st.warning("Market basket analysis requires BillNo and Itemname columns")
    
    # Download cleaned data
    st.subheader("Download Cleaned Data")
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=df_clean.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_ecommerce_data.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a dataset to get started.")