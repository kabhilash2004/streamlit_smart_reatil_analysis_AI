import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from groq import Groq
import os
import re
import io
import base64
from datetime import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


st.set_page_config(
    page_title="Smart Retail AI Analyst",
    page_icon="🚀",
    layout="wide"
)



COLUMN_ALIASES = {
    'BillNo': ['billno', 'invoice', 'invoiceid', 'bill number'], 'Itemname': ['itemname', 'item', 'product', 'product name', 'description'],
    'Quantity': ['quantity', 'qty', 'units', 'count'], 'Date': ['date', 'invoicedate', 'timestamp'],
    'Price': ['price', 'unitprice', 'cost', 'rate'], 'CustomerID': ['customerid', 'customer id', 'userid'],
    'Country': ['country', 'location'], 'State': ['state', 'province', 'region']
}
REQUIRED_COLUMNS = ['Itemname', 'Quantity', 'Price', 'Date', 'BillNo']

@st.cache_data
def auto_map_columns(df):
    rename_map = {}
    for std_name, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col.lower().replace(" ", "").replace("_", "") in aliases:
                rename_map[col] = std_name
                break
    df_renamed = df.rename(columns=rename_map)
    missing = [col for col in REQUIRED_COLUMNS if col not in df_renamed.columns]
    return df_renamed, missing

@st.cache_data
def clean_data_and_create_features(_df):
    df_clean = _df.copy()
    for col in ['Itemname', 'Country', 'State']:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna('Unknown')
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean.dropna(subset=['Date'], inplace=True)
        df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
        df_clean['Hour'] = df_clean['Date'].dt.hour
        df_clean['Month'] = df_clean['Date'].dt.to_period('M')
    for col in ['Quantity', 'Price']:
        if col in df_clean.columns: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    if all(k in df_clean.columns for k in ['Quantity', 'Price']):
        df_clean['TotalSales'] = df_clean['Quantity'] * df_clean['Price']
    return df_clean

@st.cache_data
def generate_ai_driven_analysis(_df):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY environment variable not set. Please set it before running the app."}

    try:
        client = Groq(api_key=api_key)
        
        buffer = io.StringIO()
        _df.info(buf=buffer)
        data_info = buffer.getvalue()

        system_prompt = """
        You are an expert data analyst and business consultant. Your client, a retail business owner, has provided you with their sales data. Your task is to conduct a complete analysis and present your findings in a professional, text-only report.

        **CRITICAL INSTRUCTIONS:**
        1.  **DO NOT** generate any Python code in your response.
        2.  **DO NOT** mention charts or graphs. Instead, describe the patterns and insights in clear, narrative text.
        3.  Your entire output **MUST** be in Markdown format.

        **REPORT STRUCTURE (Follow this sequence exactly):**

        **Part 1: Data Preview & Initial Assessment**
        Begin with a brief but thorough technical overview of the data. This section must include:
        * **Dataset Shape:** The number of rows and columns.
        * **Data Types:** A summary of each column's data type (Dtype).
        * **Initial Assessment:** A short paragraph detailing your first impressions of the data's quality, completeness, and the potential for analysis.

        ---

        **Part 2: Deep-Dive Analysis**
        This is the core of your analysis. Autonomously explore the data from every perspective it allows (sales trends, product performance, geographical patterns, customer behavior, etc.). For each significant finding, you MUST provide:
        1.  A Markdown heading for the finding (e.g., `### Finding: Afternoon Hours Drive Peak Sales`).
        2.  A detailed, insightful explanation of the finding, describing the trend and what you observed in the data. For example, instead of showing a graph, say "The data clearly shows that sales begin to climb after 11 AM, peaking between 2 PM and 5 PM before tapering off in the evening."

        ---

        **Part 3: Complete Explanation for the Business Owner**
        Conclude with a final section titled `### Your Business Story: A Complete Explanation of the Data`. In this section, you must:
        1.  **Speak in simple, non-technical, and encouraging language.** Avoid all technical jargon.
        2.  **Tell a Story with the Data:** Weave your key findings from Part 2 into a cohesive narrative. Start with the big picture and then drill down into the details.
        3.  **Provide a Complete Explanation:** Explain what is going well and identify areas for improvement.
        4.  **Give Actionable Advice:** Based on your complete explanation, provide a bulleted list of clear, concrete steps the owner can take to improve their business.
        """

        user_prompt = f"""
        Here is the technical information about the dataset:
        {data_info}
        
        And here are the first 5 rows:
        {_df.head().to_string()}

        Please perform your comprehensive analysis now, following the three-part structure precisely. Remember, your entire output must be text only.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3
        )
        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"error": f"An error occurred with the AI API: {e}"}

@st.cache_data
def generate_future_predictions(_df):
    """Uses Groq to generate a forward-looking, strategic report."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY environment variable not set."}

    try:
        client = Groq(api_key=api_key)
        
        
        sales_summary = _df.groupby(_df['Date'].dt.to_period('M'))['TotalSales'].sum().to_string()
        top_products = _df.groupby('Itemname')['TotalSales'].sum().nlargest(5).to_string()
        top_states = _df.groupby('State')['TotalSales'].sum().nlargest(5).to_string() if 'State' in _df.columns else "Not Available"

        system_prompt = """
        You are a world-class business strategist and futurist. Your client, a retail business owner, has provided you with their historical sales data. Your task is to analyze these past trends to provide a forward-looking strategic report.

        **CRITICAL INSTRUCTIONS:**
        1.  **DO NOT** generate any Python code.
        2.  **DO NOT** mention charts or graphs.
        3.  Your entire output **MUST** be a text-only report in Markdown format.
        4.  Base your predictions and advice on the data provided.

        **REPORT STRUCTURE (Follow this sequence exactly):**

        **Part 1: Future Predictions (Forecasting)**
        Based on the historical sales trends, provide a likely forecast for the next 6-12 months.
        - Discuss seasonality: Which months do you predict will be strong or weak?
        - Product Trends: Which products are likely to continue their growth trajectory? Are there any that might decline?

        **Part 2: Growth Opportunities**
        Identify the most promising areas for business growth.
        - **Market Expansion:** Which states or customer segments represent untapped potential?
        - **Product Strategy:** What new products could be introduced that are similar to current best-sellers? Suggest potential product bundles based on sales patterns.
        - **Marketing Initiatives:** Propose 2-3 specific marketing campaigns based on the data (e.g., "A 'Weekend Bonanza' sale to capitalize on high Saturday traffic").

        **Part 3: Potential Risks & Mitigation**
        Identify potential risks the business might face in the future.
        - **Market Risks:** Is the business too reliant on a single product or state? What if a competitor emerges?
        - **Operational Risks:** Are there potential supply chain issues for top-selling products?
        - **Mitigation Strategies:** For each risk, suggest a strategy to reduce its impact.

        **Part 4: Final Strategic Summary**
        Conclude with a high-level summary of your most important recommendations for the business owner.
        """

        user_prompt = f"""
        Here is a summary of my historical sales data:
        - Monthly Sales Trend:
        {sales_summary}
        - Top 5 Products by Sales:
        {top_products}
        - Top 5 States by Sales:
        {top_states}

        Please provide your forward-looking strategic report now.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.5
        )
        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"error": f"An error occurred with the AI API: {e}"}

@st.cache_data
def create_pdf_report(report_title, report_text, _charts=None):
    if not os.path.exists('DejaVuSans.ttf') or not os.path.exists('DejaVuSans-Bold.ttf'):
        return {"error": "Font files not found. Please place them in the project directory."}
    
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.set_font('DejaVu', '', 16)
    pdf.cell(0, 10, report_title, 0, 1, 'C'); pdf.ln(10)
    pdf.set_font('DejaVu', '', 11)
    for line in report_text.split('\n'):
        if line.startswith('### '):
            pdf.set_font('DejaVu', 'B', 14); pdf.multi_cell(0, 10, line.replace('### ', '')); pdf.set_font('DejaVu', '', 11)
        elif line.startswith('**') and line.endswith('**'):
            pdf.set_font('DejaVu', 'B', 11); pdf.multi_cell(0, 10, line.replace('**', '')); pdf.set_font('DejaVu', '', 11)
        else:
            pdf.multi_cell(0, 10, line)
    if _charts:
        for title, fig in _charts.items():
            pdf.add_page(); pdf.set_font('DejaVu', 'B', 14); pdf.cell(0, 10, title, 0, 1)
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight')
                pdf.image(buf, w=180)
    return {"data": pdf.output(dest='S').encode('latin-1')}




with st.sidebar:
    st.image("picg.jpg", width=100)
    st.title("Smart Retail AI Analyst")
    page = st.radio(
        "Choose your analysis method",
        ("Standard Dashboard", "AI-Driven Analysis", "Future Strategy Report"),
        label_visibility="hidden"
    )

st.header(f"📊 {page}")


if page == "Standard Dashboard":
    st.info("Upload your data to see a detailed, multi-tab analysis of your business.")
    uploaded_file = st.file_uploader("Choose a file for the Standard Dashboard", type=["csv", "xlsx"], key="uploader_std")

    if uploaded_file:
        try:
            df_original = pd.read_csv(uploaded_file, encoding='latin1') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            df_renamed, missing = auto_map_columns(df_original)
            if missing:
                st.error(f"Upload failed. Missing required columns: {', '.join(missing)}")
            else:
                df_clean = clean_data_and_create_features(df_renamed)
                st.success("File Processed! View the full analysis below.")
                
                dash_tab1, dash_tab2, dash_tab3, dash_tab4, dash_tab5, dash_tab6 = st.tabs([
                    "Sales Overview", "Product Analysis", "Time Analysis", 
                    "Geographical Analysis", "Customer Analysis", "Advanced Insights"
                ])
                
                with dash_tab1:
                    st.subheader("Sales Overview")
                    total_sales = df_clean['TotalSales'].sum()
                    avg_order_value = df_clean['TotalSales'].mean()
                    total_orders = df_clean['BillNo'].nunique()
                    avg_items_per_order = df_clean.groupby('BillNo')['Quantity'].sum().mean()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Sales", f"₹{total_sales:,.2f}"); col2.metric("Avg Order Value", f"₹{avg_order_value:,.2f}")
                    col3.metric("Total Orders", f"{total_orders:,}"); col4.metric("Avg Items / Order", f"{avg_items_per_order:.1f}")
                    sales_over_time = df_clean.groupby(df_clean['Date'].dt.to_period('M'))['TotalSales'].sum().reset_index()
                    sales_over_time['Date'] = sales_over_time['Date'].astype(str)
                    fig, ax = plt.subplots(figsize=(10, 5)); sns.lineplot(data=sales_over_time, x='Date', y='TotalSales', marker='o', ax=ax)
                    plt.xticks(rotation=45); st.pyplot(fig)

                with dash_tab2:
                    st.subheader("Top 10 Products by Revenue")
                    top_revenue_products = df_clean.groupby('Itemname')['TotalSales'].sum().nlargest(10)
                    fig, ax = plt.subplots(figsize=(10, 6)); top_revenue_products.plot(kind='bar', ax=ax)
                    ax.set_ylabel("Total Revenue (₹)"); plt.xticks(rotation=45); st.pyplot(fig)

                with dash_tab3:
                    st.subheader("Sales by Day of Week")
                    sales_by_day = df_clean.groupby('DayOfWeek')['TotalSales'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    fig, ax = plt.subplots(figsize=(10, 5)); sales_by_day.plot(kind='bar', ax=ax)
                    ax.set_ylabel("Total Sales (₹)"); st.pyplot(fig)

                with dash_tab4:
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Country' in df_clean.columns:
                            st.subheader("Top 10 Countries by Sales")
                            sales_by_country = df_clean.groupby('Country')['TotalSales'].sum().nlargest(10)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sales_by_country.plot(kind='bar', ax=ax, color='skyblue')
                            ax.set_ylabel("Total Sales (₹)")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.warning("No 'Country' column found.")
                    
                    with col2:
                        if 'State' in df_clean.columns:
                            st.subheader("Top 10 States by Sales")
                            sales_by_state = df_clean.groupby('State')['TotalSales'].sum().nlargest(10)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sales_by_state.plot(kind='bar', ax=ax, color='lightgreen')
                            ax.set_ylabel("Total Sales (₹)")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.warning("No 'State' column found.")

                with dash_tab5:
                    if 'CustomerID' in df_clean.columns:
                        st.subheader("Top 10 Customers by Spending")
                        top_customers = df_clean.groupby('CustomerID')['TotalSales'].sum().nlargest(10)
                        fig, ax = plt.subplots(figsize=(10, 5)); top_customers.plot(kind='bar', ax=ax)
                        ax.set_ylabel("Total Spending (₹)"); st.pyplot(fig)
                    else: st.warning("No 'CustomerID' column found.")
                
                with dash_tab6:
                    st.subheader("Market Basket Analysis")
                    if all(c in df_clean.columns for c in ['BillNo', 'Itemname']):
                        with st.spinner("Running analysis..."):
                            try:
                                transaction_list = df_clean.groupby('BillNo')['Itemname'].apply(list).tolist()
                                te = TransactionEncoder(); te_ary = te.fit(transaction_list).transform(transaction_list)
                                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                                frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
                                if not frequent_itemsets.empty:
                                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                                    st.dataframe(rules.sort_values('confidence', ascending=False).head(10))
                                else: st.warning("No frequent itemsets found with current settings.")
                            except Exception as e: st.error(f"Market basket analysis failed: {e}")
                    else: st.warning("Requires 'BillNo' and 'Itemname' columns.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


elif page == "AI-Driven Analysis":
    st.info("Let our AI Data Analyst perform a comprehensive, exploratory analysis of your data and generate a unique report.")
    uploaded_file_ai = st.file_uploader("Choose a file for the AI Analyst", type=["csv", "xlsx"], key="uploader_ai")
    if uploaded_file_ai:
        try:
            df_original = pd.read_csv(uploaded_file_ai, encoding='latin1') if uploaded_file_ai.name.endswith('.csv') else pd.read_excel(uploaded_file_ai)
            df_renamed, missing = auto_map_columns(df_original)
            if missing:
                st.error(f"Upload failed. Missing required columns: {', '.join(missing)}")
            else:
                df_processed = clean_data_and_create_features(df_renamed)
                if st.button("✨ Let AI Analyze My Data", type="primary", key="ai_button"):
                    with st.spinner("Your AI Data Scientist is analyzing the data..."):
                        ai_result = generate_ai_driven_analysis(df_processed)
                        st.session_state.ai_result = ai_result # Store the entire dictionary
                        st.session_state.df_for_exec = df_processed
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if 'ai_result' in st.session_state and 'df_for_exec' in st.session_state:
        st.markdown("---")
        st.subheader("AI Data Scientist Report")
        
        ai_result = st.session_state.ai_result
        if "error" in ai_result:
            st.error(ai_result["error"])
        else:
          
            
            report_text = ai_result["response"]
            
            charts_for_pdf = {}
            
            last_index = 0
            code_matches = list(re.finditer(r"```python\n(.*?)```", report_text, re.DOTALL))
            for i, match in enumerate(code_matches):
                start, end = match.span()
                st.markdown(report_text[last_index:start], unsafe_allow_html=True)
                last_index = end
                code_block = match.group(1)
                local_scope = {'df': st.session_state.df_for_exec, 'plt': plt, 'sns': sns}
                try:
                    exec(code_block, local_scope)
                    fig = local_scope.get('fig')
                    if fig:
                        st.pyplot(fig)
                        title_search = re.search(r"ax\.set_title\(['\"](.*?)['\"]\)", code_block)
                        chart_title = title_search.group(1) if title_search else f"Chart_{i+1}"
                        charts_for_pdf[chart_title] = fig
                except Exception as e:
                    st.error(f"Could not execute AI-generated code: {e}")
            st.markdown(report_text[last_index:], unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Download Report")
            
            pdf_result = create_pdf_report("AI-Driven Analysis Report", report_text, charts_for_pdf)
            if "error" in pdf_result:
                st.error(pdf_result["error"])
            else:
                st.download_button(
                    label="📥 Download as PDF",
                    data=pdf_result["data"],
                    file_name=f"AI_Data_Scientist_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )


elif page == "Future Strategy Report":
    st.info("Upload your historical data to let our AI Business Strategist provide a forward-looking report with predictions and growth opportunities.")
    uploaded_file_future = st.file_uploader("Choose a file for the Future Strategy Report", type=["csv", "xlsx"], key="uploader_future")

    if uploaded_file_future:
        try:
            df_original = pd.read_csv(uploaded_file_future, encoding='latin1') if uploaded_file_future.name.endswith('.csv') else pd.read_excel(uploaded_file_future)
            df_renamed, missing = auto_map_columns(df_original)
            if missing:
                st.error(f"Upload failed. Missing required columns: {', '.join(missing)}")
            else:
                df_processed = clean_data_and_create_features(df_renamed)
                if st.button("🔮 Generate Future Strategy", type="primary", key="future_button"):
                    with st.spinner("Your AI Strategist is analyzing future trends..."):
                        future_result = generate_future_predictions(df_processed)
                        if "error" in future_result:
                            st.error(future_result["error"])
                        else:
                            st.session_state.future_report_text = future_result["response"]
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if 'future_report_text' in st.session_state:
        st.markdown("---")
        st.subheader("AI Business Strategy Report")
        
        report_text = st.session_state.future_report_text
        st.markdown(report_text, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Download Strategy Report")
        pdf_result = create_pdf_report("Future Strategy Report", report_text) # No charts are passed
        if "error" in pdf_result:
            st.error(pdf_result["error"])
        else:
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_result["data"],
                file_name=f"AI_Future_Strategy_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )