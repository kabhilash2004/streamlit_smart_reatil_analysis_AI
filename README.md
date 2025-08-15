# streamlit_smart_reatil_analysis_AI
🚀 Smart Retail AI Analyst
An intelligent, multi-page Streamlit web application that provides both standard business intelligence dashboards and dynamic, AI-driven exploratory data analysis for retail sales data.

Overview
This application serves as a comprehensive tool for retail business owners to gain deep insights from their sales data. Users can upload a CSV or Excel file and choose between two powerful analysis modes:

Standard Dashboard: A traditional, multi-tab dashboard offering pre-defined visualizations of key metrics, including sales trends, product performance, geographical analysis, and market basket insights.

AI-Driven Analysis: A cutting-edge feature where a powerful Large Language Model (Llama 3 via Groq) acts as an autonomous data scientist. The AI explores the data, generates its own hypotheses, and creates a unique, in-depth report with dynamically generated graphs and business explanations.

The application also includes a forward-looking Future Strategy Report, where the AI analyzes historical data to provide predictions and strategic recommendations.

✨ Key Features
Sidebar Navigation: A clean, professional user interface with navigation on the left sidebar.

Comprehensive Standard Dashboard: Includes detailed tabs for Sales, Product, Time, Geography, Customer, and Market Basket Analysis.

Dynamic AI Data Scientist: Leverages advanced prompt engineering to make a Large Language Model perform a full exploratory data analysis, generate Python code for visualizations, and execute it in real-time.

Future Strategy AI: A dedicated module where the AI acts as a business consultant, providing forecasts, growth opportunities, and risk assessments.

PDF Report Generation: Users can download the AI-generated reports, complete with charts and text, as a professional PDF document.

Secure API Key Handling: Uses environment variables for secure management of the Groq API key.

🛠️ Technology Stack
Language: Python

Web Framework: Streamlit

Data Analysis: Pandas, Matplotlib, Seaborn

Advanced Analytics: mlxtend for Market Basket Analysis

AI & LLM: Groq API (Llama 3 70B Model)

PDF Generation: FPDF

Fonts: DejaVu Sans & DejaVu Sans Bold for Unicode support in PDFs

⚙️ Setup and Usage
To run this project locally, follow these steps:

1. Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install the required libraries:

Bash

pip install -r requirements.txt
(Note: Make sure to create a requirements.txt file with all the necessary packages like streamlit, pandas, groq, fpdf, etc.)

3. Set your Groq API Key:

On Windows (Command Prompt):

Bash

set GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
On macOS/Linux:

Bash

export GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
4. Run the Streamlit application:

Bash

streamlit run app.py
