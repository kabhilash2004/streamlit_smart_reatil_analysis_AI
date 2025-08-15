# streamlit_smart_reatil_analysis_AI
# 🚀 Smart Retail AI Analyst

An intelligent, multi-page Streamlit web application that provides both standard business intelligence dashboards and dynamic, AI-driven exploratory data analysis for retail sales data.

![Smart Retail AI Analyst](https://i.imgur.com/your-app-image.png) 
*(**Note:** Replace this with a screenshot of your app)*

## Overview

This application serves as a comprehensive tool for retail business owners to gain deep insights from their sales data. Users can upload a CSV or Excel file and choose between three powerful analysis modes:

1.  **Standard Dashboard:** A traditional, multi-tab dashboard offering pre-defined visualizations of key metrics, including sales trends, product performance, geographical analysis, customer behavior, and market basket insights.
2.  **AI-Driven Analysis:** A cutting-edge feature where a powerful Large Language Model (Llama 3 via Groq) acts as an autonomous data scientist. The AI explores the data, generates its own hypotheses, and creates a unique, in-depth report with dynamically generated graphs and business explanations.
3.  **Future Strategy Report:** A forward-looking analysis where the AI acts as a business strategist, using historical data to provide forecasts, identify growth opportunities, and highlight potential risks.

## ✨ Key Features

* **Sidebar Navigation:** A clean, professional user interface with navigation on the left sidebar.
* **Comprehensive Standard Dashboard:** Includes detailed tabs for Sales, Product, Time, Geography, Customer, and Market Basket Analysis.
* **Dynamic AI Data Scientist:** Leverages advanced prompt engineering to make a Large Language Model perform a full exploratory data analysis, generate Python code for visualizations, and execute it in real-time.
* **Future Strategy AI:** A dedicated module where the AI provides predictive insights and strategic recommendations.
* **PDF Report Generation:** Users can download the AI-generated reports, complete with charts and text, as a professional PDF document.
* **Secure API Key Handling:** Uses environment variables for secure management of the Groq API key, ensuring secrets are never exposed in the code.

## 🛠️ Technology Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Data Analysis:** Pandas, Matplotlib, Seaborn
* **Advanced Analytics:** `mlxtend` for Market Basket Analysis
* **AI & LLM:** Groq API (Llama 3 70B Model)
* **PDF Generation:** FPDF
* **Fonts:** DejaVu Sans & DejaVu Sans Bold for Unicode support in PDFs

