French Stock Value Investor Analysis Tool
A Python tool for analyzing French stocks based on value investing principles. This tool fetches data from Yahoo Finance, calculates key value metrics, and generates visualizations to help identify potential investment opportunities.
Features

Fetches historical price data and company fundamentals for French stocks
Calculates value investing metrics (P/E ratio, dividend yield, price-to-book, etc.)
Scores stocks based on value criteria
Generates comparison visualizations
Exports analysis results to CSV files

Requirements

Python 3.6+
pandas
matplotlib
numpy
yfinance

Installation
bashCopygit clone https://github.com/username/french-stock-analyzer-tool.git
cd french-stock-analyzer-tool
pip install -r requirements.txt
Usage
Run the analyzer:
bashCopypython french_stock_analyzer.py
The tool will:

Fetch data for predefined CAC 40 stocks
Calculate value metrics for each stock
Generate a value score based on key criteria
Create visualizations in the stock_data directory
Output top investment opportunities based on value criteria

Visualizations
The tool generates three key visualizations:

Value score comparison chart
P/E ratio vs dividend yield scatter plot
Radar chart comparing fundamental metrics of top stocks

Disclaimer
This tool is for educational purposes only and does not constitute financial advice. Always conduct your own research before making investment decisions.
