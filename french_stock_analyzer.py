# French Stock Value Investor Analysis Tool
# -------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import time
import yfinance as yf

# List of major French stocks on Euronext Paris (CAC 40 components)
FRENCH_STOCKS = {
    'Air Liquide': 'AI.PA',
    'TotalEnergies': 'TTE.PA',
    'LVMH': 'MC.PA',
    'Danone': 'BN.PA', 
    'Sanofi': 'SAN.PA',
    'Orange': 'ORA.PA',
    'Michelin': 'ML.PA',
    'L\'Oreal': 'OR.PA',
    'Carrefour': 'CA.PA',
    'AXA': 'CS.PA'
}

class ValueInvestorAnalyzer:
    """A class to analyze French stocks based on value investing principles."""
    
    def __init__(self):
        """Initialize and create directory for data storage."""
        self.data_dir = 'stock_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_stock_data(self, symbol):
        """Fetch historical stock data using Yahoo Finance."""
        try:
            print(f"Fetching data for {symbol}...")
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                print(f"No data returned for {symbol}")
                return None
                
            # Save to CSV
            csv_path = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}.csv")
            df.to_csv(csv_path)
            print(f"Price data saved to {csv_path}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_company_overview(self, symbol):
        """Fetch fundamental company data for value analysis."""
        try:
            print(f"Fetching company info for {symbol}...")
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info or len(info) < 5:
                print(f"No valid info for {symbol}")
                return None
                
            # Save to CSV
            overview_df = pd.DataFrame([info])
            csv_path = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}_overview.csv")
            overview_df.to_csv(csv_path, index=False)
            
            return info
            
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None
    
    def calculate_value_metrics(self, symbol, price_data, info_data):
        """Calculate key value investing metrics for analysis."""
        if price_data is None or info_data is None or price_data.empty:
            return None
            
        # Extract key metrics from info data
        try:
            # Get current price and 52-week high/low
            current_price = price_data['Close'].iloc[-1]
            
            # Extract metrics from info data with appropriate defaults
            metrics = {
                'Symbol': symbol,
                'Name': info_data.get('shortName', 'Unknown'),
                'Sector': info_data.get('sector', 'Unknown'),
                'Industry': info_data.get('industry', 'Unknown'),
                'Market Cap (M€)': info_data.get('marketCap', 0) / 1000000,
                'P/E Ratio': info_data.get('trailingPE', 0),
                'PEG Ratio': info_data.get('pegRatio', 0),
                'Price to Book': info_data.get('priceToBook', 0),
                'EPS': info_data.get('trailingEps', 0),
                'Dividend Yield (%)': info_data.get('dividendYield', 0) * 100 if info_data.get('dividendYield') else 0,
                'Profit Margin': info_data.get('profitMargins', 0),
                'ROE': info_data.get('returnOnEquity', 0),
                'Debt to Equity': info_data.get('debtToEquity', 0) / 100 if info_data.get('debtToEquity') else 0,
                'Current Ratio': info_data.get('currentRatio', 0),
                '52W High': info_data.get('fiftyTwoWeekHigh', 0),
                '52W Low': info_data.get('fiftyTwoWeekLow', 0),
                'Current Price': current_price
            }
            
            # Calculate additional metrics
            if metrics['52W High'] > 0:
                metrics['% From 52W High'] = ((metrics['Current Price'] / metrics['52W High']) - 1) * 100
            else:
                metrics['% From 52W High'] = 0
                
            # Calculate 1-year price change if we have enough data
            if len(price_data) > 200:
                year_ago = price_data['Close'].iloc[-252] if len(price_data) >= 252 else price_data['Close'].iloc[0]
                metrics['1Y Price Change (%)'] = ((current_price / year_ago) - 1) * 100
            else:
                metrics['1Y Price Change (%)'] = 0
                
            # Calculate volatility (annualized standard deviation of returns)
            if len(price_data) > 20:
                daily_returns = price_data['Close'].pct_change().dropna()
                metrics['Volatility (%)'] = daily_returns.std() * np.sqrt(252) * 100
            else:
                metrics['Volatility (%)'] = 0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics for {symbol}: {str(e)}")
            return None
    
    def value_score(self, metrics):
        """Calculate a value score based on common value investing criteria.
        
        Higher score = better value investment candidate
        """
        if metrics is None:
            return 0
            
        score = 0
        
        # P/E Ratio (lower is better for value)
        pe = metrics.get('P/E Ratio', 100)
        if 0 < pe < 10:
            score += 20
        elif 10 <= pe < 15:
            score += 15
        elif 15 <= pe < 20:
            score += 10
        elif 20 <= pe < 25:
            score += 5
            
        # Price to Book (lower is better for value)
        pb = metrics.get('Price to Book', 10)
        if 0 < pb < 1:
            score += 20
        elif 1 <= pb < 2:
            score += 15
        elif 2 <= pb < 3:
            score += 10
        elif 3 <= pb < 4:
            score += 5
            
        # Dividend Yield (higher is better for value)
        div_yield = metrics.get('Dividend Yield (%)', 0)
        if div_yield > 5:
            score += 20
        elif 3 < div_yield <= 5:
            score += 15
        elif 2 < div_yield <= 3:
            score += 10
        elif 1 < div_yield <= 2:
            score += 5
            
        # Debt to Equity (lower is better)
        de = metrics.get('Debt to Equity', 100)
        if 0 < de < 0.5:
            score += 15
        elif 0.5 <= de < 1:
            score += 10
        elif 1 <= de < 1.5:
            score += 5
            
        # Current Ratio (higher is better for financial strength)
        cr = metrics.get('Current Ratio', 0)
        if cr > 2:
            score += 15
        elif 1.5 < cr <= 2:
            score += 10
        elif 1 < cr <= 1.5:
            score += 5
            
        # Distance from 52-week high (larger negative is better for value)
        dist_high = metrics.get('% From 52W High', 0)
        if dist_high < -30:
            score += 10
        elif -30 <= dist_high < -20:
            score += 7
        elif -20 <= dist_high < -10:
            score += 5
            
        # Profit Margin (higher is better)
        margin = metrics.get('Profit Margin', 0)
        if margin > 0.2:
            score += 10
        elif 0.1 < margin <= 0.2:
            score += 7
        elif 0.05 < margin <= 0.1:
            score += 5
            
        # Return on Equity (higher is better)
        roe = metrics.get('ROE', 0)
        if roe > 0.2:
            score += 10
        elif 0.15 < roe <= 0.2:
            score += 7
        elif 0.1 < roe <= 0.15:
            score += 5
        
        return score
    
    def analyze_french_stocks(self, stocks_dict):
        """Analyze multiple French stocks and rank them by value criteria."""
        results = []
        
        for name, symbol in stocks_dict.items():
            print(f"\nAnalyzing {name} ({symbol})...")
            
            # Fetch data
            price_data = self.fetch_stock_data(symbol)
            if price_data is None or price_data.empty:
                print(f"Skipping {symbol} due to missing price data")
                continue
                
            # Small delay to prevent overloading the API
            time.sleep(1)
            
            overview_data = self.fetch_company_overview(symbol)
            if overview_data is None:
                print(f"Skipping {symbol} due to missing overview data")
                continue
                
            # Calculate metrics
            metrics = self.calculate_value_metrics(symbol, price_data, overview_data)
            if metrics is None:
                print(f"Skipping {symbol} due to calculation error")
                continue
                
            # Calculate value score
            value_score = self.value_score(metrics)
            metrics['Value Score'] = value_score
            
            results.append(metrics)
            print(f"Completed analysis for {name} with value score: {value_score}")
            
            # Small delay between stocks
            time.sleep(1)
            
        # Convert to DataFrame and sort by value score
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('Value Score', ascending=False)
            
            # Save results
            results_path = os.path.join(self.data_dir, 'value_analysis_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"\nAnalysis complete. Results saved to {results_path}")
            
            return results_df
        else:
            print("No valid results obtained.")
            return None
    
    def visualize_results(self, results_df):
        """Create visualizations of the analysis results."""
        if results_df is None or results_df.empty:
            print("No data to visualize")
            return
            
        # 1. Value Score Comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(results_df['Name'], results_df['Value Score'], color='royalblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Value Investment Score Comparison - French Stocks')
        plt.xlabel('Company')
        plt.ylabel('Value Score (higher is better)')
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.0f}', ha='center', va='bottom')
        
        plt.savefig(os.path.join(self.data_dir, 'value_score_comparison.png'))
        
        # 2. P/E Ratio vs Dividend Yield scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(results_df['P/E Ratio'], 
                             results_df['Dividend Yield (%)'],
                             s=results_df['Market Cap (M€)'] / 1000,  # Size based on market cap
                             alpha=0.7)
        
        # Add company labels
        for i, name in enumerate(results_df['Name']):
            plt.annotate(name, 
                        (results_df['P/E Ratio'].iloc[i], results_df['Dividend Yield (%)'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('P/E Ratio vs Dividend Yield - Size represents Market Cap')
        plt.xlabel('P/E Ratio (lower is typically better for value)')
        plt.ylabel('Dividend Yield % (higher is typically better for value)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'pe_vs_dividend.png'))
        
        # 3. Fundamental metrics comparison for top 5 stocks
        top5 = results_df.head(5)
        metrics_to_plot = ['P/E Ratio', 'Price to Book', 'Dividend Yield (%)', 
                           'Debt to Equity', 'ROE']
        
        # Normalize metrics for better comparison
        normalized_data = pd.DataFrame()
        for metric in metrics_to_plot:
            # Handle metrics where lower is better
            if metric in ['P/E Ratio', 'Price to Book', 'Debt to Equity']:
                max_val = results_df[metric].max()
                if max_val > 0:
                    normalized_data[metric] = 1 - (top5[metric] / max_val)
                else:
                    normalized_data[metric] = 0
            # Handle metrics where higher is better
            else:
                max_val = results_df[metric].max()
                if max_val > 0:
                    normalized_data[metric] = top5[metric] / max_val
                else:
                    normalized_data[metric] = 0
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        for i, company in enumerate(top5['Name']):
            values = normalized_data.iloc[i].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=company)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], metrics_to_plot)
        
        plt.title('Value Metrics Comparison - Top 5 Stocks')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'radar_comparison.png'))
        
        print("Visualizations saved to the stock_data directory")

def main():
    """Main function to run the French stock analyzer."""
    print("=== French Stock Value Investor Analysis Tool ===")
    print("Using Yahoo Finance API - No API key needed!")
    
    # Create analyzer instance
    analyzer = ValueInvestorAnalyzer()
    
    # Analyze French stocks
    results = analyzer.analyze_french_stocks(FRENCH_STOCKS)
    
    if results is not None:
        # Display top 5 stocks by value score
        print("\n=== Top Value Investment Opportunities ===")
        top_stocks = results.head(5)[['Name', 'Symbol', 'Sector', 'Value Score', 
                                     'P/E Ratio', 'Dividend Yield (%)', 'Price to Book']]
        print(top_stocks)
        
        # Create visualizations
        analyzer.visualize_results(results)
        
        print("\nAnalysis complete! Review the details in the 'stock_data' directory.")
        print("Remember: This is an educational tool, not financial advice!")
    
if __name__ == "__main__":
    main()