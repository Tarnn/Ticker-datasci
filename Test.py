import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Define the companies (FAANG + Tesla)
companies = ['META', 'AMZN', 'AAPL', 'NFLX', 'GOOGL', 'TSLA']

# Generate synthetic closing prices for 30 days (April 1, 2025 - April 24, 2025, excluding weekends)
dates = pd.date_range(start='2025-04-01', end='2025-04-24', freq='B')
n_days = len(dates)

# Synthetic closing prices
closing_prices = {
    'META': 300 + np.cumsum(np.random.randn(n_days) * 2),
    'AMZN': 175 + np.cumsum(np.random.randn(n_days) * 1.5),
    'AAPL': 220 + np.cumsum(np.random.randn(n_days) * 1.2),
    'NFLX': 650 + np.cumsum(np.random.randn(n_days) * 3),
    'GOOGL': 150 + np.cumsum(np.random.randn(n_days) * 1),
    'TSLA': 250 + np.cumsum(np.random.randn(n_days) * 2.5)
}

# Convert to DataFrame
price_df = pd.DataFrame(closing_prices, index=dates)

# Synthetic order counts
order_counts = {
    'META': 1200,
    'AMZN': 1500,
    'AAPL': 2000,
    'NFLX': 800,
    'GOOGL': 1300,
    'TSLA': 1700
}

# Synthetic hourly orders (9 AM to 4 PM)
hours = [f'{h} AM' if h < 12 else f'{h-12} PM' if h != 12 else '12 PM' for h in range(9, 16)]
hourly_orders = {
    'META': [50, 60, 70, 80, 90, 100, 110],
    'AMZN': [70, 80, 90, 100, 110, 120, 130],
    'AAPL': [100, 110, 120, 130, 140, 150, 160],
    'NFLX': [30, 40, 50, 60, 70, 80, 90],
    'GOOGL': [60, 70, 80, 90, 100, 110, 120],
    'TSLA': [80, 90, 100, 110, 120, 130, 140]
}

# Synthetic fill ratios
fill_ratios = {
    'META': 0.75,
    'AMZN': 0.60,
    'AAPL': 0.85,
    'NFLX': 0.50,
    'GOOGL': 0.70,
    'TSLA': 0.65
}

# Synthetic execution performance
execution_ratios = {
    'META': 0.98,
    'AMZN': 0.97,
    'AAPL': 0.99,
    'NFLX': 0.96,
    'GOOGL': 0.98,
    'TSLA': 0.97
}

# Introduce a data gap
price_df.loc['2025-04-15', 'TSLA'] = np.nan

# Task 1: Calculate Volatility
volatilities = {}
for company in companies:
    prices = price_df[company].dropna()
    returns = (prices[1:].values - prices[:-1].values) / prices[:-1].values
    mean_return = np.mean(returns)
    variance = np.mean((returns - mean_return) ** 2)
    volatility = np.sqrt(variance)
    volatilities[company] = volatility

print("Volatilities (Standard Deviation of Daily Returns):")
for company, vol in volatilities.items():
    print(f"{company}: {vol:.4f} ({vol*100:.2f}%)")

sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1])
print("\nTop 5 by Lowest Volatility:")
for i, (company, vol) in enumerate(sorted_volatilities[:5], 1):
    print(f"{i}. {company}: {vol*100:.2f}%")

# Task 2: Top 5 Most Ordered Securities
sorted_orders = sorted(order_counts.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 Most Ordered Securities:")
for i, (company, count) in enumerate(sorted_orders[:5], 1):
    print(f"{i}. {company}: {count} orders")

# Task 3: Plot Hourly Orders
plt.figure(figsize=(10, 6))
for company in companies:
    plt.plot(hours, hourly_orders[company], marker='o', label=company)

plt.title("Hourly Orders for Information Technology Sector")
plt.xlabel("Time")
plt.ylabel("Number of Orders")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 4: Top 5 Lowest Fill Ratios
sorted_fill_ratios = sorted(fill_ratios.items(), key=lambda x: x[1])
print("\nTop 5 Securities with Lowest Fill Ratio:")
for i, (company, ratio) in enumerate(sorted_fill_ratios[:5], 1):
    print(f"{i}. {company}: {ratio:.2f}")

# Task 5: Top 5 Best Execution
sorted_execution = sorted(execution_ratios.items(), key=lambda x: x[1])
print("\nTop 5 Securities with Best Execution (Lowest Price Relative to Placement):")
for i, (company, ratio) in enumerate(sorted_execution[:5], 1):
    print(f"{i}. {company}: {ratio:.2f}")

# Task 6: Find Data Gaps
print("\nData Gaps (Missing Prices):")
for company in companies:
    missing_dates = price_df[company][price_df[company].isna()].index
    if not missing_dates.empty:
        print(f"{company} has missing prices on: {', '.join(map(str, missing_dates))}")

print("\nInvalid Prices (Zero or Negative):")
for company in companies:
    invalid_prices = price_df[company][price_df[company] <= 0]
    if not invalid_prices.empty:
        print(f"{company} has invalid prices on: {', '.join(map(str, invalid_prices.index))}")
