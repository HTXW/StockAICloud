import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = 'HDFCBANK.NS'
csv_file_path = 'HDFCBANK_DATA.csv'

stock_data = yf.download(ticker, start='2020-01-01', end='2024-08-31')
stock_data.to_csv(csv_file_path)

stock_data = pd.read_csv(csv_file_path, skiprows=3, header=None)

stock_data.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.sort_values('Date', inplace=True)

stock_data.set_index('Date', inplace=True)
monthly_data = stock_data['Close'].resample('ME').mean()

monthly_data = monthly_data.to_frame(name='Monthly_Avg_Close')
monthly_data['Year'] = monthly_data.index.year
monthly_data['Month'] = monthly_data.index.month

monthly_data = monthly_data[
    (monthly_data['Year'] >= 2020) &
    (monthly_data['Year'] <= 2024)
]

# -------------------------------------------------------------
# 5. Pivot so rows=Month, columns=Year (monthly average close)
# -------------------------------------------------------------
pivot_df = monthly_data.pivot_table(index='Month', columns='Year', values='Monthly_Avg_Close')
pivot_df.sort_index(axis=1, inplace=True)  # Sort years in ascending order

# Print the table
print(pivot_df)

# --------------------
# 6. Plot with mpl
# --------------------
plt.figure(figsize=(9, 6), dpi=600)
pivot_df.plot(ax=plt.gca(), marker='o')
plt.title('HDFC Bank Monthly Average Close Price (Jan 2020 – Aug 2024)', fontsize=14, pad=12)
plt.xlabel('Month', labelpad=8.5, fontsize=13)
plt.ylabel('Average Close Price - Indian Rupees (INR)', labelpad=8.5, fontsize=13)
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.show()