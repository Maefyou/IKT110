import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import json


def get_sold_products(file_path=None):
    transaction_df = pd.DataFrame()
    
    # Read the JSON file using json module
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Iterate through each day
    for day, transactions in data.items():
        # Each 'transactions' is a list of transaction dictionaries
        for transaction in transactions:
            sold = list(zip(transaction.get('merch_types', []), transaction.get('merch_amounts', [])))
            for item, amount in sold:
                new_row = {'item': item, 'amount_sold': amount}
                transaction_df = pd.concat([transaction_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # group by item and sum only numeric columns
    transaction_df = transaction_df.groupby('item', as_index=False).agg({
        'amount_sold': 'sum',
    })

    return transaction_df


def build_product_dataframe():
    # Get all JSON files from the amounts directory
    amount_files = glob("data/amounts/*.json")

    # Read all amount JSON files and combine them into a single DataFrame
    dfs = []
    for file_path in amount_files:
        df = pd.read_json(file_path, orient='index')
        df.columns = ['amount_bought']

        # add supplier prices
        supplier_price_file_path = 'data/supplier_prices/supplier_prices.json'
        supplier_price_df = pd.read_json(supplier_price_file_path, orient='index')
        supplier_price_df.columns = ['supplier_price']
        df = df.merge(supplier_price_df, left_index=True, right_index=True)

        # read price file corresponding to the amount file
        price_file_path = file_path.replace("amounts", "prices")
        price_df = pd.read_json(price_file_path, orient='index')
        price_df.columns = ['sell_price']

        # Merge amount and price DataFrames on index
        df = df.merge(price_df, left_index=True, right_index=True)

        # get sold ammounts and merge, items not in sold should have sold amount 0
        sold_df = get_sold_products(file_path.replace("amounts", "transactions"))
        df = df.merge(sold_df, left_index=True, right_on='item', how='left')
        df['amount_sold'] = df['amount_sold'].fillna(0)
        df = df.set_index(df['item'])
        df = df.drop(columns=['item'])

        # insert week number from filename
        week_number = os.path.basename(file_path).split('_')[1].split('.')[0]
        df.insert(0, 'week_number', int(week_number))

        # return the combined DataFrame to the list
        dfs.append(df)
        print(f"Processed file: {file_path}")

    # Combine all DataFrames
    df = pd.concat(dfs, axis=0)

    print(f'{"-" * 40}\nCombined DataFrame:')
    print(f"Total rows: {len(df)}")
    print("\nFirst few rows:")
    print(df.head(16))
    print("-" * 40)
    return df


def calculate_revenue(df):
    df['revenue'] = df['amount_sold'] * df['sell_price'] - df['supplier_price'] * df['amount_bought']

    print(f'{"-" * 40}\nDataFrame with Revenue:')
    print(f"Total rows: {len(df)}")
    print("\nFirst few rows:")
    print(df.head(5))
    print("-" * 40)

    return df


def plot_items_performance(df, title_suffix=''):
    # Get unique items
    items = df.index.unique()
    n_items = len(items)
    
    # Calculate grid dimensions (rows x cols)
    n_cols = 3  # Number of columns
    n_rows = (n_items + n_cols - 1) // n_cols  # Calculate rows needed
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Create subplots for each item
    for idx, item in enumerate(items):
        item_df = df[df.index == item]
        # plot item bought and sold over weeks
        axes[idx].plot(item_df['week_number'], item_df['amount_bought'], label='Amount Bought', marker='o', markersize=10, linewidth=2)
        axes[idx].plot(item_df['week_number'], item_df['amount_sold'], label='Amount Sold', marker='s', markersize=5, linewidth=2)
        axes[idx].set_title(f'Performance of {item} Over Weeks')
        axes[idx].set_xlabel('Week Number')
        axes[idx].set_ylabel('Amount')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_items, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust spacing with more control
    plt.tight_layout(pad=2.0, h_pad=10.0, w_pad=3.0)
    # Or use subplots_adjust for manual control:
    # plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.savefig('analysis/' + title_suffix + 'items_performance.png')


def predict_demand_polyfit(df):
    predicted_df = df.copy()
    # for each item predict next weeks demand using numpy polyfit
    items = df.index.unique()
    for item in items:
        item_df = df[df.index == item]
        if len(item_df) >= 2:  # Need at least 2 points to fit a line
            # Fit a linear trend line (degree 1 polynomial)
            coefficients = np.polyfit(item_df['week_number'], item_df['amount_sold'], 1)
            polynomial = np.poly1d(coefficients)
            next_week = item_df['week_number'].max() + 1
            predicted_demand = max(polynomial(next_week),0)
            print(f'Predicted demand for {item} in week {next_week}: {predicted_demand:.2f}')

            # Append predicted demand to DataFrame
            new_row = {'week_number': next_week,
                       'amount_bought': None,
                       'supplier_price': item_df['supplier_price'].iloc[0],
                       'sell_price': item_df['sell_price'].iloc[0],
                       'amount_sold': predicted_demand,
                       'revenue': None}
            predicted_df = pd.concat([predicted_df, pd.DataFrame([new_row], index=[item])], ignore_index=False)
        else:
            print(f'Not enough data to predict demand for {item}, assuming same demand.')
            last_amount_sold = item_df['amount_sold'].iloc[-1]
            next_week = item_df['week_number'].max() + 1
            new_row = {'week_number': next_week,
                       'amount_bought': None,
                       'supplier_price': item_df['supplier_price'].iloc[0],
                       'sell_price': item_df['sell_price'].iloc[0],
                       'amount_sold': last_amount_sold,
                       'revenue': None}
            predicted_df = pd.concat([predicted_df, pd.DataFrame([new_row], index=[item])], ignore_index=False)

    return predicted_df


def predict_demand_statsmodels(df):
    predicted_df = df.copy()

    items = df.index.unique()
    for item in items:
        item_df = df[df.index == item]  # Filter for specific item
        
        if len(item_df) >= 2:  # Need at least 2 points to fit a line
            # Use only week_number and amount_sold for the model
            X = sm.add_constant(item_df['week_number'])
            y = item_df['amount_sold']
            
            model = sm.OLS(y, X)
            results = model.fit()
            
            next_week = item_df['week_number'].max() + 1
            # Create prediction input as a 2D array
            X_pred = sm.add_constant([[next_week]], has_constant='add')
            predicted_demand = max(results.predict(X_pred)[0], 0)
            
            print(f'Predicted demand for {item} in week {next_week}: {predicted_demand:.2f}')
            
            new_row = {'week_number': next_week,
                       'amount_bought': None,
                       'supplier_price': item_df['supplier_price'].iloc[0],
                       'sell_price': item_df['sell_price'].iloc[0],
                       'amount_sold': predicted_demand,
                       'revenue': None}
            predicted_df = pd.concat([predicted_df, pd.DataFrame([new_row], index=[item])], ignore_index=False)
        else:
            print(f'Not enough data to predict demand for {item}, assuming same demand.')
            last_amount_sold = item_df['amount_sold'].iloc[-1]
            next_week = item_df['week_number'].max() + 1
            new_row = {'week_number': next_week,
                       'amount_bought': None,
                       'supplier_price': item_df['supplier_price'].iloc[0],
                       'sell_price': item_df['sell_price'].iloc[0],
                       'amount_sold': last_amount_sold,
                       'revenue': None}
            predicted_df = pd.concat([predicted_df, pd.DataFrame([new_row], index=[item])], ignore_index=False)

    return predicted_df


df = build_product_dataframe()
df = calculate_revenue(df)
df.sort_values(by='week_number', ascending=True, inplace=True)
print(df)
plot_items_performance(df, title_suffix='original_')