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
    df['revenue'] = df['amount_sold'] * df['sell_price']

    print(f'{"-" * 40}\nDataFrame with Revenue:')
    print(f"Total rows: {len(df)}")
    print("\nFirst few rows:")
    print(df.head(5))
    print("-" * 40)

    return df


def calculate_profit(df):
    df['profit'] = df['amount_sold'] * df['sell_price'] - df['supplier_price'] * df['amount_bought']

    print(f'{"-" * 40}\nDataFrame with profit:')
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


def plot_week_profit_by_product(df, week_number=5):
    # Filter for the specified week
    week_df = df[df['week_number'] == week_number].copy()
    
    if week_df.empty:
        print(f"No data available for week {week_number}")
        return
    
    # Get products and their revenue (profit)
    products = week_df.index
    profits = week_df['profit']
    
    # Create colors based on profit (green for positive, red for negative)
    colors = ['green' if profit > 0 else 'red' for profit in profits]
    
    # Create the bar chart
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    bars = plt.bar(products, profits, color=colors)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.title(f'Profit by Product - Week {week_number}')
    plt.xlabel('Product')
    plt.ylabel('Profit')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.savefig(f'analysis/week_{week_number}_profit_by_product.png')
    plt.close()


def plot_week_utilization_by_product(df, week_number=5):
    # Filter for the specified week
    week_df = df[df['week_number'] == week_number].copy()
    
    if week_df.empty:
        print(f"No data available for week {week_number}")
        return
    
    week_df.sort_values(by='item', inplace=True)

    # Calculate utilization percentage (amount_sold / amount_bought * 100)
    week_df['utilization'] = (week_df['amount_sold'] / week_df['amount_bought']) * 100
    
    # Get products and their utilization
    products = week_df.index
    utilization = week_df['utilization']
    
    # Create colors based on utilization (green for high, yellow for medium, red for low)
    colors = ['green' if u >= 80 else 'yellow' if u >= 50 else 'red' for u in utilization]
    
    # Create the bar chart
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    bars = plt.bar(products, utilization, color=colors)
    
    # Add a horizontal line at 100%
    plt.axhline(y=100, color='blue', linestyle='--', linewidth=0.8, label='100% Utilization')
    
    plt.title(f'Item Utilization - Week {week_number}')
    plt.xlabel('Product')
    plt.ylabel('Utilization (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.savefig(f'assets/week_{week_number}_utilization_by_product.png')
    plt.close()


def plot_overall_utilization_by_product(df):
    # Group by product and sum amounts across all weeks
    overall_df = df.groupby(df.index).agg({
        'amount_bought': 'sum',
        'amount_sold': 'sum'
    })

    overall_df.sort_values(by='item', inplace=True)
    
    # Calculate overall utilization percentage
    overall_df['utilization'] = (overall_df['amount_sold'] / overall_df['amount_bought']) * 100
    
    # Get products and their utilization
    products = overall_df.index
    utilization = overall_df['utilization']
    
    # Create colors based on utilization (green for high, yellow for medium, red for low)
    colors = ['green' if u >= 80 else 'yellow' if u >= 50 else 'red' for u in utilization]
    
    # Create the bar chart
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    bars = plt.bar(products, utilization, color=colors)
    
    # Add a horizontal line at 100%
    plt.axhline(y=100, color='blue', linestyle='--', linewidth=0.8, label='100% Utilization')
    
    plt.title('Overall Item Utilization - All Weeks')
    plt.xlabel('Product')
    plt.ylabel('Utilization (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.savefig('assets/overall_utilization_by_product.png')
    plt.close()


def build_all_transaction_details():
    """
    Build a comprehensive DataFrame containing all transaction details.
    
    Returns:
        DataFrame with columns: week, day, item, amount_sold, customer, worker, 
                                transaction_type, supplier_price, sell_price
    """
    transaction_rows = []
    
    # Load supplier prices (global across all weeks)
    supplier_price_file_path = 'data/supplier_prices/supplier_prices.json'
    if os.path.exists(supplier_price_file_path):
        supplier_price_df = pd.read_json(supplier_price_file_path, orient='index')
        supplier_price_df.columns = ['supplier_price']
    else:
        supplier_price_df = pd.DataFrame(columns=['supplier_price'])
    
    # Get all transaction files
    transaction_files = glob("data/transactions/*.json")
    
    for tfile in transaction_files:
        # Extract week number from filename
        try:
            week_number = int(os.path.basename(tfile).split('_')[1].split('.')[0])
        except Exception:
            week_number = None
        
        # Load corresponding price file for this week
        price_file_path = tfile.replace("transactions", "prices")
        if os.path.exists(price_file_path):
            price_df = pd.read_json(price_file_path, orient='index')
            price_df.columns = ['sell_price']
        else:
            price_df = pd.DataFrame(columns=['sell_price'])
        
        # Load transaction data
        with open(tfile, 'r') as f:
            data = json.load(f)
        
        # Process each day's transactions
        for day, transactions in data.items():
            for transaction in transactions:
                # Get transaction metadata
                customer_id = transaction.get('customer_id', None)
                worker_id = transaction.get('register_worker', None)
                transaction_type = transaction.get('transaction_type', None)
                
                # Process each item in the transaction
                merch_types = transaction.get('merch_types', [])
                merch_amounts = transaction.get('merch_amounts', [])
                
                for item, amount in zip(merch_types, merch_amounts):
                    # Get prices for this item
                    sell_price = price_df.loc[item, 'sell_price'] if item in price_df.index else None
                    supplier_price = supplier_price_df.loc[item, 'supplier_price'] if item in supplier_price_df.index else None
                    
                    # Create row for this item in this transaction
                    transaction_rows.append({
                        'week': week_number,
                        'day': int(day),
                        'item': item,
                        'amount_sold': amount,
                        'customer': customer_id,
                        'worker': worker_id,
                        'transaction_type': transaction_type,
                        'supplier_price': supplier_price,
                        'sell_price': sell_price
                    })
    
    # Build DataFrame
    df = pd.DataFrame(transaction_rows)
    
    # Sort by week, day, and item for better organization
    df.sort_values(by=['week', 'day', 'item'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f'{"-" * 40}\nAll Transaction Details DataFrame:')
    print(f"Total rows: {len(df)}")
    print(f"Weeks covered: {df['week'].unique()}")
    print("\nFirst few rows:")
    print(df.head(10))
    print("\nColumn types:")
    print(df.dtypes)
    print("-" * 40)
    
    return df


def plot_daily_sales_and_revenue(trans_df, week_number=5):
    """
    Create a bar chart showing daily sales amount and revenue for a specific week.
    For each day, displays two bars: total items sold and total revenue.
    Uses the DataFrame from build_all_transaction_details().
    """
    # Filter for the specified week
    week_df = trans_df[trans_df['week'] == week_number].copy()
    
    if week_df.empty:
        print(f"No transaction data available for week {week_number}")
        return
    
    # Calculate revenue for each transaction
    week_df['revenue'] = week_df['amount_sold'] * week_df['sell_price']
    
    # Group by day and sum
    daily_stats = week_df.groupby('day').agg({
        'amount_sold': 'sum',
        'revenue': 'sum'
    }).reset_index()

    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(daily_stats['day']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, daily_stats['amount_sold'], width, label='Items Sold', color='skyblue')
    bars2 = ax.bar(x + width/2, daily_stats['revenue'], width, label='Revenue', color='lightgreen')
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Count / Revenue')
    ax.set_title(f'Daily Sales and Revenue - Week {week_number}')
    ax.set_xticks(x)
    day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    ax.set_xticklabels([day_names.get(int(day), str(day)) for day in daily_stats['day']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'assets/week_{week_number}_daily_sales_revenue.png')
    plt.close()
    
    print(f"Saved daily sales and revenue chart for week {week_number}")


def plot_all_time_sales_and_revenue(trans_df):
    """
    Create a bar chart showing average daily sales amount and revenue across all weeks.
    X-axis shows day of the week (Monday to Sunday).
    Left y-axis for items sold, right y-axis for revenue.
    Uses the DataFrame from build_all_transaction_details().
    """
    if trans_df.empty:
        print("No transaction data available")
        return
    
    # Calculate revenue for each transaction
    trans_df = trans_df.copy()
    trans_df['revenue'] = trans_df['amount_sold'] * trans_df['sell_price']
    
    # Group by day of week and calculate average
    daily_stats = trans_df.groupby('day').agg({
        'amount_sold': 'mean',
        'revenue': 'mean'
    }).reset_index()
    
    # Create the grouped bar chart with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(daily_stats['day']))
    width = 0.35
    
    # Plot items sold on left y-axis
    bars1 = ax1.bar(x - width/2, daily_stats['amount_sold'], width, label='Avg Items Sold', color='skyblue')
    ax1.set_xlabel('Day of Week', fontsize=12)
    ax1.set_ylabel('Average Items Sold', color='skyblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    # Create second y-axis for revenue
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, daily_stats['revenue'], width, label='Avg Revenue', color='lightgreen')
    ax2.set_ylabel('Average Revenue (NOK)', color='lightgreen', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='lightgreen')
    
    # Set x-axis labels
    ax1.set_title('Average Daily Sales and Revenue - All Weeks', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    ax1.set_xticklabels([day_names.get(int(day), str(day)) for day in daily_stats['day']], rotation=45, ha='right')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('assets/all_time_daily_sales_revenue.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved all-time daily sales and revenue chart")


def plot_top_workers_by_revenue(trans_df, week_number=5, top_n=10):
    """
    Create a bar chart showing top N workers by revenue for a specific week.
    Shows revenue (left y-axis) and transaction count (right y-axis).
    Joins with workers data to display worker names.
    """
    # Filter for the specified week
    week_df = trans_df[trans_df['week'] == week_number].copy()
    
    if week_df.empty:
        print(f"No transaction data available for week {week_number}")
        return
    
    # Calculate revenue for each transaction
    week_df['revenue'] = week_df['amount_sold'] * week_df['sell_price']
    
    # Group by worker and aggregate
    worker_stats = week_df.groupby('worker').agg({
        'revenue': 'sum',
        'amount_sold': 'count'  # Count transactions
    }).rename(columns={'amount_sold': 'transaction_count'})
    
    # Sort by revenue and get top N
    worker_stats = worker_stats.sort_values('revenue', ascending=False).head(top_n)
    
    # Load worker names from JSONL file
    workers_file = 'data/workers/workers_original.jsonl'
    worker_names = {}
    if os.path.exists(workers_file):
        with open(workers_file, 'r') as f:
            for line in f:
                worker_data = json.loads(line)
                worker_id = worker_data.get('worker_id')
                name = worker_data.get('name', f'Worker {worker_id}')
                worker_names[worker_id] = name
    else:
        print(f"Warning: Workers file not found at {workers_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file at absolute path: {os.path.abspath(workers_file)}")
    
    # Add worker names to the dataframe
    worker_stats['name'] = worker_stats.index.map(lambda x: worker_names.get(x, f'Worker {x}'))
    
    # Create the chart with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(worker_stats))
    width = 0.4
    
    # Plot revenue on left y-axis
    bars1 = ax1.bar(x - width/2, worker_stats['revenue'], width, label='Revenue', color='lightgreen')
    ax1.set_xlabel('Worker', fontsize=12)
    ax1.set_ylabel('Revenue (NOK)', color='lightgreen', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='lightgreen')
    
    # Create second y-axis for transaction count
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, worker_stats['transaction_count'], width, label='Transaction Count', color='skyblue')
    ax2.set_ylabel('Number of Transactions', color='skyblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='skyblue')
    
    # Set title and x-axis labels
    ax1.set_title(f'Top Workers by Revenue - Week {week_number}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(worker_stats['name'], rotation=45, ha='right')
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'assets/week_{week_number}_top_workers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved top {top_n} workers chart for week {week_number}")
    print("\nTop Workers Summary:")
    print(worker_stats[['name', 'revenue', 'transaction_count']])


df = build_product_dataframe()
df = calculate_revenue(df)
df = calculate_profit(df)
df.sort_values(by='week_number', ascending=True, inplace=True)
print(df)
plot_items_performance(df, title_suffix='original_')
plot_week_profit_by_product(df, week_number=5)
plot_week_utilization_by_product(df, week_number=5)
plot_overall_utilization_by_product(df)

# print stats
week_5_df = df[df['week_number'] == 5]
print(week_5_df)
print(f'Total Revenue: {week_5_df["revenue"].sum()}')
print(f'Total Profit: {week_5_df["profit"].sum()}')
print(f'Total inventory bought cost: {(week_5_df["supplier_price"] * week_5_df["amount_bought"]).sum()}')
print(f'Total inventory not sold worth: {((week_5_df["amount_bought"]-week_5_df["amount_sold"])*week_5_df["supplier_price"]).sum()}')

# Build comprehensive transaction details DataFrame
all_transactions_df = build_all_transaction_details()

# Plot daily sales using the comprehensive transaction details
plot_daily_sales_and_revenue(all_transactions_df, week_number=5)
plot_all_time_sales_and_revenue(all_transactions_df)
plot_top_workers_by_revenue(all_transactions_df, week_number=5, top_n=10)