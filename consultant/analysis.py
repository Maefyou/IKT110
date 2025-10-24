import pandas as pd
import json
import os
import matplotlib.pyplot as plt


# Path to the transactions directory
transactions_dir = "data/transactions"
prices_dir = "data/prices"
suppliers_prices_dir = "data/supplier_prices"
amounts_dir = "data/amounts"

def load_transactions_to_dataframe():
    # Initialize list to store all transaction rows
    all_transactions = []

    # Get all transaction files
    transaction_files = sorted([f for f in os.listdir(transactions_dir) if f.startswith("transactions_") and f.endswith(".json")])

    # Process each transaction file
    for file_name in transaction_files:
        # Extract week number from filename (e.g., "transactions_0.json" -> week 0)
        week = int(file_name.split("_")[1].split(".")[0])
        
        file_path = os.path.join(transactions_dir, file_name)
        
        # Load the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each day's transactions
        for day, transactions in data.items():
            day = int(day)
            
            # Process each transaction
            for transaction in transactions:
                customer_id = transaction.get("customer_id")
                worker_id = transaction.get("register_worker")
                transaction_type = transaction.get("transaction_type")
                
                merch_types = transaction.get("merch_types", [])
                merch_amounts = transaction.get("merch_amounts", [])
                
                # Create a row for each product in the transaction
                for product, amount in zip(merch_types, merch_amounts):
                    all_transactions.append({
                        "week": week,
                        "day": day,
                        "product": product,
                        "amount": amount,
                        "worker_id": worker_id,
                        "customer_id": customer_id,
                        "transactiontype": transaction_type
                    })

    # Create DataFrame
    df = pd.DataFrame(all_transactions)

    # Display basic info
    print(f"Total rows: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataFrame info:")
    print(df.info())
    print(f"\nUnique weeks: {sorted(df['week'].unique())}")
    print(f"\nUnique days: {sorted(df['day'].unique())}")
    print(f"\nUnique products: {sorted(df['product'].unique())}")
    print(f"\nTransaction types: {df['transactiontype'].unique()}")

    return df


def load_product_info_to_dataframe():
    '''
    Loads product information from prices, amounts, and supplier_prices directories.
    Combines them based on product and week.
    
    Returns:
        pd.DataFrame: DataFrame with columns: week, product, amount_bought, supplier_price, sell_price
    '''
    # Load supplier prices (single file, applies to all weeks)
    supplier_prices_file = os.path.join(suppliers_prices_dir, "supplier_prices.json")
    with open(supplier_prices_file, 'r') as f:
        supplier_prices = json.load(f)
    
    # Get all weekly files from prices and amounts directories
    price_files = sorted([f for f in os.listdir(prices_dir) if f.startswith("prices_") and f.endswith(".json")])
    amount_files = sorted([f for f in os.listdir(amounts_dir) if f.startswith("amounts_") and f.endswith(".json")])
    
    # Initialize list to store all product rows
    all_products = []
    
    # Process each week
    for price_file, amount_file in zip(price_files, amount_files):
        # Extract week number from filename (e.g., "prices_0.json" -> week 0)
        week = int(price_file.split("_")[1].split(".")[0])
        
        # Load prices for this week
        price_path = os.path.join(prices_dir, price_file)
        with open(price_path, 'r') as f:
            prices = json.load(f)
        
        # Load amounts for this week
        amount_path = os.path.join(amounts_dir, amount_file)
        with open(amount_path, 'r') as f:
            amounts = json.load(f)
        
        # Combine data for each product
        for product_name in prices.keys():
            all_products.append({
                "week": week,
                "product": product_name,
                "amount_bought": amounts.get(product_name, 0),
                "supplier_price": supplier_prices.get(product_name, 0),
                "sell_price": prices.get(product_name, 0)
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_products)
    
    # Display basic info
    print(f"Total product entries: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataFrame info:")
    print(df.info())
    
    return df


def plot_products_week_daily_sells(df = None, week = 5):
    '''
    Plots daily sales of all products over time.
    Inputs:
        df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded from files.
    Outputs:
        A line plot showing daily sales of all products over time.
    '''
    
    if df is None:
        df = load_transactions_to_dataframe()

    # Group by week, day, and product to get total amount sold per day for each product
    daily_sales = df.groupby(['week', 'day', 'product'])['amount'].sum().reset_index()
        
    # reduce transactions down to only one week
    daily_sales = daily_sales[daily_sales['week'] == week]
    
    # Create a complete range of days (1-7)
    all_days = list(range(1, 8))
        
    # Create the plot
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    # Plot a line for each product in the same graph
    for product in daily_sales['product'].unique():
        product_sales = daily_sales[daily_sales['product'] == product]
        
        # Create a complete day-amount mapping with 0 for missing days
        day_amount_map = dict(zip(product_sales['day'], product_sales['amount']))
        amounts = [day_amount_map.get(day, 0) for day in all_days]
        
        plt.plot(all_days, amounts, marker='o', label=product)
    
    plt.xlabel('Day')
    plt.ylabel('Amount Sold')
    plt.title(f'Daily Sales by Product - Week {week}')
    plt.xticks(all_days)  # Ensure all days are shown on x-axis
    plt.ylim(bottom=0)  # Set y-axis to start at 0
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'analysis/daily_sales_week_{week}.png')

    
    # Plot every product alone in its own graph
    os.makedirs('analysis/daily_sales_week', exist_ok=True)
    
    # Load product info to get initial inventory amounts
    product_info_df = load_product_info_to_dataframe()
    
    for product in daily_sales['product'].unique():
        # Create the plot with two y-axes
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
        product_sales = daily_sales[daily_sales['product'] == product]
        
        # Create a complete day-amount mapping with 0 for missing days
        day_amount_map = dict(zip(product_sales['day'], product_sales['amount']))
        amounts = [day_amount_map.get(day, 0) for day in all_days]
        
        # Plot daily sales on first y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Amount Sold', color=color1)
        ax1.plot(all_days, amounts, marker='o', label=f'{product} - Daily Sales', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(all_days)
        ax1.set_ylim(bottom=0)  # Set y-axis to start at 0
        
        # Calculate remaining inventory for each day
        # Get initial inventory for this product and week
        product_week_info = product_info_df[(product_info_df['product'] == product) & 
                                            (product_info_df['week'] == week)]
        
        if not product_week_info.empty:
            initial_inventory = product_week_info['amount_bought'].iloc[0]
            
            # Calculate cumulative sales and remaining inventory
            remaining_inventory = []
            cumulative_sales = 0
            for day_amount in amounts:
                cumulative_sales += day_amount
                remaining_inventory.append(initial_inventory - cumulative_sales)
            
            # Plot remaining inventory on second y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Remaining Inventory', color=color2)
            ax2.plot(all_days, remaining_inventory, marker='s', label=f'{product} - Remaining Inventory', 
                    color=color2, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(bottom=0)  # Set y-axis to start at 0
        
        plt.title(f'Daily Sales and Inventory - {product} - Week {week}')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        if not product_week_info.empty:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(lines1, labels1, loc='upper right')
        
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.savefig(f'analysis/daily_sales_week/daily_sales_week_{product}.png')
        plt.close(fig)


def plot_worker_sales_performance_week(transactions_df=None, week=5):
    '''
    Plots worker sales performance over all days in transactions.
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded from files.
        week (int): The week number to analyze.
    Outputs:
        - A combined plot showing all workers' sales per day (normalized)
        - Individual plots for each worker showing total sales and revenue per day
    '''
    
    if transactions_df is None:
        transactions_df = load_transactions_to_dataframe()
    
    # Filter transactions for the specified week
    week_transactions = transactions_df[transactions_df['week'] == week].copy()
    
    # Load prices for revenue calculation
    product_info_df = load_product_info_to_dataframe()
    week_prices = product_info_df[product_info_df['week'] == week].set_index('product')['sell_price'].to_dict()
    
    # Get all unique days in the transactions for this week
    all_days = sorted(week_transactions['day'].unique())
    if len(all_days) == 0:
        print(f"No transactions found for week {week}")
        return
    
    # Calculate sales per worker per day
    worker_daily_sales = week_transactions.groupby(['worker_id', 'day'])['amount'].sum().reset_index()
    worker_daily_sales.columns = ['worker_id', 'day', 'total_items']
    
    # Get all unique workers
    all_workers = week_transactions['worker_id'].unique()
    
    # Count days each worker worked
    worker_days_worked = week_transactions.groupby('worker_id')['day'].nunique().to_dict()
    
    # Prepare data for combined plot - normalized per day
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for worker_id in all_workers:
        worker_sales = worker_daily_sales[worker_daily_sales['worker_id'] == worker_id]
        
        # Get daily sales for all days
        daily_sales = []
        for day in all_days:
            day_sales = worker_sales[worker_sales['day'] == day]['total_items'].sum()
            daily_sales.append(day_sales)
        
        # Calculate average sales per day for this worker
        total_sales = sum(daily_sales)
        days_worked = worker_days_worked.get(worker_id, 1)
        avg_per_day = total_sales / days_worked if days_worked > 0 else 0
        
        # Only plot if worker has any sales
        if total_sales > 0:
            ax.plot(all_days, daily_sales, marker='o', label=f'{worker_id[-8:]} (avg: {avg_per_day:.1f}/day)', alpha=0.7)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Items Sold per Day')
    ax.set_title(f'Worker Sales Performance (Per Day) - Week {week}')
    ax.set_xticks(all_days)
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'analysis/worker_performance_week_{week}.png', bbox_inches='tight')
    plt.close(fig)
    
    # Create individual worker plots
    os.makedirs('analysis/worker_performance', exist_ok=True)
    
    for worker_id in all_workers:
        worker_sales = worker_daily_sales[worker_daily_sales['worker_id'] == worker_id]
        
        if worker_sales.empty:
            continue
        
        # Calculate total items sold per day
        daily_items = []
        for day in all_days:
            day_items = worker_sales[worker_sales['day'] == day]['total_items'].sum()
            daily_items.append(day_items)
        
        # Calculate revenue per day
        worker_day_transactions = week_transactions[week_transactions['worker_id'] == worker_id]
        daily_revenue = []
        for day in all_days:
            day_trans = worker_day_transactions[worker_day_transactions['day'] == day]
            revenue = 0
            for _, row in day_trans.iterrows():
                product = row['product']
                amount = row['amount']
                price = week_prices.get(product, 0)
                revenue += amount * price
            daily_revenue.append(revenue)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Total Items Sold', color=color1)
        ax1.plot(all_days, daily_items, marker='o', label='Items Sold', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(all_days)
        ax1.set_ylim(bottom=0)
        
        # Second y-axis for revenue
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Revenue Generated ($)', color=color2)
        ax2.plot(all_days, daily_revenue, marker='s', label='Revenue', color=color2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)
        
        # Calculate totals for title
        total_items = sum(daily_items)
        total_revenue = sum(daily_revenue)
        
        plt.title(f'Worker Performance - {worker_id[-12:]} - Week {week}\nTotal: {total_items} items, ${total_revenue:.2f} revenue')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'analysis/worker_performance/worker_{worker_id}.png')
        plt.close(fig)
    
    print(f"Worker performance plots created for week {week}")


def plot_worker_sales_performance_all(transactions_df=None):
    '''
    Plots worker sales performance over all weeks and days in transactions.
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded from files.
    Outputs:
        - A combined plot showing all workers' sales per day across all weeks
        - Individual plots for each worker showing total sales and revenue per day across all weeks
    '''
    
    if transactions_df is None:
        transactions_df = load_transactions_to_dataframe()
    
    # Create a combined week-day identifier for sorting
    transactions_df = transactions_df.copy()
    transactions_df['week_day'] = transactions_df['week'] * 10 + transactions_df['day']
    
    # Get all unique week-day combinations sorted
    all_week_days = sorted(transactions_df['week_day'].unique())
    
    if len(all_week_days) == 0:
        print(f"No transactions found")
        return
    
    # Load product info for all weeks for revenue calculation
    product_info_df = load_product_info_to_dataframe()
    
    # Calculate sales per worker per week-day
    worker_daily_sales = transactions_df.groupby(['worker_id', 'week', 'day', 'week_day'])['amount'].sum().reset_index()
    worker_daily_sales.columns = ['worker_id', 'week', 'day', 'week_day', 'total_items']
    
    # Get all unique workers
    all_workers = transactions_df['worker_id'].unique()
    
    # Count days each worker worked
    worker_days_worked = transactions_df.groupby('worker_id')['week_day'].nunique().to_dict()
    
    # Prepare data for combined plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for worker_id in all_workers:
        worker_sales = worker_daily_sales[worker_daily_sales['worker_id'] == worker_id]
        
        # Get daily sales for all week-days
        daily_sales = []
        for week_day in all_week_days:
            day_sales = worker_sales[worker_sales['week_day'] == week_day]['total_items'].sum()
            daily_sales.append(day_sales)
        
        # Calculate average sales per day for this worker
        total_sales = sum(daily_sales)
        days_worked = worker_days_worked.get(worker_id, 1)
        avg_per_day = total_sales / days_worked if days_worked > 0 else 0
        
        # Only plot if worker has any sales
        if total_sales > 0:
            ax.plot(range(len(all_week_days)), daily_sales, marker='o', 
                   label=f'{worker_id[-8:]} (avg: {avg_per_day:.1f}/day)', alpha=0.7, markersize=4)
    
    # Create x-axis labels showing week and day
    x_labels = [f'W{wd//10}D{wd%10}' for wd in all_week_days]
    ax.set_xlabel('Week-Day')
    ax.set_ylabel('Items Sold per Day')
    ax.set_title(f'Worker Sales Performance (Per Day) - All Weeks')
    ax.set_xticks(range(len(all_week_days)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'analysis/worker_performance_all_weeks.png', bbox_inches='tight')
    plt.close(fig)
    
    # Create individual worker plots
    os.makedirs('analysis/worker_performance_all', exist_ok=True)
    
    for worker_id in all_workers:
        worker_sales = worker_daily_sales[worker_daily_sales['worker_id'] == worker_id]
        
        if worker_sales.empty:
            continue
        
        # Calculate total items sold per day
        daily_items = []
        daily_revenue = []
        
        for week_day in all_week_days:
            week = week_day // 10
            day = week_day % 10
            
            # Get items sold
            day_items = worker_sales[worker_sales['week_day'] == week_day]['total_items'].sum()
            daily_items.append(day_items)
            
            # Calculate revenue for this day
            worker_day_transactions = transactions_df[
                (transactions_df['worker_id'] == worker_id) & 
                (transactions_df['week'] == week) & 
                (transactions_df['day'] == day)
            ]
            
            # Get prices for this week
            week_prices = product_info_df[product_info_df['week'] == week].set_index('product')['sell_price'].to_dict()
            
            revenue = 0
            for _, row in worker_day_transactions.iterrows():
                product = row['product']
                amount = row['amount']
                price = week_prices.get(product, 0)
                revenue += amount * price
            daily_revenue.append(revenue)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 6))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Week-Day')
        ax1.set_ylabel('Total Items Sold', color=color1)
        ax1.plot(range(len(all_week_days)), daily_items, marker='o', label='Items Sold', color=color1, markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(range(len(all_week_days)))
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylim(bottom=0)
        
        # Second y-axis for revenue
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Revenue Generated ($)', color=color2)
        ax2.plot(range(len(all_week_days)), daily_revenue, marker='s', label='Revenue', 
                color=color2, linestyle='--', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)
        
        # Calculate totals for title
        total_items = sum(daily_items)
        total_revenue = sum(daily_revenue)
        
        plt.title(f'Worker Performance - {worker_id[-12:]} - All Weeks\nTotal: {total_items} items, ${total_revenue:.2f} revenue')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'analysis/worker_performance_all/worker_{worker_id}.png')
        plt.close(fig)
    
    print(f"Worker performance plots created for all weeks")


df = pd.DataFrame()
df = load_transactions_to_dataframe()
plot_worker_sales_performance_all(df)