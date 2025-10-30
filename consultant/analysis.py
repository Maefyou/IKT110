import pandas as pd
import json
import os
import matplotlib.pyplot as plt


# Path to the transactions directory
transactions_dir = "data/transactions"
prices_dir = "data/prices"
suppliers_prices_dir = "data/supplier_prices"
amounts_dir = "data/amounts"
workers_dir = "data/workers"

analysis_dir = "analysis"
output_dir = "output"


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


def load_workers_to_dataframe():
    '''
    Loads worker information from workers.jsonl file.
    
    Returns:
        pd.DataFrame: DataFrame with columns: name, worker_id, age, salary
    '''
    workers_file = os.path.join(workers_dir, "workers.jsonl")
    
    # Read JSONL file (one JSON object per line)
    workers_list = []
    with open(workers_file, 'r') as f:
        for line in f:
            workers_list.append(json.loads(line.strip()))
    
    # Create DataFrame
    df = pd.DataFrame(workers_list)
    
    # Display basic info
    print(f"Total workers: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nDataFrame info:")
    print(df.info())
    print(f"\nAge statistics:")
    print(df['age'].describe())
    print(f"\nSalary statistics:")
    print(df['salary'].describe())
    
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
    subdir_path = f'analysis/daily_sales_week_{week}'
    os.makedirs(subdir_path, exist_ok=True)
    
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
        plt.savefig(f'{subdir_path}/daily_sales_week_{product}.png')
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


def calculate_salary_spending(workers_df=None):
    '''
    since we are forced to pay all workers their weekly salaray even tho they dont work all week, we just sum up all workers salaries
    '''
    if workers_df is None:
        workers_df = load_workers_to_dataframe()
    
    total_salary = workers_df['salary'].sum()
    print(f"Total weekly salary spending for all workers: ${total_salary:.2f}")
    return total_salary


def calculate_inventory_spending(amounts_dict):
    '''
    Calculates the total spending to buy specific amounts of items based on supplier prices.
    
    Inputs:
        amounts_dict (dict): Dictionary with product names as keys and amounts to buy as values.
                            Example: {"apple": 100, "banana": 50, "orange": 75}
    
    Returns:
        dict: Dictionary containing total spending, per-product breakdown, and summary statistics
    '''
    
    # Load supplier prices
    supplier_prices_file = os.path.join(suppliers_prices_dir, "supplier_prices.json")
    with open(supplier_prices_file, 'r') as f:
        supplier_prices = json.load(f)
    
    # Calculate costs for each product
    product_costs = {}
    total_spending = 0
    
    for product, amount in amounts_dict.items():
        if product not in supplier_prices:
            print(f"Warning: Product '{product}' not found in supplier prices. Skipping.")
            continue
        
        price = supplier_prices[product]
        cost = price * amount
        
        product_costs[product] = {
            'amount': amount,
            'unit_price': price,
            'total_cost': cost
        }
        
        total_spending += cost
    
    # Create result summary
    result = {
        'total_spending': total_spending,
        'total_items': sum(amounts_dict.values()),
        'num_products': len(product_costs),
        'product_breakdown': product_costs
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INVENTORY PURCHASE SPENDING ANALYSIS")
    print(f"{'='*60}")
    print(f"Total products to buy: {result['num_products']}")
    print(f"Total items to buy: {result['total_items']}")
    print(f"Total spending: ${result['total_spending']:.2f}")
    print(f"Average cost per item: ${result['total_spending'] / result['total_items']:.2f}" if result['total_items'] > 0 else "N/A")
    print(f"{'='*60}")
    
    # Print breakdown by product (sorted by cost, descending)
    if product_costs:
        print("\nProduct breakdown (sorted by total cost):")
        sorted_products = sorted(product_costs.items(), key=lambda x: x[1]['total_cost'], reverse=True)
        for product, details in sorted_products:
            print(f"  {product:20s}: {details['amount']:6d} units × ${details['unit_price']:6.2f} = ${details['total_cost']:8.2f}")
    
    print(f"{'='*60}\n")
    
    return result


def calculate_sell_prices(amounts_dict, desired_profit, workers_df=None):
    '''
    Calculates optimal sell prices for products based on costs and desired profit.
    
    Inputs:
        amounts_dict (dict): Dictionary with product names as keys and amounts to buy as values.
        desired_profit (float): Desired profit amount in dollars.
        workers_df (pd.DataFrame): DataFrame containing worker information. If None, it will be loaded.
    
    Returns:
        dict: Dictionary containing sell prices for each product and detailed cost breakdown
    '''
    
    # Calculate inventory spending
    print("Calculating inventory costs...")
    inventory_result = calculate_inventory_spending(amounts_dict)
    inventory_cost = inventory_result['total_spending']
    
    # Calculate salary spending
    print("\nCalculating salary costs...")
    salary_cost = calculate_salary_spending(workers_df)
    
    # Total costs
    total_costs = inventory_cost + salary_cost
    
    # Total revenue needed to achieve desired profit
    total_revenue_needed = total_costs + desired_profit
    
    # Load supplier prices
    supplier_prices_file = os.path.join(suppliers_prices_dir, "supplier_prices.json")
    with open(supplier_prices_file, 'r') as f:
        supplier_prices = json.load(f)
    
    # Calculate total items
    total_items = sum(amounts_dict.values())
    
    # Calculate average sell price needed per item
    average_sell_price_needed = total_revenue_needed / total_items if total_items > 0 else 0
    
    # Calculate sell prices for each product using iterative approach
    # Strategy: margin_percentage = supplier_price^(1-n) / supplier_price
    # This creates inverse relationship: cheap products get higher margins, expensive get lower
    
    # Iterative search for optimal n value
    n = 0.0
    step = 0.1
    best_n = 0.0
    best_sell_prices = {}
    best_revenue = 0
    
    # Start with large steps, then refine
    for precision_level in [1.0, 0.1, 0.01, 0.001]:
        step = precision_level
        found_better = False
        
        # Try increasing n until we overshoot, then back off
        while True:
            current_n = n
            sell_prices_temp = {}
            total_revenue_temp = 0
            
            # Calculate prices for this n value
            for product, amount in amounts_dict.items():
                supplier_price = supplier_prices.get(product, 0)
                
                if supplier_price > 0:
                    # Calculate margin percentage: supplier_price^(1-n) / supplier_price
                    margin_percentage = ((supplier_price) ** (1 - current_n)) / supplier_price
                    
                    # Calculate sell price: supplier_price * (1 + margin_percentage)
                    sell_price = supplier_price * (1 + margin_percentage)
                else:
                    sell_price = supplier_price
                
                sell_prices_temp[product] = sell_price
                total_revenue_temp += sell_price * amount
            
            # Check if this n value gives us enough revenue
            if total_revenue_temp >= total_revenue_needed:
                # We have enough revenue, save this as best and try to lower prices (increase n)
                best_n = current_n
                best_sell_prices = sell_prices_temp.copy()
                best_revenue = total_revenue_temp
                n += step
                found_better = True
            else:
                # Not enough revenue, we've gone too far
                # If we found better in this precision level, use the last good value
                if found_better:
                    n = best_n
                else:
                    # Need to go back and try smaller n
                    n -= step
                break
        
        # Reset to best_n for next precision level
        n = best_n
    
    # If we never found a solution that meets revenue, use the last attempt
    if not best_sell_prices:
        n = 0.0
        for product, amount in amounts_dict.items():
            supplier_price = supplier_prices.get(product, 0)
            if supplier_price > 0:
                margin_percentage = (supplier_price ** (1 - n)) / supplier_price
                sell_price = supplier_price * (1 + margin_percentage)
            else:
                sell_price = supplier_price
            best_sell_prices[product] = sell_price
            best_revenue += sell_price * amount
    
    # Round prices and calculate final revenue
    sell_prices = {}
    total_revenue = 0
    for product, price in best_sell_prices.items():
        sell_prices[product] = round(price, 2)
        total_revenue += sell_prices[product] * amounts_dict[product]
    
    # Calculate actual profit
    actual_profit = total_revenue - total_costs
    
    # Calculate average markup percentage (for display purposes)
    if inventory_cost > 0:
        avg_markup_percentage = (total_revenue - inventory_cost) / inventory_cost
    else:
        avg_markup_percentage = 0
    
    # Store the n value used
    optimal_n = best_n
    
    # Create result
    result = {
        'sell_prices': sell_prices,
        'inventory_cost': inventory_cost,
        'salary_cost': salary_cost,
        'total_costs': total_costs,
        'desired_profit': desired_profit,
        'total_revenue_needed': total_revenue_needed,
        'actual_total_revenue': total_revenue,
        'actual_profit': actual_profit,
        'markup_percentage': avg_markup_percentage * 100,
        'total_items': total_items,
        'average_sell_price': total_revenue / total_items if total_items > 0 else 0,
        'profit_margin': (actual_profit / total_revenue * 100) if total_revenue > 0 else 0,
        'optimal_n': optimal_n
    }
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"SELL PRICE CALCULATION")
    print(f"{'='*70}")
    print(f"\nCOST BREAKDOWN:")
    print(f"  Inventory costs:           ${inventory_cost:>15,.2f}")
    print(f"  Salary costs:              ${salary_cost:>15,.2f}")
    print(f"  {'─'*45}")
    print(f"  Total costs:               ${total_costs:>15,.2f}")
    
    print(f"\nPROFIT TARGETS:")
    print(f"  Desired profit:            ${desired_profit:>15,.2f}")
    print(f"  Revenue needed:            ${total_revenue_needed:>15,.2f}")
    
    print(f"\nPRICING STRATEGY:")
    print(f"  Optimal n parameter:       {optimal_n:>14.3f}")
    print(f"  Average markup:            {avg_markup_percentage * 100:>14.2f}%")
    print(f"  Formula: margin% = price^(1-n) / price")
    print(f"  Total items to sell:       {total_items:>15,}")
    print(f"  Average sell price/item:   ${result['average_sell_price']:>15,.2f}")
    
    print(f"\nPROJECTED RESULTS (if all items sell):")
    print(f"  Total revenue:             ${total_revenue:>15,.2f}")
    print(f"  Actual profit:             ${actual_profit:>15,.2f}")
    print(f"  Profit margin:             {result['profit_margin']:>14.2f}%")
    print(f"  Difference from desired:   ${actual_profit - desired_profit:>15,.2f}")
    
    print(f"\n{'='*70}")
    print(f"SELL PRICES BY PRODUCT:")
    print(f"{'='*70}")
    print(f"{'Product':<20} {'Supplier $':>12} {'Sell $':>12} {'Markup':>12} {'Quantity':>12}")
    print(f"{'-'*70}")
    
    for product in sorted(sell_prices.keys()):
        supplier_price = supplier_prices.get(product, 0)
        sell_price = sell_prices[product]
        markup = ((sell_price - supplier_price) / supplier_price * 100) if supplier_price > 0 else 0
        quantity = amounts_dict[product]
        print(f"{product:<20} ${supplier_price:>11.2f} ${sell_price:>11.2f} {markup:>11.2f}% {quantity:>12,}")
    
    print(f"{'='*70}\n")
    
    return result


def write_amounts_and_prices(amounts_dict, prices_dict):
    '''
    Writes amounts and prices dictionaries to JSON files in the output directory.
    
    Inputs:
        amounts_dict (dict): Dictionary with product names as keys and amounts as values.
        prices_dict (dict): Dictionary with product names as keys and prices as values.
    
    Returns:
        tuple: Paths to the created files (amounts_file, prices_file)
    '''
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write amounts to new_amounts.json
    amounts_file = os.path.join(output_dir, "new_amounts.json")
    with open(amounts_file, 'w') as f:
        json.dump(amounts_dict, f, indent=2)
    print(f"Amounts saved to {amounts_file}")
    
    # Write prices to new_prices.json
    prices_file = os.path.join(output_dir, "new_prices.json")
    with open(prices_file, 'w') as f:
        json.dump(prices_dict, f, indent=2)
    print(f"Prices saved to {prices_file}")
    
    return amounts_file, prices_file


def find_products_sold_with_remaining_inventory(transactions_df=None, product_info_df=None):
    '''
    Creates a DataFrame containing rows for days where a product was sold 
    and there was still inventory remaining at the start of the day.
    
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded.
        product_info_df (pd.DataFrame): DataFrame containing product info. If None, it will be loaded.
    
    Returns:
        pd.DataFrame: DataFrame with columns: week, day, product, amount_sold, remaining_inventory
    '''
    
    if transactions_df is None:
        transactions_df = load_transactions_to_dataframe()
    
    if product_info_df is None:
        product_info_df = load_product_info_to_dataframe()
    
    # Group transactions by week, day, and product to get daily sales
    daily_sales = transactions_df.groupby(['week', 'day', 'product'])['amount'].sum().reset_index()
    daily_sales.columns = ['week', 'day', 'product', 'amount_sold']
    
    # Get initial inventory for each product per week
    inventory_map = product_info_df.set_index(['week', 'product'])['amount_bought'].to_dict()
    
    # List to store results
    results = []
    
    # Process each week
    for week in sorted(transactions_df['week'].unique()):
        week_sales = daily_sales[daily_sales['week'] == week].copy()
        
        # Get unique products in this week
        products = week_sales['product'].unique()
        
        for product in products:
            # Get initial inventory for this product in this week
            initial_inventory = inventory_map.get((week, product), 0)
            
            # Get sales for this product, sorted by day
            product_sales = week_sales[week_sales['product'] == product].sort_values('day')
            
            # Track cumulative sales
            cumulative_sales = 0
            
            for _, row in product_sales.iterrows():
                day = row['day']
                amount_sold = row['amount_sold']
                
                # Add to cumulative sales
                cumulative_sales += amount_sold
                
                # Calculate remaining inventory at end of this day
                remaining_inventory = initial_inventory - cumulative_sales
                current_inventory = initial_inventory - (cumulative_sales - amount_sold)
                
                # Only include if product was sold (amount_sold > 0) and current inventory > 0
                if amount_sold > 0 and current_inventory > 0:
                    results.append({
                        'week': week,
                        'day': day,
                        'product': product,
                        'amount_sold': amount_sold,
                        'remaining_inventory': remaining_inventory
                    })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display basic info
    if not df.empty:
        print(f"\n{'='*70}")
        print(f"PRODUCTS SOLD WITH REMAINING INVENTORY")
        print(f"{'='*70}")
        print(f"Total occurrences: {len(df)}")
        print(f"Unique weeks: {sorted(df['week'].unique())}")
        print(f"Unique products: {sorted(df['product'].unique())}")
        print(f"\nFirst 10 rows:")
        print(df.head(10))

        
    df = df.groupby(['week', 'product']).agg(
        avg_amount_sold=('amount_sold', 'mean'),
        total_amount_sold=('amount_sold', 'sum'),
        days_with_sales=('day', 'count'),
        remaining_inventory=('remaining_inventory', 'min')
    ).reset_index()
    return df


def estimate_purchase_amounts(transactions_df=None, product_info_df=None, target_week=None, days_to_stock=7):
    '''
    Estimates the amount to buy for each product based on average daily sales.
    Uses data from find_products_sold_with_remaining_inventory.
    
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded.
        product_info_df (pd.DataFrame): DataFrame containing product info. If None, it will be loaded.
        target_week (int): If specified, uses only that week's data. If None, uses all weeks.
        days_to_stock (int): Number of days to stock for (default: 7 for a full week).
    
    Returns:
        dict: Dictionary with product names as keys and recommended purchase amounts as values
    '''
    
    # Get the dataframe with sales and remaining inventory data
    df = find_products_sold_with_remaining_inventory(transactions_df, product_info_df)
    
    if df.empty:
        print("No data available to estimate purchase amounts.")
        return {}
    
    # Filter by target week if specified
    if target_week is not None:
        df = df[df['week'] == target_week]
        if df.empty:
            print(f"No data available for week {target_week}")
            return {}
    
    # Calculate purchase amounts based on average daily sales
    purchase_amounts = {}
    
    print(f"\n{'='*70}")
    print(f"ESTIMATED PURCHASE AMOUNTS")
    print(f"{'='*70}")
    print(f"Based on: {'Week ' + str(target_week) if target_week is not None else 'All weeks'}")
    print(f"Stocking for: {days_to_stock} days")
    print(f"\n{'Product':<20} {'Avg/Day':>12} {'Days Sold':>12} {'Recommended':>12}")
    print(f"{'-'*70}")
    
    for _, row in df.iterrows():
        product = row['product']
        avg_daily = row['avg_amount_sold']
        days_with_sales = row['days_with_sales']
        
        # Estimate: avg daily sales × days to stock for
        estimated = int(avg_daily * days_to_stock)
        
        purchase_amounts[product] = estimated
        
        print(f"{product:<20} {avg_daily:>12.1f} {days_with_sales:>12} {estimated:>12}")
    
    total_items = sum(purchase_amounts.values())
    print(f"{'-'*70}")
    print(f"{'TOTAL':<20} {'':<12} {'':<12} {total_items:>12}")
    print(f"{'='*70}\n")
    
    return purchase_amounts


def create_schedule(workers_per_department_per_shift=10):
    workers = load_workers_to_dataframe()

    # distribute all workers evenly across days and shifts
    schedule = dict()
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    shifts = [1, 2]
    departments = ['registers', 'utilities']
    worker_index = 0
    
    total_shifts_assigned = 0
    for day in days:
        schedule[day] = []
        for shift in shifts:
            for department in departments:
                # assign workers per department per shift
                for _ in range(workers_per_department_per_shift):
                    total_shifts_assigned += 1
                    if worker_index >= len(workers):
                        worker_index = 0
                    worker_id = workers.iloc[worker_index]['worker_id']
                    schedule[day].append({
                        "worker_id": worker_id,
                        "department": department,
                        "shift": shift
                    })
                    worker_index += 1
    
    # write schedule to json file
    os.makedirs(output_dir, exist_ok=True)
    schedule_file = os.path.join(output_dir, "new_schedule.json")
    with open(schedule_file, 'w') as f:
        json.dump(schedule, f, indent=2)
    print(f"New schedule created and saved to {schedule_file}")
    print(f'total workers: {len(workers)}')
    print(f"Total shifts assigned: {total_shifts_assigned}")

    return True


def analyze_weekly_product_performance(transactions_df=None, product_info_df=None):
    '''
    Analyzes financial performance for each week and product.
    
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded.
        product_info_df (pd.DataFrame): DataFrame containing product info (amounts and prices). If None, it will be loaded.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - week: Week number
            - product: Product name
            - amount_bought: Amount purchased from supplier
            - supplier_price: Price per unit from supplier
            - amount_sold: Total amount sold during the week
            - sell_price: Price per unit sold to customers
            - inventory_spending: Total cost to purchase inventory (amount_bought × supplier_price)
            - revenue: Total revenue from sales (amount_sold × sell_price)
            - profit: Revenue minus inventory spending
            - inventory_wasted: Amount of inventory not sold (amount_bought - amount_sold)
            - inventory_wasted_worth: Value of wasted inventory at 25% of supplier price
    '''
    
    if transactions_df is None:
        transactions_df = load_transactions_to_dataframe()
    
    if product_info_df is None:
        product_info_df = load_product_info_to_dataframe()
    
    # Load supplier prices (constant across all weeks)
    supplier_prices_file = os.path.join(suppliers_prices_dir, "supplier_prices.json")
    with open(supplier_prices_file, 'r') as f:
        supplier_prices = json.load(f)
    
    # Calculate total amount sold per week per product
    weekly_sales = transactions_df.groupby(['week', 'product'])['amount'].sum().reset_index()
    weekly_sales.columns = ['week', 'product', 'amount_sold']
    
    # Merge with product info (amounts bought and sell prices)
    # product_info_df has: week, product, amount_bought, supplier_price, sell_price
    merged_df = pd.merge(
        product_info_df,
        weekly_sales,
        on=['week', 'product'],
        how='left'
    )
    
    # Fill NaN values in amount_sold with 0 (products that were bought but not sold)
    merged_df['amount_sold'] = merged_df['amount_sold'].fillna(0)
    
    # Calculate financial metrics
    merged_df['inventory_spending'] = merged_df['amount_bought'] * merged_df['supplier_price']
    merged_df['revenue'] = merged_df['amount_sold'] * merged_df['sell_price']
    merged_df['profit'] = merged_df['revenue'] - merged_df['inventory_spending']
    merged_df['inventory_wasted'] = merged_df['amount_bought'] - merged_df['amount_sold']
    merged_df['inventory_wasted_worth'] = merged_df['inventory_wasted'] * merged_df['supplier_price']
    
    # Ensure inventory_wasted is not negative (shouldn't happen, but just in case)
    merged_df['inventory_wasted'] = merged_df['inventory_wasted'].clip(lower=0)
    merged_df['inventory_wasted_worth'] = merged_df['inventory_wasted_worth'].clip(lower=0)
    
    # Select and order columns
    result_df = merged_df[[
        'week',
        'product',
        'amount_bought',
        'supplier_price',
        'amount_sold',
        'sell_price',
        'inventory_spending',
        'revenue',
        'profit',
        'inventory_wasted',
        'inventory_wasted_worth'
    ]]
    
    # Sort by week and product
    result_df = result_df.sort_values(['week', 'product']).reset_index(drop=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"WEEKLY PRODUCT PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total records: {len(result_df)}")
    print(f"Weeks analyzed: {sorted(result_df['week'].unique())}")
    print(f"Products: {sorted(result_df['product'].unique())}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY:")
    print(f"{'='*80}")
    print(f"Total inventory spending:      ${result_df['inventory_spending'].sum():>15,.2f}")
    print(f"Total revenue:                 ${result_df['revenue'].sum():>15,.2f}")
    print(f"Total profit:                  ${result_df['profit'].sum():>15,.2f}")
    print(f"Total inventory wasted:        {result_df['inventory_wasted'].sum():>15,.0f} units")
    print(f"Total wasted inventory worth:  ${result_df['inventory_wasted_worth'].sum():>15,.2f}")
    
    # Summary by week
    print(f"\n{'='*80}")
    print(f"SUMMARY BY WEEK:")
    print(f"{'='*80}")
    weekly_summary = result_df.groupby('week').agg({
        'inventory_spending': 'sum',
        'revenue': 'sum',
        'profit': 'sum',
        'inventory_wasted': 'sum',
        'inventory_wasted_worth': 'sum'
    }).reset_index()
    
    print(f"\n{'Week':>6} {'Spending':>15} {'Revenue':>15} {'Profit':>15} {'Wasted':>10} {'Wasted $':>15}")
    print(f"{'-'*80}")
    for _, row in weekly_summary.iterrows():
        print(f"{row['week']:>6} ${row['inventory_spending']:>14,.2f} ${row['revenue']:>14,.2f} ${row['profit']:>14,.2f} {row['inventory_wasted']:>9,.0f} {row['inventory_wasted_worth']:>14,.2f}")
    
    print(f"\n{'='*80}")
    print(f"First 10 rows of detailed data:")
    print(result_df.tail(13).to_string())
    print(f"{'='*80}\n")
    
    return result_df


def plot_transactions_per_week(transactions_df=None):
    '''
    Counts the total number of transactions per week and plots it as a line plot.
    
    Inputs:
        transactions_df (pd.DataFrame): DataFrame containing transaction data. If None, it will be loaded.
    
    Returns:
        pd.DataFrame: DataFrame with columns: week, transaction_count
    '''
    
    if transactions_df is None:
        transactions_df = load_transactions_to_dataframe()
    
    # Count unique transactions per week
    # Each row in the dataframe represents a product in a transaction
    # We need to group by week, customer_id, and day to count unique transactions
    transactions_per_week = transactions_df.groupby('week').size().reset_index(name='transaction_count')
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TRANSACTIONS PER WEEK")
    print(f"{'='*50}")
    print(f"{'Week':>6} {'Transaction Count':>20}")
    print(f"{'-'*50}")
    
    for _, row in transactions_per_week.iterrows():
        print(f"{row['week']:>6} {row['transaction_count']:>20,}")
    
    total = transactions_per_week['transaction_count'].sum()
    print(f"{'-'*50}")
    print(f"{'TOTAL':>6} {total:>20,}")
    print(f"{'='*50}\n")
    
    # Create line plot
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    plt.plot(transactions_per_week['week'], transactions_per_week['transaction_count'], 
             marker='o', linewidth=2, markersize=8, color='tab:cyan')
    
    plt.xlabel('Week')
    plt.ylabel('Transaction Count')
    plt.title('Total Transactions per Week')
    plt.grid(True, alpha=0.3)
    plt.xticks(transactions_per_week['week'])
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(analysis_dir, exist_ok=True)
    plt.savefig(f'{analysis_dir}/transactions_per_week.png')
    plt.close()
    
    print(f"Line plot saved to {analysis_dir}/transactions_per_week.png")
    
    return transactions_per_week



plot_transactions_per_week()