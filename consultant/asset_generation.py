import json
import matplotlib.pyplot as plt
import os

def read_performance_data(filepath):
    """Read performance data from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_performance_chart(data, index=0, output_dir='assets'):
    """Create a bar chart showing financial performance."""
    # Extract values from data
    revenue = data['Revenue']
    salary_cost = data['Salary']
    inventory_purchase = data['Inventory_Bought_Cost']
    inventory_waste = data['Inventory_Wasted_Worth']
    gross_profit = data['Revenue'] - (salary_cost + inventory_purchase)
    
    # Set up the figure with dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Categories and values
    categories = ['Revenue', 'Salary cost', 'Inventory\npurchase', 'Inventory\nwaste', 'Gross profit']
    values = [revenue, salary_cost, inventory_purchase, inventory_waste, gross_profit]
    
    # Colors for each bar (matching the image style)
    colors = ['#2d8c72', '#4a5f8c', '#5a93a8', '#8b6f47', '#b84c4c']
    
    # Create bars
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of each bar
    for bar, value in zip(bars, values):
        height = bar.get_height()
        # Format the label
        if value >= 0:
            label = f'${value:,.0f}'
            va = 'bottom'
            y_pos = height
        else:
            label = f'-${abs(value):,.0f}'
            va = 'top'
            y_pos = height
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                label,
                ha='center', va=va, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bar.get_facecolor(), 
                         edgecolor='none', alpha=0.8))
    
    # Customize the plot
    ax.set_ylabel('Amount in NOK', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'performance_chart_{index}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {output_path}")
    return output_path

def main():
    """Main function to process all performance data and generate charts."""
    # Path to the performance data file
    performance_file = 'analysis/Performance.jsonl'
    
    # Check if file exists
    if not os.path.exists(performance_file):
        print(f"Error: {performance_file} not found!")
        return
    
    # Read all performance data
    performance_data = read_performance_data(performance_file)
    
    print(f"Found {len(performance_data)} performance record(s)")
    
    # Generate a chart for each line in the file
    for i, data in enumerate(performance_data):
        print(f"\nGenerating chart {i+1}/{len(performance_data)}...")
        create_performance_chart(data, index=i)
    
    print(f"\nAll charts generated successfully!")

if __name__ == '__main__':
    main()
