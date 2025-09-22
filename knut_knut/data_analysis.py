import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


road_function_params = {}


def load_data(file_path):
    """Load data from jsonl file into a pandas DataFrame."""
    return pd.read_json(file_path, lines=True)


def preprocess_data(df):
    """convert timestrings to minutes from midnight and create smoothed traveltime column"""
    df['depature_dt'] = pd.to_datetime(df['depature'], format='%H:%M')
    df['arrival_dt'] = pd.to_datetime(df['arrival'], format='%H:%M')

    # average dep time duplicates per road
    df = df.groupby(['road', 'depature_dt'], as_index=False).agg({
        'depature': 'first',
        'arrival': 'first',
        'depature_dt': 'first',
        'arrival_dt': 'median'
    })
    df.reset_index(drop=True, inplace=True)

    df['dep_minutes'] = df['depature_dt'].dt.hour * \
        60 + df['depature_dt'].dt.minute
    df['arr_minutes'] = df['arrival_dt'].dt.hour * \
        60 + df['arrival_dt'].dt.minute

    df['traveltime'] = df['arr_minutes'] - df['dep_minutes']  # in minutes

    # smooth traveltime using exponential moving average per road
    df = df.sort_values(['road', 'dep_minutes'])
    df['smoothed_traveltime'] = df.groupby('road')['traveltime'].transform(
        lambda x: x.ewm(alpha=0.2, adjust=False).mean()
    )

    return df


def plot_traveltime(df, output_file='traveltime_plot.png'):
    """Plot traveltime smoothed vs unsmoothed over time for all roads sorted by road and depature"""
    plt.figure(figsize=(12, 6))

    df.sort_values(['road', 'dep_minutes'], inplace=True)
    for road, group in df.groupby('road'):
        plt.plot(group['dep_minutes'], group['traveltime'],
                 marker='o', linestyle='', alpha=0.3, label=f'Raw {road}')
        plt.plot(group['dep_minutes'], group['smoothed_traveltime'],
                 linestyle='-', linewidth=2, label=f'Smoothed {road}')

    plt.xlabel('Departure Time (minutes from midnight)')
    plt.ylabel('Travel Time (minutes)')
    plt.title('Travel Time Over Time by Road')
    plt.legend()
    plt.grid(True)
    plt.show()


df = load_data('traffic.jsonl')
df = preprocess_data(df)
plot_traveltime(df)
print(road_function_params)
