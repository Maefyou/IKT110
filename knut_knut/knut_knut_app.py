import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from flask import request
from flask import send_from_directory
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues in Flask

app = Flask(__name__)
df = None  # Global variable to hold the DataFrame
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


def get_travel_times_at(dep_time):
    """Get travel times for all roads at a specific departure time."""
    times = {}

    for road_name in df['road'].unique():  # Fixed: iterate over unique road names
        road_data = df[df['road'] == road_name]  # Filter data for this road
        road_data = road_data.sort_values(
            'dep_minutes')  # Sort by departure time

        # If the exact departure time exists, use it
        exact_match = road_data[road_data['dep_minutes'] == dep_time]
        if not exact_match.empty:
            travel_time = exact_match['smoothed_traveltime'].iloc[0]
        else:
            # Linear interpolation between closest times

            # Find times before and after
            before_times = road_data[road_data['dep_minutes'] <= dep_time]
            after_times = road_data[road_data['dep_minutes'] >= dep_time]

            if before_times.empty and after_times.empty:
                # No data for this road
                travel_time = np.nan
            elif before_times.empty:
                # Only future times available - use the earliest
                travel_time = after_times['smoothed_traveltime'].iloc[0]
            elif after_times.empty:
                # Only past times available - use the latest
                travel_time = before_times['smoothed_traveltime'].iloc[-1]
            else:
                # Interpolate between before and after
                before_row = before_times.iloc[-1]  # Latest time before
                after_row = after_times.iloc[0]     # Earliest time after

                # Linear interpolation
                x1, y1 = before_row['dep_minutes'], before_row['smoothed_traveltime']
                x2, y2 = after_row['dep_minutes'], after_row['smoothed_traveltime']

                if x1 == x2:  # Same time (shouldn't happen, but safety check)
                    travel_time = y1
                else:
                    # Linear interpolation formula: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
                    travel_time = y1 + (y2 - y1) * (dep_time - x1) / (x2 - x1)

        times[road_name] = travel_time

    return times


def get_best_route(dep_hour=0, dep_min=0):
    """Get the best route and estimated travel time for a given departure time."""
    out = ""
    if not dep_hour.isdigit():
        dep_hour = 00
        out += "<p>Invalid hour input, using 0.</p>"
    if not dep_min.isdigit():
        dep_min = 00
        out += "<p>Invalid minute input, using 0.</p>"
    dep_time = int(dep_hour) * 60 + int(dep_min)
    interpolated_times = get_travel_times_at(dep_time)
    best_road = min(interpolated_times, key=interpolated_times.get)
    est_travel_time = interpolated_times[best_road]

    out += """
    <p>
    Best route by nearest point linear interpolation: <br>
    Departure time: {}:{} <br> 
    Best travel route: {} <br> 
    Estimated travel time of {:.1f} minutes. </p> 
    <p><a href="/">Back</a></p>
    """.format(dep_hour, dep_min, best_road, est_travel_time)

    return out


@app.route('/')
def get_departure_time():
    return """
    	<h3>Knut Knut Transport AS</h3>
        <form action="/get_best_route" method="get">
            <label for="hour">Hour:</label>
            <select name="hour" id="hour">
                <option value="06">06</option>
                <option value="07">07</option>
                <option value="08">08</option>
                <option value="09">09</option>
                <option value="10">10</option>
                <option value="11">11</option>
                <option value="12">12</option>
                <option value="13">13</option>
                <option value="14">14</option>
                <option value="15">15</option>
                <option value="16">16</option>     
            </select>
            
            <label for="mins">Mins:</label>
            <input type="text" name="mins" size="2"/>
            <input type="submit">
        </form>
    """


@app.route("/get_best_route")
def get_route():
    departure_h = request.args.get('hour')
    departure_m = request.args.get('mins')

    better_route_info = get_best_route(departure_h, departure_m)
    return better_route_info


if __name__ == '__main__':
    print("<starting>")
    df = load_data('traffic.jsonl')
    df = preprocess_data(df)
    app.run()
    print("<done>")
