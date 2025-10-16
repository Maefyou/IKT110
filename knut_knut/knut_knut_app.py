"""
Knut Knut Transport AS Flask App
Copilot was used to help writing some functions.
"""

from flask import Flask
from flask import request
from flask import send_from_directory
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
df = None  # Global variable to hold the DataFrame
loaded_params = None  # Global variable to hold loaded parameters


def load_data(file_path):
    """Load data from jsonl file into a pandas DataFrame."""
    df = pd.read_json(file_path, lines=True)
    return df


def plot_travel_times(df):
    """Plot travel times for each road and save the plot as an image."""
    df['travel_time'] = pd.to_datetime(
        df['arrival']) - pd.to_datetime(df['depature'])
    for road, group in df.groupby('road'):
        group = group.sort_values(by='depature')
        times = (group['travel_time'].dt.total_seconds() / 60).astype(float)
        depatures = pd.to_datetime(
            group['depature']).dt.hour * 60 + pd.to_datetime(group['depature']).dt.minute
        plt.figure()
        plt.scatter(depatures, times)
        plt.title(f'Travel Times for {road}')
        plt.xlabel('Departure Time')
        plt.ylabel('Travel Time')
        plt.tight_layout()
        road = road.replace("-", "").replace(">", "")
        plt.savefig(f'{road}_travel_times.png')


def predict_ACD(dep_datetime, params):
    """
    0: ACD 5 params: x**2 like
    all params should be normalized
    """
    dep_mins = dep_datetime.hour * 60 + dep_datetime.minute
    traveltime_ACD = params[0] *(dep_mins-params[1])**2
    traveltime_ACD += params[2]
    return traveltime_ACD


def predict_ACE(dep_datetime, df):
    """
    1: ACE no params needed
    """
    df = df[df['road'] == 'A->C->E']
    df = df.sort_values(by='depature')
    #closest_index = (df['values'] - target_value).abs().idxmin()
    dep_mins = dep_datetime.hour * 60 + dep_datetime.minute
    df['dep_mins'] = pd.to_datetime(df['depature']).dt.hour * 60 + pd.to_datetime(df['depature']).dt.minute
    closest_index = (df['dep_mins'] - dep_mins).abs().idxmin()
    traveltime_ACE = (pd.to_datetime(df.loc[closest_index, 'arrival']) - pd.to_datetime(df.loc[closest_index, 'depature'])).total_seconds() / 60
    print(traveltime_ACE)
    return traveltime_ACE


def predict_BCE(dep_datetime, params):
    """
    3: BCE sägezahn: 4 params
    all params should be normalized
    """
    dep_mins = dep_datetime.hour * 60 + dep_datetime.minute
    traveltime_BCE = params[0] *((dep_mins-params[1]) % params[2]) + params[3]
    return traveltime_BCE


def predict_BCD(dep_datetime, params):
    """
    2: BCD parabola + sägezahn: 7 params
    all params should be normalized
    """
    dep_mins = dep_datetime.hour * 60 + dep_datetime.minute
    traveltime_BCD = params[0]*(dep_mins-params[1])**2
    traveltime_BCD += params[2]
    traveltime_BCD += params[3]*((dep_mins-params[4]) % params[5]) + params[6]
    return traveltime_BCD


def train_model(df):
    """Train models using fortuna for each road and return parameters."""
    best_params = {key: None for key in df['road'].unique()}
    #observation based seeding
    best_params['A->C->E'] = None  # No params to train for ACE
    best_params['A->C->D'] = [0.1,700.,80.]  # 3 params for ACD
    best_params['B->C->D'] = [0.001,700.,70.,-1,60.,80., 0.]  # 7 params for BCD
    best_params['B->C->E'] = [-0.7,50.,100.,70.]  # 4 params for BCE
    for road, group in df.groupby('road'):
        if road == 'A->C->E':
            continue  # No params to train for ACE
        params = best_params[road].copy()
        best_loss = float('inf')
        group = group.sort_values(by='depature')
        epochs = 1000
        for epoch in range(epochs):
            params = best_params[road] * (1 + np.random.normal(0, 0.05,len(params)))
            loss = 0
            for _, row in group.iterrows():
                match road:
                    case 'A->C->D':
                        y_hat = predict_ACD(pd.to_datetime(row['depature']), params)
                    case 'B->C->D':
                        y_hat = predict_BCD(pd.to_datetime(row['depature']), params)
                    case 'B->C->E':
                        y_hat = predict_BCE(pd.to_datetime(row['depature']), params)
                y = (pd.to_datetime(row['arrival']) - pd.to_datetime(row['depature'])).total_seconds() / 60
                loss += (y - y_hat) ** 2
            loss /= len(group)
            if loss < best_loss:
                best_loss = loss
                best_params[road] = params
            if epoch % 10 == 0:
                print(f"Road: {road}, Epoch: {epoch}, Loss: {best_loss:.4f}")
            
    #save best params to a file
    with open('best_params.txt', 'w') as f:
        for road, params in best_params.items():
            f.write(f"{road}: {params}\n")


def plot_predictions_vs_actual(df, best_params):
    """Plot predictions vs actual travel times for each road."""
    for road, group in df.groupby('road'):
        group = group.sort_values(by='depature')
        depatures = pd.to_datetime(group['depature']).dt.hour * 60 + pd.to_datetime(group['depature']).dt.minute
        actual_times = (pd.to_datetime(group['arrival']) - pd.to_datetime(group['depature'])).dt.total_seconds() / 60
        predicted_times = []
        for _, row in group.iterrows():
            match road:
                case 'A->C->D':
                    y_hat = predict_ACD(pd.to_datetime(row['depature']), best_params[road])
                case 'A->C->E':
                    y_hat = predict_ACE(pd.to_datetime(row['depature']), df)
                case 'B->C->D':
                    y_hat = predict_BCD(pd.to_datetime(row['depature']), best_params[road])
                case 'B->C->E':
                    y_hat = predict_BCE(pd.to_datetime(row['depature']), best_params[road])
            predicted_times.append(y_hat)
        print(predicted_times[:5])
        print(actual_times[:5])
        plt.figure()
        plt.scatter(depatures, actual_times, label='Actual', color='blue')
        plt.scatter(depatures, predicted_times, label='Predicted', color='red', alpha=0.5)
        plt.title(f'Predictions vs Actual for {road}')
        plt.xlabel('Departure Time (mins)')
        plt.ylabel('Travel Time (mins)')
        plt.legend()
        plt.tight_layout()
        road_clean = road.replace("-", "").replace(">", "")
        plt.savefig(f'{road_clean}_predictions_vs_actual.png')


def load_best_params(file_path='best_params.txt'):
    """Load best parameters from a file."""
    best_params = {}
    with open(file_path, 'r') as f:
        for line in f:
            road, params_str = line.strip().split(': ')
            if params_str == 'None':
                best_params[road] = None
            else:
                params = list(map(float, params_str.strip('[]').split(' ')))
                best_params[road] = params
    return best_params


def get_best_route(dep_hour=0, dep_min=0):
    """Get the best route and estimated travel time for a given departure time."""
    dep_datetime = dt.datetime(2023, 1, 1, int(dep_hour), int(dep_min))
    routes = {
        'A->C->D': predict_ACD(dep_datetime, loaded_params['A->C->D']),
        'A->C->E': predict_ACE(dep_datetime, df),
        'B->C->D': predict_BCD(dep_datetime, loaded_params['B->C->D']),
        'B->C->E': predict_BCE(dep_datetime, loaded_params['B->C->E']),
    }
    best_road = min(routes, key=routes.get)
    est_travel_time = routes[best_road]

    return best_road, est_travel_time



def get_best_route_text(dep_hour=0, dep_min=0):
    """Get the best route and estimated travel time for a given departure time."""
    out = ""
    if not dep_hour.isdigit():
        dep_hour = 6
        out += "<p>Hour input must be a digit. Defaulting to 6.</p>"
    if not dep_min.isdigit():
        dep_min = 00
        out += "<p>Minute input must be a digit. Defaulting to 0.</p>"
    match int(dep_hour):
        case h if 6 <= h <= 16:
            pass
        case h if h > 16:
            dep_hour = 16
            out += "<p>Hour too late, using 16.</p>"
        case _:
            dep_hour = 0
            out += "<p>Hour too early, using 6.</p>"
    match int(dep_min):
        case m if 0 <= m <= 59:
            pass
        case m if m > 59:
            dep_min = 59
            out += "<p>Minute too high, using 59.</p>"
        case _:
            dep_min = 00
            out += "<p>Invalid minute input, using 0.</p>"
    
    best_road, est_travel_time = get_best_route(dep_hour, dep_min)

    # Calculate best route for 10 minutes from now
    dep_datetime = dt.datetime(2023, 1, 1, int(dep_hour), int(dep_min))
    dep_datetime_plus10 = dep_datetime + dt.timedelta(minutes=10)
    # Check if the new time is within valid range (6-16 hours)
    if dep_datetime_plus10.hour <= 16:
        best_road_plus10, est_travel_time_plus10 = get_best_route(dep_datetime_plus10.hour, dep_datetime_plus10.minute)
        
        plus10_info = """
        <p>
        Best route in 10 minutes: <br>
        Departure time: {}:{:02d} <br> 
        Best travel route: {} <br> 
        Estimated travel time of {:.1f} minutes. </p>
        """.format(dep_datetime_plus10.hour, dep_datetime_plus10.minute, best_road_plus10, est_travel_time_plus10)
    else:
        plus10_info = "<p>No route available 10 minutes later (outside operating hours 6-16).</p>"

    out += """
    <p>
    Best route: <br>
    Departure time: {}:{} <br> 
    Best travel route: {} <br> 
    Estimated travel time of {:.1f} minutes. </p> 
    {}
    <p><a href="/">Back</a></p>
    """.format(dep_hour, dep_min, best_road, est_travel_time, plus10_info)

    return out


def avg_travel_time_vs_best():
    traveltimes = pd.DataFrame(columns=['dep_time', 'A->C->D', 'A->C->E', 'B->C->D', 'B->C->E', 'mean','min'])
    for hour in range(6, 17):
        for minute in range(0, 60, 5):
            dep_datetime = dt.datetime(2023, 1, 1, hour, minute)
            row = {'dep_time': hour*60 + minute}
            row['A->C->D'] = predict_ACD(dep_datetime, loaded_params['A->C->D'])
            row['A->C->E'] = predict_ACE(dep_datetime, df)
            row['B->C->D'] = predict_BCD(dep_datetime, loaded_params['B->C->D'])
            row['B->C->E'] = predict_BCE(dep_datetime, loaded_params['B->C->E'])
            row['mean'] = np.mean([row['A->C->D'], row['A->C->E'], row['B->C->D'], row['B->C->E']])
            row['min'] = np.min([row['A->C->D'], row['A->C->E'], row['B->C->D'], row['B->C->E']])
            traveltimes = pd.concat([traveltimes, pd.DataFrame([row])], ignore_index=True)
    plt.figure()
    plt.plot(traveltimes['dep_time'], traveltimes['mean'], label='Mean Travel Time', color='blue')
    plt.plot(traveltimes['dep_time'], traveltimes['min'], label='Best Travel Time', color='red')
    plt.title('Mean vs Best Travel Time')
    plt.xlabel('Departure Time (mins)')
    plt.ylabel('Travel Time (mins)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean_vs_best_travel_time.png')



    overall_avg_avg = traveltimes['mean'].mean()
    overall_best_avg = traveltimes['min'].mean()
    improvement = (overall_avg_avg - overall_best_avg) / overall_avg_avg * 100

    #write stats to a file
    with open('travel_time_stats.txt', 'w') as f:
        f.write(f"Overall average travel time: {overall_avg_avg:.2f} mins\n")
        f.write(f"Overall average best travel time: {overall_best_avg:.2f} mins\n")
        f.write(f"Improvement: {improvement:.2f}%\n")


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

    better_route_info = get_best_route_text(departure_h, departure_m)
    return better_route_info


if __name__ == '__main__':
    print("<starting>")
    loaded_params = load_best_params('best_params.txt')
    df = load_data('traffic.jsonl')
    #train_model(df)
    app.run()
    #avg_travel_time_vs_best()
    print("<done>")