import pandas as pd
import json


def build_worker_dataframe(worker_file_path='', schedule_file_path=''):
    """ Build a DataFrame containing worker salary analysis based on shifts worked.
    Args:
        worker_file_path (str): Path to the workers.jsonl file.
        schedule_file_path (str): Path to the schedules.json file.
    Returns:
        pd.DataFrame: DataFrame with worker salary analysis.
    """

    #import workers.jsonl file
    workers_df = pd.read_json(worker_file_path, lines=True)
    #rename columns
    workers_df.rename(columns={'salary': 'full_week_salary'}, inplace=True)

    # read schedules.json file
    with open(schedule_file_path, 'r') as f:
        schedule_data = json.load(f)
    
    worked_shifts = pd.DataFrame()
    for day, shifts in schedule_data.items():
        for shift in shifts:
            worker_id = shift.get('worker_id', '')
            shift_worked = shift.get('shift', 0)
            new_row = {"day": day,
                       "worker_id": worker_id,
                       "shifts_worked": shift_worked}
            worked_shifts = pd.concat([worked_shifts, pd.DataFrame([new_row])], ignore_index=True)
    
    # sum up shifts worked per worker
    worked_shifts = worked_shifts.groupby('worker_id', as_index=False).agg({
        'shifts_worked': 'sum',
    })

    # merge workers_df and worked_shifts on worker_id
    final_df = workers_df.merge(worked_shifts, left_on='worker_id', right_on='worker_id', how='left')
    final_df['shifts_worked'] = final_df['shifts_worked'].fillna(0)

    # calculate salary per shift
    final_df['salary_per_shift'] = final_df['full_week_salary'] / 14

    # calculate total salary based on shifts worked
    final_df['total_salary'] = final_df['salary_per_shift'] * final_df['shifts_worked']

    print(final_df)
    return final_df


def build_worker_dataframe_csv(worker_file_path='', schedule_file_path=''):
    """ Build a DataFrame containing worker salary analysis based on shifts worked.
    Args:
        worker_file_path (str): Path to the workers.jsonl file.
        schedule_file_path (str): Path to the schedules.csv file.
    Returns:
        pd.DataFrame: DataFrame with worker salary analysis."""
     #import workers.jsonl file
    workers_df = pd.read_json(worker_file_path, lines=True)
    #rename columns
    workers_df.rename(columns={'salary': 'full_week_salary'}, inplace=True)

    # read schedules from csv file
    schedule_df = pd.read_csv(schedule_file_path)
    schedule_df.drop(columns=['day', 'department'], inplace=True, errors='ignore')
    schedule_df = schedule_df.groupby(['worker_id'], as_index=False).agg({'shift': 'sum'})

    # merge workers_df and schedule_df on worker_id
    final_df = workers_df.merge(schedule_df, left_on='worker_id', right_on='worker_id', how='left')
    final_df['shift'] = final_df['shift'].fillna(0)

    # calculate salary per shift
    final_df['salary_per_shift'] = final_df['full_week_salary'] / 14

    # calculate total salary based on shifts worked
    final_df['total_salary'] = final_df['salary_per_shift'] * final_df['shift']

    print(final_df)
    return final_df


def build_jsonl_file(df, output_file_path=''):
    """ Build a JSONL file containing worker salary analysis.
    Args:
        df (pd.DataFrame): DataFrame with worker salary analysis.
        output_file_path (str): Path to the output JSONL file.
    Returns:
        None
    """

    with open(output_file_path, 'w') as f:
        for line in df.itertuples():
            record = {
                "name": line.name,
                "worker_id": line.worker_id,
                "age": line.age,
                "salary": line.total_salary
            }
            f.write(json.dumps(record) + '\n')

df = pd.DataFrame()
df = build_worker_dataframe_csv('data/workers/workers.jsonl', 'data/schedules/Proposed Schedule.csv')
build_jsonl_file(df, 'analysis/proposed_workers_information.jsonl')