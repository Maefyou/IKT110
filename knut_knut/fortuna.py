# and reports the total exec time of the cell.

import numpy as np
import plotly.express as px
import tqdm
import multiprocessing as mp
from multiprocessing import Queue
import time
from tqdm import tqdm

NUMBER_OF_PROCESSES = 5
NUMBER_OF_MODELS = 20
TOTAL_MODELS = NUMBER_OF_PROCESSES * NUMBER_OF_MODELS  # Calculate total models


def predict(x, theta):
    # change to sum of 3 sin() terms.
    # use np.sin() and not math.sin().
    result = np.zeros_like(x)
    for i in range(0, 3*(len(theta)//3), 3):
        result += theta[i] * np.sin(theta[i+1] * (x + theta[i+2]))
    return result

    # Example hypothesis function h_1(x) with 3 parameters
    return theta[0] - (theta[1]*np.sin(theta[2]*x))*(theta[3]*(x+theta[4]))


def predict_h(x, theta):
    # Example hypothesis function h_2(x) with 5 parameters
    return theta[0] - (theta[1]*np.sin(theta[2]*x))*(theta[3]*(x+theta[4]))


def sample_theta(size_of_theta):
    # Do NOT CHANGE.
    theta = np.random.uniform(-4, 4, size=size_of_theta)
    return theta


def mutate_theta(theta, mutation_rate=0.001):
    # Mutate each parameter with a certain probability.
    theta += np.random.normal(0, mutation_rate, size=theta.shape)
    return theta


def get_loss(y_hat, ys):
    # No change needed, returns quadratic loss.
    loss = ((y_hat - ys)**2).sum()
    return loss


xs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
              4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4])
ys = np.array([4.03, 4.19, 4.26, 4.25, 4.17, 4.03, 3.85, 3.63, 3.40, 3.16, 2.93, 2.72, 2.53, 2.39, 2.28, 2.21, 2.18, 2.19, 2.22, 2.27, 2.33, 2.39, 2.44, 2.45, 2.43, 2.36, 2.22, 2.02, 1.75, 1.41, 1.00, 0.52, -0.01, -
              0.60, -1.22, -1.86, -2.50, -3.13, -3.72, -4.27, -4.75, -5.15, -5.45, -5.65, -5.74, -5.70, -5.55, -5.29, -4.92, -4.44, -3.89, -3.26, -2.58, -1.86, -1.12, -0.39, 0.32, 0.98, 1.60, 2.14, 2.61, 2.99, 3.28, 3.47, 3.57])

ys_h_1 = np.array([15.98, 21.42, 24.1, 23.87, 21.0, 16.11, 10.06, 3.79, -1.8, -6.01, -8.39, -8.82, -7.47, -4.77, -1.3, 2.31, 5.49, 7.81, 9.04, 9.18, 8.44, 7.15, 5.72, 4.54, 3.89, 3.9, 4.52, 5.54, 6.63, 7.39, 7.49, 6.7, 4.97,
                  2.44, -0.54, -3.48, -5.8, -6.98, -6.61, -4.51, -0.79, 4.2, 9.8, 15.21, 19.57, 22.09, 22.2, 19.63, 14.52, 7.41, -0.83, -9.07, -16.11, -20.83, -22.36, -20.27, -14.59, -5.91, 4.72, 15.9, 26.06, 33.69, 37.58, 36.94, 31.64])
ys_h_2 = np.array([[5.87, 5.83, 5.74, 5.62, 5.48, 16.58, 18.21, 18.49, 17.54, 15.54, 4.15, 3.9, 3.65, 3.42, 3.2, -1.9, -3.32, -3.97, -3.91, -3.27, 2.38, 2.36, 2.39, 2.46, 2.57, 2.2, 1.93, 1.22, 0.17, -1.06, 4.18, 4.59,
                  5.04, 5.52, 6.02, -1.49, 0.74, 3.55, 6.75, 10.06, 9.39, 9.95, 10.5, 11.02, 11.52, 16.0, 12.82, 8.47, 3.21, -2.62, 13.61, 13.76, 13.84, 13.84, 13.77, -24.16, -22.03, -17.9, -11.95, -4.55, 11.63, 10.99, 10.27, 9.47, 8.61]])


# change to the size of theta ( 9 ) (for h(x) how many parameters does it have?)
# params for h: 5
def train_model(xs, ys, result_queue):
    n_params = 9
    best_loss_local = float('inf')
    best_theta_local = sample_theta(n_params)
    alpha_start = 1
    alpha_end = 1e-12
    delta_alpha = (alpha_end-alpha_start) / 100000
    alpha = alpha_start

    for _ in range(100000):
        alpha += delta_alpha
        curr_theta = mutate_theta(best_theta_local.copy(), alpha)
        y_hat = predict_h(xs, curr_theta)
        curr_loss = get_loss(y_hat, ys)

        if best_loss_local > curr_loss:
            best_loss_local, best_theta_local = curr_loss, curr_theta
        
    result_queue.put((best_theta_local, best_loss_local))


def train_model_thread(xs, ys, result_queue, number_of_models):
    try:
        for _ in range(number_of_models):
            train_model(xs, ys, result_queue)
    except Exception as e:
        print(f"Error in process: {e}")
    print(f"Process {mp.current_process().pid} finished training.")


if __name__ == "__main__":

    # to get a solid estimate -> you should train at least 100 models and take the average performance.
    result_queue = Queue()
    processes = []

    for _ in range(NUMBER_OF_PROCESSES):
        p = mp.Process(target=train_model_thread, args=(
            xs, ys_h_1, result_queue, NUMBER_OF_MODELS))
        p.start()
        processes.append(p)
        print(f"Started process {p.pid}")

    results = []

    with tqdm(total=TOTAL_MODELS, desc="Collecting Results") as pbar:
        while len(results) < TOTAL_MODELS:
            if not result_queue.empty():
                results.append(result_queue.get())
                pbar.update(1)  # Increment progress bar
            # Small sleep to prevent busy waiting
            time.sleep(0.01)

    for p in processes:
        p.join()
    print("All processes joined.")

    # Calculate and print average loss
    avg_loss = sum(loss for _, loss in results) / len(results)
    print(f"Average loss over all models: {avg_loss}")

    # Find and print the best model
    best_theta, best_loss = min(results, key=lambda x: x[1])
    print(f"Best loss: {best_loss} with theta: {best_theta}")
    fig = px.line(x=xs, y=ys, title="f(x) vs Fortuna solution")
    fig.add_scatter(x=xs, y=predict_h(xs, best_theta),
                    mode='lines', name="y_hat")
    fig.update_layout(xaxis_range=[xs.min(), xs.max()], yaxis_range=[-6, 6])
    fig.show()