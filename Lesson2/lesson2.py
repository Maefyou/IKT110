import json
import random 
 
in_file = "dataset.jsonl"
xs = []
ys = []
 
with open(in_file, "r") as fp:
    for line in fp:
        d = json.loads(line)
        x = d["x"]
        y = d["y"]
        xs.append(x)
        ys.append(y)
 
print("some values:")
print(xs[0])
print(ys[0])
 
 
def my_model_function(theta, x):
    return theta[0]*x[0] + theta[1]*x[1] + theta[2]*x[2] + theta[3]

# L2 loss
def my_loss(y_hat, y):
    return  (y_hat - y) ** 3
 
def train_model(theta, loss, learning_rate) -> [float]:
    theta_new = theta.copy()
    theta[0] += learning_rate * loss
    theta[1] += learning_rate * loss
    theta[2] += learning_rate * loss
    theta[3] += learning_rate * loss
    return theta_new

best_loss = float('inf')
best_model = None
theta = [random.uniform(-1, 1) for _ in range(4)]
for guess in range(100):
 
    loss = 0.0
    for x, y in zip(xs, ys):
        y_hat = my_model_function(theta, x)
        loss += my_loss(y_hat, y)
        theta = train_model(theta, loss, learning_rate=0.0001)
 
    if loss < best_loss:
        best_loss = loss
        best_model = theta
        print("new best loss:", best_loss)
 
 
print("best loss:", best_loss)
print("best model:", best_model)