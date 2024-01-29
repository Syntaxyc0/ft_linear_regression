import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict(theta0, theta1, mileage):
    return theta0 + theta1 * mileage

def normalize_data(mileage, price, mileage_max, price_max):
    mileage = mileage / mileage_max
    price = price / price_max
    return mileage, price

def denormalize_thetas(theta0, theta1, mileage_max, price_max):
    denorm0 = theta0 * price_max
    denorm1 = theta1 * price_max / mileage_max
    return denorm0, denorm1

def update_thetas(mileage, price, theta0, theta1, learning_rate, sample_size):
    gradient0 = (learning_rate / sample_size) * np.sum(predict(theta0, theta1, mileage) - price)
    gradient1 = (learning_rate / sample_size) * np.sum((predict(theta0, theta1, mileage) - price) * mileage)
    return (theta0 - gradient0, theta1 - gradient1) 

def calculate_cost(mileage, price, theta0, theta1, sample_size):
    pred = predict(theta0, theta1, mileage)
    error = np.abs(pred - price)
    return np.sum(error) / sample_size


def main():
    data = pd.read_csv("data.csv")
    learning_rate = 0.1
    epochs = 5000
    mileage = data.iloc[:, 0]
    price = data.iloc[:, 1]
    sample_size = mileage.shape[0]
    mileage_max = mileage.max()
    price_max = price.max()
    mileage, price = normalize_data(mileage, price, mileage_max, price_max)
    theta0, theta1 = 0.0, 0.0
    costs = []
    with tqdm(total=epochs, desc="Training", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i in range (epochs):
            theta0, theta1 = update_thetas(mileage, price, theta0, theta1, learning_rate, sample_size)
            costs.append(calculate_cost(mileage, price, theta0, theta1, sample_size))
            pbar.update(1)
    theta0, theta1 = denormalize_thetas(theta0, theta1, mileage_max, price_max)
    np.save("thetas" ,np.array([theta0, theta1]))
    np.save("costs", costs)

if (__name__ == "__main__"):
    main()