import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from estimate import get_thetas

def get_costs() :
    try:
        costs = np.load("costs.npy")
    except:
        costs = np.array(1)
    return costs

def main():
    data = pd.read_csv("data.csv")
    fig, ax = plt.subplots(1,3)
    theta0, theta1 = get_thetas()
    costs = get_costs()

    ax[0].scatter(data=data, x="km", y="price")
    ax[0].set_title("Raw Dataset")
    ax[0].set_xlabel("Mileage (km)")
    ax[0].set_ylabel("Price")
    ax[1].scatter(data=data, x="km", y="price")
    ax[1].set_title("linear regression")
    ax[1].set_xlabel("Mileage (km)")
    ax[1].set_ylabel("Price")
    x = data["km"]
    y = theta0 + theta1 * x
    ax[1].plot(x, y, 'r')
    ax[2].set_title("Error")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Error")
    ax[2].plot(costs)
    plt.show()










if (__name__ == "__main__"):
    main()