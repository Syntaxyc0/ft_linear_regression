import numpy as np

def estimatePrice(theta0 = 0, theta1 = 0, mileage = 0):
    return (theta0 + (theta1 * mileage))

def get_thetas() :
    try:
        thetas = np.load("thetas.npy")
        theta0, theta1 = thetas[0], thetas[1]
    except:
        theta0, theta1 = (0,0)
        print("The model isn't trained yet, you should use the training function, thetas initialized to (0,0)")
    return (theta0, theta1)

def main() :
    mileage = input("Mileage: ")
    try:
        mileage = int(mileage)
    except:
        print("Wrong format input")
        exit(1)
    theta0, theta1 = get_thetas()
    price = estimatePrice(theta0, theta1, mileage)
    if (price < 0):
        print("With this mileage, the predictions are that this car isn't worth anything, the estimated value is less than 0")
        exit(0)
    print("Estimated price for", mileage, "km is {:.0f}$".format(price))

if (__name__ == "__main__"):
    main()