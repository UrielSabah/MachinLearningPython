import numpy as np
import itertools

np.random.seed(42)


def preprocess(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column. 

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns a two vales:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """

    ###########################################################################
    ###########################################################################
    meanX = X.mean(axis=0)
    maxX = X.max(axis=0)
    minX = X.min(axis=0)
    X = (X - meanX) / (maxX - minX)
    y = (y - np.average(y)) / (max(y) - min(y))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.  

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # Use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################

    J = np.square((X.dot(theta)) - y)
    J = J.sum(axis=0) / (2 * X.shape[0])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by 
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = []  # Use a python list to save cost in every iteration
    ###########################################################################
    ###########################################################################
    for i in range(num_iters):
        theta = theta - ((alpha / len(X)) * ((X.dot(theta) - y).dot(X)))
        J_history.append(compute_cost(X, y, theta))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """

    pinv_theta = None
    ###########################################################################
    ###########################################################################
    XT = np.transpose(X)
    pinv_theta = (np.linalg.inv(XT.dot(X)).dot(XT)).dot(y)  #

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = []  # Use a python list to save cost in every iteration
    ###########################################################################
    ###########################################################################
    J_history = []
    for i in range(num_iters):
        theta = theta - ((alpha / X.shape[0]) * ((X.dot(theta) - y).dot(X)))
        J_history.append(compute_cost(X, y, theta))
        num_iters = num_iters - 1
        if i > 0:
            if abs(J_history[i - 1] - J_history[i]) < 1e-8:
                break
    #  if J_history <
    return theta, J_history
    ####################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary 
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training 
    for every value of alpha.
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(42)
    theta = np.random.random(size=2)
    for i in range(len(alphas)):
        J_history = efficient_gradient_descent(X, y, theta, alphas[i], iterations)[1]
        alpha_dict[alphas[i]] = J_history[len(J_history) - 1]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """
    triplets = list(itertools.combinations(X.columns, 3))

    return triplets


def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best 
    minimizes the cost function. For better performance, you should use the 
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    best_triplet = None
    np.random.seed(42)
    theta = np.random.random(size=4)
    minimum = 1234567
    y = np.array(df["price"])  # take the target value
    
    for i in range(len(triplets)):  # iterate over all possible triplets
        X = np.array(df[list(triplets[i])])  # take each 3 culomns
        X, y = preprocess(X, y)  # preprocess the new matrix
        X = np.column_stack((np.ones((X.shape[0], 1)), X))  # bias trick
        J = efficient_gradient_descent(X, y, theta, alpha, num_iter)[1]
        if (J[len(J) - 1] < minimum):
            minimum = J[len(J) - 1]
            best_triplet = list(triplets[i])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_triplet
