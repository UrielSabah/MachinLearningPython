from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """

    idx = permutation(len(data))
    data, labels = data[idx], labels[idx]

    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])

    maxi = int(train_ratio * max_count)

    if max_count:
        train_data = array(data[:maxi])
        train_labels = array(labels[:maxi])

        test_data = array(data[maxi:max_count])
        test_labels = array(labels[maxi:max_count])

    # print(train_data.shape[0])
    # print(train_labels.shape[0])
    # print(test_data.shape[0])
    # print(test_labels.shape[0])

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    tpr = 0.0
    fpr = 0.0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(labels.shape[0]):
        if labels[i] == 1:
            if prediction[i] == 1:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if prediction[i] == 1:
                fp = fp + 1
            else:
                tn = tn + 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    tpr = (tp / (tp + fn))
    fpr = (fp / (tn + fp))
    # print(tpr)
    # print(fpr)
    # print(accuracy)
    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []
    size = len(folds_array)

    for i in range(size):
        test_data = folds_array[i]
        test_labels = labels_array[i]

        X = folds_array[:i] + folds_array[i + 1:]
        y = labels_array[:i] + labels_array[i + 1:]
        # print(X)
        X = concatenate(X)
        y = concatenate(y)

        # print(y)
        # print(X)
        clf.fit(X, y)
        tp, fp, accu = get_stats(clf.predict(test_data), test_labels)

        tpr.append(tp)
        fpr.append(fp)
        accuracy.append(accu)
    # print(mean(tpr))
    # print(mean(fpr))
    # print(mean(accuracy))

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=(
                         {'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05},
                         {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params

    tpr = []
    fpr = []
    accuracy = []

    for i in range(len(kernels_list)):
        # idx = permutation(len(data_array))
        # folds_array, labels_array = data_array[idx], labels_array[idx]
        # print(data)
        # print(labels)
        #
        data = array_split(data_array, folds_count)
        labels = array_split(labels_array, folds_count)

        current_parameter = list(kernel_params[i].items()).pop()

        current_kernel = kernels_list[i]
        
        current_parameter_type = current_parameter[0]
        current_parameter_value = current_parameter[1]

        clf = None
        if current_parameter_type == 'degree':
            clf = SVC(degree=current_parameter_value, gamma=SVM_DEFAULT_GAMMA, kernel=current_kernel, C=SVM_DEFAULT_C)
        elif current_parameter_type == 'gamma':
            clf = SVC(degree=SVM_DEFAULT_DEGREE, gamma=current_parameter_value, kernel=current_kernel, C=SVM_DEFAULT_C)
        elif current_parameter_type == 'C':
            clf = SVC(degree=SVM_DEFAULT_DEGREE, gamma=SVM_DEFAULT_GAMMA, kernel=current_kernel,
                      C=current_parameter_value)

        mean_tpr, mean_fpr, mean_accu = get_k_fold_stats(folds_array=data, labels_array=labels, clf=clf)

        tpr.append(mean_tpr)
        fpr.append(mean_fpr)
        accuracy.append(mean_accu)

    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy

    return svm_df


def get_most_accurate_kernel(accu):
    """
    :return: integer representing the row number of the most accurate kernel
    """

    best_kernel, val = max(enumerate(accu), key=lambda x: x[1])

    return best_kernel


def get_kernel_with_highest_score(score):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel, val = max(enumerate(score), key=lambda x: x[1])

    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    highest_score_kernel = get_kernel_with_highest_score(df['score'])

    b = -(alpha_slope * x[highest_score_kernel]) + y[highest_score_kernel]
    equation = poly1d([alpha_slope, b])

    line = equation([0, 1])
    plt.plot([0, 1], line, color='blue')
    plt.scatter(x, y, alpha=1)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC curve (TPR vs FPR)')
    plt.show()


def evaluate_c_param(data_array, labels_array, folds_count, best_kernel_type):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param best_kernel_type: best kernel chosen from compare_svm
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """
    res = pd.DataFrame()

    kernels_list = []
    kernel_params = []
    map = {}
    rangei = [1, 0, -1, -2, -3, -4]
    rangej = [3, 2, 1]
    for i in rangei:
        for j in rangej:
            kernel_params.append(best_kernel_type)
            map['C'] = (10 ** i) * (j / 3)
            kernels_list.append(map)
            map = {}

    res = compare_svms(data_array, labels_array, folds_count, kernel_params, kernels_list)

    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel_type, best_kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    type = list(best_kernel_params.keys())[0]

    clf = None
    if type == 'degree':
        clf = SVC(degree=best_kernel_params.get(type), gamma=SVM_DEFAULT_GAMMA, kernel=best_kernel_type,
                  C=SVM_DEFAULT_C, class_weight='balanced')
    elif type == 'gamma':
        clf = SVC(degree=SVM_DEFAULT_DEGREE, gamma=best_kernel_params.get(type), kernel=best_kernel_type,
                  C=SVM_DEFAULT_C, class_weight='balanced')

    elif type == 'C':
        clf = SVC(degree=SVM_DEFAULT_DEGREE, gamma=SVM_DEFAULT_GAMMA, kernel=best_kernel_type,
                  C=best_kernel_params.get(type), class_weight='balanced')

    clf.fit(train_data, train_labels)
    tpr, fpr, accuracy = get_stats(clf.predict(test_data), test_labels)
    kernel_type = best_kernel_type
    kernel_params = best_kernel_params

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
