import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """

    extractedData = data[:, -1]
    count = np.array(np.unique(extractedData, return_counts=True)[1])
    count = count / extractedData.shape
    gini = 1 - np.sum(np.square(count))

    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """

    extractedData = data[:, -1]
    count = np.array(np.unique(extractedData, return_counts=True)[1])
    count = count / extractedData.shape

    entropy = -np.sum(count * np.log2(count))

    return entropy


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value, majorlabel, count):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.major = majorlabel  # the max label in this node
        self.children = []  # the childs of this node
        self.father = None  # the child father
        self.count = count  # the major count in this node

    def add_child(self, node):
        self.children.append(node)


def chisquare(data, bestfet, besttrashold, chi_value):
    # split tha data by feature and thrash hold
    left = data[data[:, bestfet] < besttrashold]
    right = data[data[:, bestfet] >= besttrashold]

    # calculate number of each class and all
    all_instance_in_node = data.shape[0]
    first_calss_instance = (data[:, -1] == 0).sum()
    second_calss_instance = (data[:, -1] == 1).sum()

    # calculate chi_square of less then trash and bigger then
    first = calc_chi(left, all_instance_in_node, first_calss_instance, second_calss_instance)
    second = calc_chi(right, all_instance_in_node, first_calss_instance, second_calss_instance)

    # final chi square value of this node
    chi_square_value = first + second

    return chi_square_value <= chi_table[chi_value]


def calc_chi(data, all_instance_in_node, first_calss_instance, second_calss_instance):
    df = data.shape[0]
    pf = (data[:, -1] == 0).sum()
    nf = (data[:, -1] == 1).sum()

    e_0 = df * first_calss_instance / all_instance_in_node
    e_1 = df * second_calss_instance / all_instance_in_node

    a = np.square(pf - e_0) / e_0
    b = np.square(nf - e_1) / e_1

    return a + b


def build_tree(data, impurity, chi_value):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    # if the impurity is 0 then its a leaf and return
    if impurity(data) == 0:
        countmajor = np.unique(data[:, -1], return_counts=True)
        i = np.argmax(countmajor[1])
        maj = countmajor[0][i]

        leaf = DecisionNode(None, None, maj, countmajor[1][0])
        return leaf
    # calculate the best feature, besrt thrash hold , majority and kabel
    bestfet, besttrashold, maj, i = calc_feat(data, impurity)

    # split the data for the new children's
    leftchild = data[data[:, bestfet] < besttrashold]
    rightchild = data[data[:, bestfet] >= besttrashold]

    # create this node
    root = DecisionNode(bestfet, besttrashold, maj, i)
    # if chi_value != 1 the calculate the chi_square to find out
    # if the next split is relevant
    if chi_value != 1:
        if chisquare(data, bestfet, besttrashold, chi_value):
            return root

    # build left and right child for this root
    leftchildroot = build_tree(leftchild, impurity, chi_value)
    rightchildroot = build_tree(rightchild, impurity, chi_value)

    # connect child to father
    leftchildroot.father = root
    rightchildroot.father = root

    # add child to the node list
    root.add_child(leftchildroot)
    root.add_child(rightchildroot)

    return root


def calc_feat(data, impurity):
    bestfet = -1
    minimpur = 100
    besttrashold = 0
    # iterate all features and find all the trashHold
    for fet in range(0, data.shape[1] - 1):
        col = data[:, fet]
        uniCol = np.unique(col)
        rollcol = np.roll(uniCol, (len(uniCol) - 1))
        trashHold = (rollcol + uniCol) / 2
        trashHold = trashHold[:-1]

        extractedData = data[:, [fet, -1]]
        # Split to 2 attributes
        for Tvalue in trashHold:
            left = extractedData[extractedData[:, 0] < Tvalue]
            right = extractedData[extractedData[:, 0] >= Tvalue]
            impur = impurity(left) * (left.shape[0] / data.shape[0]) + \
                    impurity(right) * (right.shape[0] / data.shape[0])

            # take the minimal impurity
            if minimpur > impur:
                minimpur = impur
                bestfet = fet
                besttrashold = Tvalue

    countmajor = np.array(np.unique(data[:, -1], return_counts=True))
    i = np.argmax(countmajor[1])
    maj = countmajor[0][i]

    return bestfet, besttrashold, maj, countmajor[1][0]


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    # if there is no children's then its a leaf and return
    # the majority,it is the prediction fot this instance
    if len(node.children) == 0:
        return node.major
    elif len(node.children) == 1 or instance[node.feature] < node.value:
        x = predict(node.children[0], instance)
    else:
        x = predict(node.children[1], instance)

    return x


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    # for each sample, if the predict equal the right label
    # then add 1 to tha accuracy, and at the end calculate
    # the percentage
    for data in dataset:
        if predict(node, data) == data[-1]:
            accuracy += 1

    accuracy = (accuracy / len(dataset)) * 100
    return accuracy


def print_tree(node, tab=0):
    """
    prints the tree according to the example in the notebook
    Input:
    - node: a node in the decision tree

    This function has no return value
    """
    # construct the number of tav for each 2 brothers
    s = "\t" * tab
    if len(node.children) == 0:
        print(s, end="")
        print("leaf: [{" + str(node.major) + ": " + str(node.count) + "}]")

    else:
        print(s, end="")
        print("[x" + str(node.feature) + " <= " + str(node.value) + "],")
        tab += 1
        print_tree(node.children[0], tab)
        print_tree(node.children[1], tab)


def get_leaf_nodes(root, leafs):
    # if their is no children then its a leaf
    if len(root.children) == 0:
        leafs.append(root)
        return leafs
    else:
        get_leaf_nodes(root.children[0], leafs)
        get_leaf_nodes(root.children[1], leafs)


def post_pruning(data, root):
    acc_list = []
    count_list = []
    count = 1
    # run until you get the root by it self
    while count > 0:
        accuracy = find_best_father(data, root)
        count = count_nodes(root)
        acc_list.append(accuracy)
        count_list.append(count)

    return acc_list, count_list


def find_best_father(data, root):
    # if their is no chldren then
    if len(root.children) == 0:
        return None
    # count leafs for this iteration
    leafs = []
    get_leaf_nodes(root, leafs)

    best_father = None
    accuracy = 0
    # loop on each leaf to find the best father
    for leaf in leafs:
        # make temp father, and save is children's for later,
        # then delete them
        check_best_father = leaf.father
        child = check_best_father.children
        check_best_father.children = []
        # calculate accuracy
        accuracy_now = calc_accuracy(root, data)
        # connect the children to theh father again,
        # because if is not the best then he need them
        check_best_father.children = child

        # if its the best accuracy until now, save it
        # and make the father as the best
        if accuracy < accuracy_now:
            accuracy = accuracy_now
            best_father = check_best_father
    # at the end make sure to delete the best fathers children's
    best_father.children = []

    return accuracy


def count_nodes(root):
    # if its leaf return 0 for not counting them
    if len(root.children) == 0:
        return 0

    return 1 + count_nodes(root.children[0]) + count_nodes(root.children[1])
