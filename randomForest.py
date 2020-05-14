import numpy as np
import random
import pandas as pd
from scipy import stats

def determine_potential_splits(data, random_features):

    potential_splits = {}
    #find the number of columns
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))
    #if we are doing random forest we need to select a random subspace of features from our feature space
    if random_features and random_features <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_features)
    #loop through every possible value to split on and add it to our dict
    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values
    return potential_splits

def split_dataset(data, column_index, value):
    #find all possible values to split on
    split_column_values = data[:, column_index]
    #type of feature is stored in a global variable
    type_of_feature = FEATURE_TYPES[column_index]
    #split data based on the type of data it is
    if type_of_feature == "categorical":
        data_b = data[split_column_values == value]
        data_a = data[split_column_values != value]
    else:
        data_b = data[split_column_values <= value]
        data_a = data[split_column_values >  value]
    return data_a, data_b

def find_best_split_from_potential(data, potential_splits):
    #set total entropy to a large number initially
    total_entropy = 99999
    #loop through each possible split and then calculate the total entropy for each split
    #if the split has a lower entropy use that split
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_a, data_b = split_dataset(data, column_index, value)
            current_total_entropy = calc_total_entropy(data_a, data_b)

            if current_total_entropy <= total_entropy:
                total_entropy = current_total_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

def get_type_of_feature():
    #hard code the feature type
    feature_types = ["categorical", "categorical", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "continuous", "categorical"]

    return feature_types

def calc_node_entropy(data):
    #only look at the label column
    label_column = data[:, -1]
    #find the number of positives to calculate the entropy
    _, total = np.unique(label_column, return_counts=True)
    #calculate the entropy
    p = total / total.sum()
    entropy = sum(p * -np.log2(p))

    return entropy

def calc_total_entropy(data_a, data_b):
    total = len(data_a) + len(data_b)
    p_data_b = len(data_b)/total
    p_data_a = len(data_a)/total

    total_entropy = (p_data_b * calc_node_entropy(data_b)) + (p_data_a * calc_node_entropy(data_a))
    return total_entropy

def find_node_purity(data):
    #label column is in the last column of the dataset
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    #if the only value in the column is 1 or 0 it's pure
    if len(unique_classes) != 1:
        return False
    else:
        return True

def node_classification(data):
    #label column is in the last column of the dataset
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    classification = unique_classes[counts_unique_classes.argmax()]

    return classification

def decision_tree_algorithm(data, counter=0, min_samples_per_node=12, max_depth=5, random_subspace_of_features=None):
    if counter == 0:
        global FEATURE_NAMES, FEATURE_TYPES
        FEATURE_TYPES = get_type_of_feature()
        #use the index as a feature name to easily make predictions from the return tree
        FEATURE_NAMES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#         ["quality assessment", "pre-screening", "MA1", "MA2", "MA3", "MA4", "MA5", "MA6", "Exudates1", "Exudates2", "Exudates3", "Exudates4", "Exudates5", "Exudates6", "Exudates7", "Exudates8", "Distance", "Diameter", "AM/FM Classification"]

    #last case to consider. If data is pure or if we are at the max depth or if the amount of data is smaller than the min samples
    if (counter == max_depth) or (len(data) < min_samples_per_node) or (find_node_purity(data)):
        classification = node_classification(data)

        return classification
    else:
        counter += 1
        #call all functions to prepare for splitting
        potential_splits = determine_potential_splits(data, random_subspace_of_features)
        split_column, split_value = find_best_split_from_potential(data, potential_splits)
        data_a, data_b = split_dataset(data, split_column, split_value)

        if len(data_b) == 0 or len(data_a) == 0:
            classification = node_classification(data)
            return classification
        #find the feature index and type of feature in the global lists
        feature_name = FEATURE_NAMES[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        #set the "question" about the data based on the type of data it is
        if type_of_feature == "continuous":
            hypothesis = "{} <= {}".format(feature_name, split_value)

        else:
            hypothesis = "{} = {}".format(feature_name, split_value)
        #create this subtree and add it to the larger tree
        sub_tree = {hypothesis: []}

        true_answer = decision_tree_algorithm(data_b, counter, min_samples_per_node, max_depth, random_subspace_of_features)
        false_answer = decision_tree_algorithm(data_a, counter, min_samples_per_node, max_depth, random_subspace_of_features)

        if true_answer != false_answer:
            sub_tree[hypothesis].append(true_answer)
            sub_tree[hypothesis].append(false_answer)
        else:
            sub_tree = true_answer

        return sub_tree

def predict_example(example, tree):
    hypothesis = list(tree.keys())[0]
    feature_name, comparison_operator, value = hypothesis.split(" ")
    # ask question the question and move down the tree
    if comparison_operator == "=":
        if str(example[int(feature_name)]) == value:
            answer = tree[hypothesis][0]
        else:
            answer = tree[hypothesis][1]
    else:
        if example[int(feature_name)] <= float(value):
            answer = tree[hypothesis][0]
        else:
            answer = tree[hypothesis][1]

    # Base case if you reach the bottom of the tree
    if not isinstance(answer, dict):
        return answer

    # Recursively go down the tree
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

def decision_tree_predictions(testing_data, tree):
    #apply the prediction to each row in the data set
    prediction = np.apply_along_axis(predict_example, 1, testing_data, tree)
    return prediction

#Random Forest Helpers:
def bootstrapping(train_data, num_of_bs):
    #randomly select rows with repetition to use as data sets for Random Forest
    bs_indices = np.random.randint(low=0, high=len(train_data), size=num_of_bs)
    data_bs = train_data[bs_indices]

    return data_bs

def random_forest_algorithm(train_data, n_trees, n_bootstrap, n_features, n_max_depth):
    forest = []
    for i in range(n_trees):
        #create multiple trees to add to our forest
        bootstrapped_data = bootstrapping(train_data, n_bootstrap)
        tree = decision_tree_algorithm(bootstrapped_data, max_depth=n_max_depth, random_subspace_of_features=n_features)
        forest.append(tree)

    return forest

def random_forest_predictions(test_data, forest):
    data_predictions = []
    #store all possible predictions in a dict
    for i in range(len(forest)):
        predictions = decision_tree_predictions(test_data, tree=forest[i])
        data_predictions.append(predictions)

    preds = np.array(data_predictions)
    random_forest_predictions = stats.mode(preds, axis=0)[0][0]
    return random_forest_predictions


def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[List[float]]
        training_label: List[int]
        testing_data: List[List[float]]

    Output:
        testing_prediction: List[int]
    Example:
    return [1]*len(testing_data)
    """
    #transform the lists into numpy arrays
    training_data_array = np.array(training_data)
    training_label_array = np.array(training_labels)
    test_array = np.array(testing_data)
    #add the label column to easily calculate purity and pass it into the random forest algorithm
    data = np.column_stack((training_data_array, training_label_array))
    forest = random_forest_algorithm(data, n_trees=23, n_bootstrap=100, n_features=9, n_max_depth=4)
    prediction = random_forest_predictions(test_array, forest)
    return prediction
