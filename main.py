import numpy as np
import sklearn
import helpers


# 5.1.3) Accuracy Table
def get_accuracy_table_and_models():
    """
    param train_data: train data
    param test_data: test data
    return: table of accuracies for different values of k and distance metric.
    """
    # create a np arry 5*2
    accuracy_table = np.zeros((5, 2))
    KNN_models_list = []
    for k, row in zip([1, 10, 100, 1000, 3000], range(5)):
        for distance_metric, col in zip(['l1', 'l2'], range(2)):
            accuracy_table[row, col], knn_model = helpers.knn_examples(train_data[:, 0:2], train_data[:, 2],
                                                                       test_data[:, 0:2],
                                                                       test_data[:, 2], k,
                                                                       distance_metric)
            KNN_models_list.append(knn_model)

    col_lable = ['l1', 'l2']
    row_lable = ['k=1', 'k=10', 'k=100', 'k=1000', 'k=3000']
    helpers.plt.table(cellText=accuracy_table, rowLabels=row_lable, colLabels=col_lable, loc='center')
    helpers.plt.axis('off')
    helpers.plt.show()

    print(accuracy_table)
    return accuracy_table, KNN_models_list





def knn_demo():
    """"""


    table, KNN_models = get_accuracy_table_and_models(train_data, test_data)

    """"
    Choose the k value with the lowest test accuracy when using the L2
    distance metric. We will call it kmin.
    • Using the given visualization helper (plot decision boundaries),
    plot the test data and color the space according to the prediction of
    the following 3 models, each in a separate plot (overall 3 plots): (i)
    distance metric = L2, k = kmax. (ii) distance metric = L2, k = kmin.
    (iii) distance metric = L1, k = kmax."""
    # 5.2.2) Decision Boundaries
    kmin_l2 = np.argmin(table[:, 1])
    kmax_l2 = np.argmax(table[:, 1])
    kmaks_l1 = np.argmax(table[:, 0])
    helpers.plot_decision_boundaries(KNN_models[kmaks_l1], test_data[:, 0:2], test_data[:, 2], "KNN L1 kmax")
    helpers.plot_decision_boundaries(KNN_models[kmin_l2 + 5], test_data[:, 0:2], test_data[:, 2], "KNN L2 kmin")
    helpers.plot_decision_boundaries(KNN_models[kmax_l2 + 5], test_data[:, 0:2], test_data[:, 2], "KNN L2 kmax")



# 5.3 Anomaly Detection Using kNN

"""
In this phase, our exploration turns towards anomaly detection, utilizing the
kNN algorithm to identify anomalies. For this specific task, we deviate from
supervised classification. Instead, we treat the training set as a single class,
ignoring individual class labels. The primary objective is to identify anomalies
within the test set by calculating kNN distances.
More specifically:
• You were given an additional test file specifically for this task, named:
AD test.csv. Your train set remains unchanged.
• Find the 5 nearest neighbors from the train set for each test sample of
AD test.csv using faiss. Use the L2 distance metric. Save the distances
to the neighbors as well.
• Sum the 5 distances to the nearest neighbors for each test sample. We
will refer to these summations as the anomaly scores.
• Find the 50 test examples with the highest anomaly scores. We will define
these points as anomalies, while the rest of the points will be defined as
normal.
• Using the matplotlib library, plot the points of AD test.csv. According
to your prediction, color the normal points in blue, and the anomalous
points in red. Additionally, in the same plot, include the data points
from train.csv colored in black and with an opacity of 0.01 (the parameter
controlling the opacity is named alpha). You might find the plt.scatter
function useful for this visualization.
"""


def AD_test_data():
    """
    Returns a table of accuracies for different values of k and distance metric."""

    AD_test_data, col_name_AD_test = helpers.read_data_demo("AD_test.csv")
    AD_test_data = AD_test_data[:, 0:2]
    print(f'AD_test_data: {AD_test_data}')
    print(f'sub_ADDAta: {AD_test_data[[0, 3, 4]]}')

    knn1 = helpers.KNNClassifier(k=5, distance_metric='l1')
    knn1.fit(train_data[:, 0:2], train_data[:, 2])
    distance, index = knn1.knn_distance(AD_test_data)
    print(f'distance: {distance}')
    print(f'index: {index}')

    anomaly_scores = np.sum(distance, axis=1)
    print(f'anomaly_scores: {anomaly_scores}')
    indices_of_largest_anomaly_score = anomaly_scores.argsort()[-50:][::-1]
    print(f'indices_of_largest: {indices_of_largest_anomaly_score}')

    anomaly_points = AD_test_data[indices_of_largest_anomaly_score]
    normal_points = np.delete(AD_test_data, indices_of_largest_anomaly_score, axis=0)

    helpers.plt.scatter(train_data[:, 0], train_data[:, 1], label='train', c='black', alpha=0.01)
    helpers.plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], label='anomaly', c='red', alpha=0.5)
    helpers.plt.scatter(normal_points[:, 0], normal_points[:, 1], label='normal', c='blue', alpha=0.5)
    # add title to each axis longitude vs latitude
    legend = helpers.plt.legend()

    # Set alpha values for legend handles
    for handle in legend.legend_handles:
        handle.set_alpha(1)

    helpers.plt.xlabel('longitude')
    helpers.plt.ylabel('latitude')
    helpers.plt.title('Anomaly Detection')

    helpers.plt.show()


"""
\Train 24 decision trees using the provided training data, classifying the long.-lat.
points into states. The 24 trees are all combinations for the following values for
these two hyper-parameters:
• Maximal depth (named max depth): (1, 2, 4, 6, 10, 20, 50, 100)
• Maximal leaf nodes(named max leaf nodes): (50, 100, 1000)
You should save all 24 tree models. For each tree you should also save its
hyper-parameters (max depth, max leaf nodes) and all 3 accuracies (training,
validation and test). We will later choose between the trees, according to these
values.    
"""

# 6.1) Decision Trees
def decision_tree_demo():
    """
    Returns a table of accuracies for different values of k and distance metric.
    """
    max_depth = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes = [50, 100, 1000]
    accuracy_table = np.zeros((8, 3, 3))
    decision_tree_models = np.zeros((8, 3), dtype=object)
    for depth, row in zip(max_depth, range(8)):
        for leaf, col in zip(max_leaf_nodes, range(3)):
            tree_classifier, train_val_test_accuracy = helpers.decision_tree_demo_v2(depth, leaf, train_data,
                                                                                     validation_data,
                                                                                     test_data)
            accuracy_table[row, col] = train_val_test_accuracy
            decision_tree_models[row, col] = tree_classifier

    column_lable = ['50', '100', '1000']
    row_lable = ['1', '2', '4', '6', '10', '20', '50', '100']
    helpers.plt.table(cellText=accuracy_table, rowLabels=row_lable, colLabels=column_lable, loc='center')
    helpers.plt.axis('off')
    helpers.plt.show()


    # calculate the best accuracy on validation data

    argmax_index = np.argmax(accuracy_table[:, :, 1])

    # Convert the flattened index to 2D index
    num_rows, num_cols = accuracy_table[:, :, 1].shape
    row_index = argmax_index // num_cols
    col_index = argmax_index % num_cols

    best_tree_model = decision_tree_models[row_index, col_index]
    helpers.plot_decision_boundaries(best_tree_model, test_data[:, 0:2], test_data[:, 2],
                                     f"Decision Tree best model, max_depth={row_index}, max_leaf_nodes={col_index}")
    helpers.plt.show()



"""
Random Forest. Random forest are a simple extension to decision trees.
Specifically, random forests train many small decision tree models on 
different subsets of the data and take their majority vote as the prediction.
To our convenience, this is also implemented in the sklearn library. In
the helpers file you can find an example for loading a random forest model,
and its interface is exactly the same as the decision trees.
"""
# 6.7)
def random_forest():
    """

    Returns:

    """
    random_forest = helpers.loading_random_forest()
    random_forest.fit(train_data[:, 0:2], train_data[:, 2])
    random_forest_prediction = random_forest.predict(test_data[:, 0:2])
    accuracy = np.mean(random_forest_prediction == test_data[:, 2])
    print(f"Accuracy: {accuracy}")
    helpers.plot_decision_boundaries(random_forest, test_data[:, 0:2], test_data[:, 2])
    helpers.plt.show()

"""
Question. Train an XGBoost model with the same parameters as
the random forest from Q6 as well as learning rate=0.1 . Report
its test accuracy and visualize the XGBoost predictions as in Q4.
How are the predictions of XGBoost different from the ones of the
random forest? Which algorithm is more successful in this task?
Explain.
"""


def xgboost():
    """

    Returns:

    """
    xgboost = helpers.loading_xgboost()
    xgboost.fit(train_data[:, 0:2], train_data[:, 2])
    xgboost_prediction = xgboost.predict(test_data[:, 0:2])
    accuracy = np.mean(xgboost_prediction == test_data[:, 2])
    print(f"Accuracy: {accuracy}")
    helpers.plot_decision_boundaries(xgboost, test_data[:, 0:2], test_data[:, 2])
    helpers.plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    train_data, col_name = helpers.read_data_demo("train.csv")
    test_data, col_name_test = helpers.read_data_demo("test.csv")
    validation_data, col_name_val = helpers.read_data_demo("validation.csv")
    knn_demo()
    AD_test_data()
    decision_tree_demo()
    random_forest()
    xgboost()