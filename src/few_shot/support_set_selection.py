import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
import pickle
from src.prepare_dataset import zero_crossings, calculateMovingAvrgMeanWithUndersampling_v2
from src.prepare_dataset import calculateOtherMLfeatures_oneCh, polygonal_approx

from multiprocessing import Pool
from functools import partial

from sklearn.ensemble import RandomForestClassifier
import joblib

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler


def get_features(signals):
    """
    Extracts features from the input signals.

    Input:
    signals (numpy.ndarray): Input signals with shape (numCh, signal_len), where numCh is the number of channels
                             and signal_len is the length of each signal.

    Returns:
    numpy.ndarray: Array containing the extracted features for each channel.

    Description:
    This function takes an input array of signals and extracts various features for each channel.
    The features include zero-crossing, band powers, LL (Low-Level) features, and a scale factor.
    The function first scales the input signals to a range of [-1, 1] and then calculates the features.
    """

    FS = 256 # Define the sampling frequency
    # %%
    numCh = signals.shape[0]  # Number of channels
    num_feat = 7  # Number of features to be extracted for each channel
    time_len = 12  # Length of each signal in seconds for calculating moving average features
    eps = 1e-6  # A small epsilon value to avoid division by zero

    # Threshold values for polygonal approximation
    EPS_thresh_arr = [0.01, 0.04, 0.1, 0.4, 0.8]

    all_features = np.zeros((num_feat * numCh))  # Zero-crossing, band powers, LL, and scale_factor

    # Calculate the scale factor for each channel to normalize the signals
    scaleFactor = np.max(signals, axis=1, keepdims=True) - np.min(signals, axis=1, keepdims=True)
    signalsScaled = (signals - np.min(signals, axis=1, keepdims=True)) / (scaleFactor + eps)
    signalsScaled = signalsScaled * 2 - 1

    for ch in range(numCh):
        signal_ch = signalsScaled[ch, :]

        # Calculate other Machine Learning features for the channel
        featOther = calculateOtherMLfeatures_oneCh(np.copy(signal_ch), FS, time_len, time_len)

        # Store the calculated features in the all_features array
        all_features[ch * num_feat:ch * num_feat + 5] = featOther
        all_features[ch * num_feat+ 5] = scaleFactor[ch] / 5000

        # Calculate zero-crossing feature for the original signal
        x = np.convolve(zero_crossings(signal_ch), np.ones(FS), mode='same')

        zeroCrossStandard = calculateMovingAvrgMeanWithUndersampling_v2(x, FS * time_len, FS)

        all_features[ch * num_feat + 6] = zeroCrossStandard

        # Calculate zero-crossing features for polygonal approximations with different thresholds
        # for EPSthrIndx, EPSthr in enumerate(EPS_thresh_arr):
        #     sigApprox = polygonal_approx(signal_ch, epsilon=EPSthr)
        #     sigApproxInterp = np.interp(np.arange(len(signal_ch)), sigApprox,
        #                                 signal_ch[sigApprox])
        #     x = np.convolve(zero_crossings(sigApproxInterp), np.ones(FS), mode='same')
        #
        #     zeroCrossApprox = calculateMovingAvrgMeanWithUndersampling_v2(x, FS * time_len, FS)
        #
        #     all_features[ch * num_feat + 7 + EPSthrIndx] = zeroCrossApprox

    return all_features


def process_file(file_path, features_dir):
    try:
        # Load data from the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Call the get_features function with the loaded data
        features = get_features(data['signals'])

        # Create a new filename for the features pickle file
        features_filename = os.path.splitext(os.path.basename(file_path))[0] + "_features.pkl"
        features_file_path = os.path.join(features_dir, features_filename)

        # Save the features in a new pickle file
        with open(features_file_path, 'wb') as features_file:
            pickle.dump(features, features_file)

        # print(f"Features extracted from '{os.path.basename(file_path)}' and saved to '{features_file_path}'")

    except Exception as e:
        print(f"Error processing file '{os.path.basename(file_path)}': {str(e)}")


def process_files():
    dir_name = "../../TUSZv2/preprocess/task-binary_datatype-train"
    # Check if the directory exists
    if not os.path.exists(dir_name):
        print(f"Directory '{dir_name}' does not exist.")
        return

    # Get a list of files in the directory
    file_list = os.listdir(dir_name)

    # Create a new directory for saving the features
    features_dir = dir_name + "_features"
    os.makedirs(features_dir, exist_ok=True)

    # Use multiprocessing to process files in parallel
    num_processes = 6  # Number of available CPU cores
    with Pool(num_processes) as pool:
        # Map the file processing function to the list of files
        partial_process_file = partial(process_file, features_dir=features_dir)
        list(tqdm(pool.imap(partial_process_file, [os.path.join(dir_name, filename) for filename in file_list]),
                  total=len(file_list), desc="Processing Files", unit="file"))


def load_features_from_folder(folder_path):
    features = []
    labels = []
    filenames = []
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith("_features.pkl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                feature_vector = pickle.load(file)
                features.append(feature_vector)

            # Extract the label from the filename (positive if "seiz" in filename, otherwise negative)
            if "seiz" in filename:
                label = 1
            else:
                label = 0
            labels.append(label)
            filenames.append(filename.split("_features.pkl")[0])

    return np.array(features), np.array(labels), np.array(filenames)


def train_random_forest():
    train_folder = "../../TUSZv2/preprocess/task-binary_datatype-train_features"
    train_features, y_train, _ = load_features_from_folder(train_folder)
    print("Train data extracted!")
    dev_folder = "../../TUSZv2/preprocess/task-binary_datatype-dev_features"
    dev_features, y_dev, _ = load_features_from_folder(dev_folder)
    print("Dev data extracted!")
    test_folder = "../../TUSZv2/preprocess/task-binary_datatype-eval_features"
    test_features, y_test, _ = load_features_from_folder(test_folder)
    print("Eval data extracted!")

    # Create a Random Forest Classifier with desired parameters
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)

    # Train the model on the training data
    rf_classifier.fit(train_features, y_train)

    # Evaluate the model on the development data
    accuracy = rf_classifier.score(dev_features, y_dev)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Test the model on the test data
    test_accuracy = rf_classifier.score(test_features, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Save the trained model to a file for future inferences
    model_filename = "../../output/rf_model.pkl"
    model_path = os.path.join(".", model_filename)
    joblib.dump(rf_classifier, model_path)
    print(f"Model saved to: {model_path}")


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def inference_random_forest():
    # Load the saved model for future inferences
    model_path = '../../output/rf_model.pkl'
    loaded_model = joblib.load(model_path)

    dev_folder = "../../TUSZv2/preprocess/task-binary_datatype-dev_features"
    dev_features, y_dev, _ = load_features_from_folder(dev_folder)
    predicted_probabilities = loaded_model.predict_proba(dev_features)
    threshold = thresh_max_f1(y_dev, predicted_probabilities[:, 1])
    print("Best threshold ", threshold)

    # Now you can use the loaded_model to make predictions on new data
    test_folder = "../../TUSZv2/preprocess/task-binary_datatype-eval_features"
    test_features, y_test, _ = load_features_from_folder(test_folder)
    predicted_probabilities = loaded_model.predict_proba(test_features)
    auc_score = roc_auc_score(y_test, predicted_probabilities[:, 1])
    print(f"AUC Score: {auc_score:.4f}")

    # Calculate the confusion matrix
    predicted_classes = (predicted_probabilities[:, 1] > threshold).astype(int)

    conf_matrix = confusion_matrix(y_test, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)


def apply_kmedoids_to_class(class_features, class_label, n_clusters):
    # Apply K-Medoids clustering to obtain n_clusters centroids for the class
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(class_features)
    centroids_indices = kmedoids.medoid_indices_

    # Get the cluster assignments for each data point
    cluster_assignments = kmedoids.predict(class_features)

    # Count the number of samples in each cluster
    samples_in_clusters = {}
    for cluster in range(n_clusters):
        samples_in_cluster = np.sum(cluster_assignments == cluster)
        samples_in_clusters[cluster] = samples_in_cluster

    print("Samples in class {}: {}".format(class_label, samples_in_clusters))

    return centroids_indices


def k_mean_clusters():
    train_folder = "../../TUSZv2/preprocess/task-binary_datatype-train_features"
    train_features, labels, filenames = load_features_from_folder(train_folder)
    train_features = train_features.astype(np.float32)

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the StandardScaler on the train_features to compute the mean and standard deviation
    scaler.fit(train_features)

    # Transform the train_features to have zero mean and unit variance
    standardized_train_features = scaler.transform(train_features)

    print("Features Mean STD", standardized_train_features.mean(axis=0),  standardized_train_features.std(axis=0))
    print("Train data extracted!", standardized_train_features.shape)
    input()
    # Number of clusters (n_clusters) for each class
    n_clusters = 25

    # Apply K-Medoids clustering to class 0 (label 0)
    class_indices = (labels == 0)
    class_features = train_features[class_indices]
    class_0_centroids = apply_kmedoids_to_class(class_features, class_label=0, n_clusters=n_clusters)
    filenames_0_centroids = filenames[class_indices][class_0_centroids]

    # Apply K-Medoids clustering to class 1 (label 1)
    class_indices = (labels == 1)
    class_features = train_features[class_indices]
    class_1_centroids = apply_kmedoids_to_class(class_features, class_label=1, n_clusters=n_clusters)
    filenames_1_centroids = filenames[class_indices][class_1_centroids]

    # class_0_centroids and class_1_centroids will each have n_clusters centroids (shape: (n_clusters, 140))
    # These centroids represent the cluster centers for each class after K-Medoids clustering.

    # Example of accessing the centroids for class 0:
    print("Centroids for Class 0:")
    print(filenames_0_centroids)

    # Example of accessing the centroids for class 1:
    print("Centroids for Class 1:")
    print(filenames_1_centroids)


if __name__ == '__main__':
    k_mean_clusters()
