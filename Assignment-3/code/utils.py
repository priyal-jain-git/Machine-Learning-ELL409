# utils1.py

import os
import numpy as np
SEED = 6  # DO NOT CHANGE
rng = np.random.default_rng(SEED)  # DO NOT CHANGE
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MNISTPreprocessor:
    # Class variables to store scaler and PCA
    scaler = StandardScaler()
    pca = None

    def __init__(self, data_dir, config=None):
        """
        Initializes the MNISTPreprocessor.

        Parameters:
        - data_dir: str, the base directory where the MNIST images are stored.
        - config: dict, configuration for preprocessing (e.g., PCA components)
        """
        self.data_dir = data_dir
        self.classes = [str(i) for i in range(10)]
        self.image_paths = self._get_image_paths()

        self.use_pca = config.get('use_pca') if config else None
        if self.use_pca and MNISTPreprocessor.pca is None:
            MNISTPreprocessor.pca = PCA(n_components=self.use_pca)

    def _get_image_paths(self):
        """
        Collects all image file paths from the data directory for all classes.
        """
        image_paths = []
        for label in self.classes:
            class_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, fname)
                    image_paths.append((image_path, int(label)))
        return image_paths

    def get_all_data(self):
        """
        Loads all data into memory, applies filtering, scaling, and PCA if specified.

        Returns:
        - X_processed: numpy array of shape (num_samples, n_features)
        - y: numpy array of shape (num_samples,)
        """

        # load and resize images
        X, y = self._load_images_and_labels()

        # Apply scaling
        if not hasattr(MNISTPreprocessor.scaler, 'mean_'):
            X = MNISTPreprocessor.scaler.fit_transform(X)
        else:
            X = MNISTPreprocessor.scaler.transform(X)

        # Apply PCA if specified
        if self.use_pca:
            if not hasattr(MNISTPreprocessor.pca, 'components_'):
                X = MNISTPreprocessor.pca.fit_transform(X)
            else:
                X = MNISTPreprocessor.pca.transform(X)

        return X, y

    def _load_images_and_labels(self):
        """
        Helper method to load images and labels.
        """
        X = []
        y = []
        for image_path, label in self.image_paths:
            image = Image.open(image_path).convert('L')
            try:
                image = image.resize((28, 28), Image.Resampling.LANCZOS)
            except AttributeError:
                image = image.resize((28, 28), Image.LANCZOS)
            image_array = np.array(image).astype(np.float32) / 255.0
            image_vector = image_array.flatten()
            X.append(image_vector)
            y.append(label)
        return np.array(X), np.array(y)

def filter_dataset(X, y, entry_number_digit):

    target_digits = [
        entry_number_digit % 10,
        (entry_number_digit - 1) % 10,
        (entry_number_digit + 1) % 10,
        (entry_number_digit + 2) % 10
    ]

    # Filter to only target digits
    condition = np.isin(y, target_digits)
    relevant_pos = np.where(condition)[0]
    X_filtered = X[relevant_pos]
    y_filtered = y[relevant_pos]

    return X_filtered, y_filtered

def convert_labels_to_svm_labels(arr, svm_pos_label=0):
    converted_labels = np.where(arr == svm_pos_label, 1, -1)
    return converted_labels

def get_metrics_as_dict(true, preds):
    assert (
        preds.shape == true.shape
    ), "Shape Mismatch. Assigning 0 score"

    report = classification_report(true, preds, output_dict=True)
    return report

def val_score(true, preds):
    assert (
        preds.shape == true.shape
    ), "Shape Mismatch. Assigning 0 score"

    report = classification_report(true, preds, output_dict=True)
    return report["macro avg"]['f1-score']
