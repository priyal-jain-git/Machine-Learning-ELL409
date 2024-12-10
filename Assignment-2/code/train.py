import pandas as pd  # for data manipulation and analysis
import numpy as np  # for numerical operations
import copy  # for creating deep copies of objects
import time  # for tracking execution time
from imblearn.over_sampling import SMOTE  # for handling class imbalance
from sklearn.utils import resample  # for handling class imbalance
from sklearn.model_selection import KFold  # for cross-validation
from sklearn.model_selection import train_test_split  # for splitting data into train and validation sets

# TODO: Figure out how to create a test.py
# TODO: Make report

class DataPreprocessor:
    def __init__(self):
        # Initialize the DataPreprocessor class with predefined feature lists
        self.numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        self.categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'default', 'housing', 'loan']

    def _load_data(self, filename):
        """
        Load data from a specified file.

        Args:
            filename (str): Path to the file to be loaded.
        
        Returns:
            DataFrame: Loaded data as a pandas DataFrame.
        """
        return pd.read_excel(filename)  # Read the Excel file and return as DataFrame
    
    def _handle_class_imbalance(self, data, target_col, method='upsample'):
        """
        Balance dataset using the specified method.

        Args:
            data (DataFrame): The input data to balance.
            target_col (str): The target column for classification.
            method (str): Method for balancing ('upsample' or 'smote').
        
        Returns:
            DataFrame: Balanced dataset.
        """
        # Split the data into positive and negative samples based on the target column
        
        pos_samples = data[data[target_col] == 'yes']
        neg_samples = data[data[target_col] == 'no']

        # Determine minority and majority classes
        if len(pos_samples) < len(neg_samples):
            minority = pos_samples
            majority = neg_samples
        else:
            minority = neg_samples
            majority = pos_samples
            
        if method == 'upsample':
            # Perform upsampling to balance the dataset
            upsampled_minority = resample(
                minority,
                replace=True,  # Allow replacement of samples
                n_samples=len(majority),  # Match the majority class size
                random_state=42  # Set random state for reproducibility
            )
            # Concatenate majority and upsampled minority classes
            return pd.concat([majority, upsampled_minority])
            
        elif method == 'smote':
            # Use SMOTE for synthetic minority over-sampling
            X = data.drop(columns=[target_col])  # Separate features
            y = data[target_col]  # Separate target
            sm = SMOTE(random_state=42)  # Initialize SMOTE
            # Fit and resample the data
            X_balanced, y_balanced = sm.fit_resample(X, y)
            # Concatenate the balanced features and target
            return pd.concat([X_balanced, pd.Series(y_balanced, name=target_col)], axis=1)
        else:
            # Raise error for unknown balancing method
            raise ValueError("Unknown Balancing Method")
        
    def _scale_numeric_features(self, data):
        """
        Normalize numeric features using standard scaling without StandardScaler.

        Args:
            data (DataFrame): The input data containing numeric features.
        
        Returns:
            DataFrame: Data with normalized numeric features.
        """

        # Calculate the mean and standard deviation for each numeric feature
        means = data[self.numeric_features].mean()
        stds = data[self.numeric_features].std()

        # Apply the standardization formula: (x - mean) / std
        data[self.numeric_features] = (data[self.numeric_features] - means) / stds
        return data  # Return the scaled data

    
    def _encode_categorical_features(self, data, encoding_method='onehot'):
        """
        Encode categorical variables using the specified method.

        Args:
            data (DataFrame): The input data containing categorical features.
            encoding_method (str): Method for encoding ('onehot' or 'label').
        
        Returns:
            DataFrame: Data with encoded categorical features.
        """
        if encoding_method == 'onehot':
            # Apply one-hot encoding to the categorical features
            encoded_data = pd.get_dummies(data, columns=self.categorical_features, drop_first=True)
            return encoded_data  # Return the encoded data
        elif encoding_method == 'label':
            # Apply label encoding to each categorical feature
            for col in self.categorical_features:
                # Get unique categories and create a mapping to integers
                unique_categories = data[col].unique()
                category_map = {category: idx for idx, category in enumerate(unique_categories)}
                
                # Map the original categories to integers
                data[col] = data[col].map(category_map)
            
            return data  # Return the encoded data

        else:
            # Raise error for unknown encoding method
            raise ValueError("Unknown Categorical Encoding Method")
    
    def _split_features_target(self, data):
        """
        Split features and target variable.

        Args:
            data (DataFrame): The input data containing features and target.
        
        Returns:
            tuple: Features (X) and target (y) as separate DataFrames/Series.
        """
        X = data.drop(columns=['y'])  # Drop target column to create features DataFrame
        # Convert target column to binary (1 for 'yes', 0 for 'no')
        y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)
        return X, y  # Return features and target
    
    def load_and_preprocess(self, filename, balance_method='upsample', encoding_method='onehot', test=True):
        """
        Main pre-processing pipeline.

        Args:
            filename (str): Path to the file to be loaded.
            balance_method (str): Method for balancing the dataset.
            encoding_method (str): Method for encoding categorical variables.
        
        Returns:
            tuple: Processed features (X) and target (y).
        """
        data = self._load_data(filename)  # Load data from the specified file
        if (test==True):
            data = data.drop(columns=['id'])
        if (test==False):
            data = self._handle_class_imbalance(data, 'y', balance_method)  # Balance the dataset
        data = self._scale_numeric_features(data)  # Scale numeric features
        data = self._encode_categorical_features(data, encoding_method)  # Encode categorical features
        if (test==False):
            return self._split_features_target(data)  # Split and return features and target
        else:
            return data
    
    def determine_feature_types(self, data):
        """
        Determine feature types.

        Args:
            data (DataFrame): The input data containing features.
        
        Returns:
            list: List of feature types ('numeric' or 'categorical').
        """
        feature_types = []  # Initialize an empty list to store feature types
        for col in data.columns:  # Iterate through each column in the DataFrame
            if col in self.numeric_features:  # Check if the column is numeric
                feature_types.append('numeric')  # Append 'numeric' to the list
            else:
                feature_types.append('categorical')  # Append 'categorical' to the list
        return feature_types  # Return the list of feature types


class DecisionTreeMetrics:
    @staticmethod
    def gini(y):
        """
        Calculate Gini impurity for a given array of class labels.

        Args:
            y (array-like): Array of class labels (0s and 1s).
        
        Returns:
            float: Gini impurity score.
        """
        m = len(y)  # Number of samples
        if m == 0:  # If no samples, return 0 impurity
            return 0
        class_counts = np.bincount(y)  # Count occurrences of each class
        probabilities = class_counts / m  # Calculate probabilities for each class
        impurity = 1 - np.sum(probabilities ** 2)  # Calculate Gini impurity
        return impurity  # Return impurity score

    @staticmethod
    def entropy(y):
        """
        Calculate entropy for a given array of class labels.

        Args:
            y (array-like): Array of class labels (0s and 1s).
        
        Returns:
            float: Entropy score.
        """
        m = len(y)  # Number of samples
        if m == 0:  # If no samples, return 0 entropy
            return 0
        class_counts = np.bincount(y)  # Count occurrences of each class
        probabilities = class_counts / m  # Calculate probabilities for each class
        nonzero_probabilities = probabilities[probabilities > 0]  # Filter non-zero probabilities
        impurity = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))  # Calculate entropy
        return impurity  # Return entropy score

    @staticmethod
    def information_gain(X_column, y, split_value, impurity_measure):
        """
        Calculate information gain from a split on a feature.

        Args:
            X_column (array-like): Feature column to split on.
            y (array-like): Array of class labels (0s and 1s).
            split_value (float): Value at which to split the feature.
            impurity_measure (str): Measure to use for impurity ('gini' or 'entropy').
        
        Returns:
            float: Information gain from the split.
        """
        left_mask = X_column < split_value  # Mask for left child
        right_mask = ~left_mask  # Mask for right child
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:  # If no samples in one side
            return 0  # No information gain

        y_left, y_right = y[left_mask], y[right_mask]  # Split the target array

        # Select the appropriate impurity function based on the criterion
        if impurity_measure == 'gini':
            impurity_fn = DecisionTreeMetrics.gini
        elif impurity_measure == 'entropy':
            impurity_fn = DecisionTreeMetrics.entropy
        else:
            raise ValueError("Unknown impurity measure")  # Raise error for unknown impurity measure

        # Calculate impurity for left and right child nodes
        impurity_left_child = impurity_fn(y_left)
        impurity_right_child = impurity_fn(y_right)
        impurity_parent = impurity_fn(y)  # Calculate impurity for parent node

        # Calculate probabilities for left and right child nodes
        probability_left = len(y_left) / len(y)
        probability_right = len(y_right) / len(y)

        # Weighted impurity of children
        weighted_impurity_children = probability_left * impurity_left_child + probability_right * impurity_right_child
        return impurity_parent - weighted_impurity_children  # Return information gain

class Node:
    def __init__(self, feature=None, split_value=None, left=None, right=None, label=None):
        """
        Initialize a Node for the decision tree.

        Args:
            feature (int): Index of the feature used for splitting.
            split_value (float): Value at which the node splits the feature.
            left (Node): Left child node.
            right (Node): Right child node.
            label (int): Class label for leaf nodes.
        """
        self.feature = feature  # Feature index for the split
        self.split_value = split_value  # Value at which to split
        self.left = left  # Left child (Node)
        self.right = right  # Right child (Node)
        self.label = label  # Class label (only for leaf nodes)
    
    def count_nodes(self):
        """
        Count the number of nodes in the subtree rooted at this node.

        Returns:
            int: Total number of nodes.
        """
        if self.label is not None:  # If it's a leaf node
            return 1  # Count this node
        else:
            left_count = 0  # Initialize left count
            right_count = 0  # Initialize right count
            
            if self.left is not None:  # If left child exists
                left_count = self.left.count_nodes()  # Recursively count left nodes
            if self.right is not None:  # If right child exists
                right_count = self.right.count_nodes()  # Recursively count right nodes
            
            return 1 + left_count + right_count  # Total nodes = 1 (this node) + left + right

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=25, criterion='gini'):
        """
        Initialize the DecisionTree classifier.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum samples required to split an internal node.
            criterion (str): Function to measure the quality of a split ('gini' or 'entropy').
        """
        self.max_depth = max_depth  # Set maximum depth
        self.min_samples_split = min_samples_split  # Set minimum samples split
        self.criterion = criterion  # Set criterion for impurity measure
        self.root = None  # Initialize root node
        self.feature_types = None  # Placeholder for feature types

    def fit(self, X, y, feature_types):
        """
        Fit the decision tree to the training data.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            feature_types (list): List of feature types (e.g., numeric, categorical).
        """
        self.feature_types = feature_types  # Store feature types
        self.root = self._build_tree(X, y, depth=0)  # Build the decision tree

    def predict(self, X):
        """
        Predict class labels for the provided features.

        Args:
            X (array-like): Feature matrix for prediction.
        
        Returns:
            array: Predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])  # Traverse tree for each sample

    def _best_split(self, X, y):
        """
        Find the best split for the dataset.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
        
        Returns:
            tuple: Best feature index and best split value.
        """
        best_gain = -1  # Initialize best gain
        best_feature = None  # Initialize best feature
        best_split = None  # Initialize best split

        # Iterate over each feature to find the best split
        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]  # Get the feature column
            unique_values = np.unique(X_column)  # Get unique values in the feature
            candidate_splits = np.linspace(np.min(unique_values), np.max(unique_values), num=10)  # Generate candidate splits

            # Iterate over each candidate split value
            for split_value in candidate_splits:
                # Calculate information gain for the split
                gain = DecisionTreeMetrics.information_gain(X_column, y, split_value, self.criterion)
                if gain > best_gain:  # If gain is better than current best
                    best_gain = gain  # Update best gain
                    best_feature = feature_idx  # Update best feature index
                    best_split = split_value  # Update best split value

        return best_feature, best_split  # Return best feature and split

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            depth (int): Current depth of the tree.
        
        Returns:
            Node: Root node of the subtree.
        """
        # Check if we should stop splitting and create a leaf node
        if self._should_stop_splitting(X, y, depth):
            return self._create_leaf(y)  # Create and return a leaf node

        feature_idx, split_value = self._best_split(X, y)  # Find best feature and split value

        # Check if a valid split was found
        if feature_idx is None or split_value is None:
            return self._create_leaf(y)  # Create a leaf if no valid split

        left_mask = X[:, feature_idx] < split_value  # Mask for left child
        right_mask = ~left_mask  # Mask for right child

        # Check if there are samples in both left and right splits
        if not np.any(left_mask) or not np.any(right_mask):
            return self._create_leaf(y)  # Create leaf if no samples in one side

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)  # Left child
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)  # Right child

        return Node(feature=feature_idx, split_value=split_value, left=left_child, right=right_child)  # Return the node

    def _should_stop_splitting(self, X, y, depth):
        """
        Check if the tree should stop splitting.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            depth (int): Current depth of the tree.
        
        Returns:
            bool: True if stopping criteria met, else False.
        """
        # Check conditions to stop splitting
        if (self.max_depth is not None and depth >= self.max_depth) or (X.shape[0] < self.min_samples_split) or len(set(y)) == 1:
            return True  # Stop splitting
        return False  # Continue splitting

    def _create_leaf(self, y):
        """
        Create a leaf node based on the class labels.

        Args:
            y (array-like): Target labels for the leaf.
        
        Returns:
            Node: Leaf node with the most common label.
        """
        label_counts = np.bincount(y)  # Count occurrences of each label
        most_common_label = label_counts.argmax()  # Find the most common label
        return Node(label=most_common_label)  # Return leaf node with most common label

    def _is_leaf(self, node):
        """
        Check if a node is a leaf.

        Args:
            node (Node): Node to check.
        
        Returns:
            bool: True if the node is a leaf, else False.
        """
        return node.label is not None  # Leaf node has a label

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            x (array-like): Sample feature vector.
            node (Node): Current node in the tree.
        
        Returns:
            int: Predicted class label.
        """
        if self._is_leaf(node):  # If it's a leaf node
            return node.label  # Return the label

        feature_value = x[node.feature]  # Get the feature value for the split
        
        # Traverse left or right subtree based on feature value
        if feature_value < node.split_value:
            return self._traverse_tree(x, node.left)  # Go left
        else:
            return self._traverse_tree(x, node.right)  # Go right

class TreePruner:
    @staticmethod
    def cross_validation_pruning(X, y, max_depths, min_samples_splits, criteria, feature_types, k_folds=3):
        """
        Perform cross-validation pruning to find the best parameters for the decision tree.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            max_depths (list): List of maximum depths to evaluate.
            min_samples_splits (list): List of minimum samples splits to evaluate.
            criteria (list): List of criteria for splitting ('gini', 'entropy').
            feature_types (list): List of feature types.
            k_folds (int): Number of folds for cross-validation.
        
        Returns:
            dict: Best parameters found.
        """
        best_params = None  # Initialize best parameters
        best_accuracy = 0  # Initialize best accuracy

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)  # Set up KFold cross-validation

        # Iterate over all combinations of parameters
        for criterion in criteria:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    # Evaluate the model using the current set of parameters
                    mean_accuracy = TreePruner._evaluate_model(X, y, criterion, max_depth, min_samples_split, feature_types, kfold)

                    # Update best parameters if current accuracy is better
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_params = {'criterion': criterion, 'max_depth': max_depth, 'min_samples_split': min_samples_split}

        return best_params  # Return the best parameters found

    @staticmethod
    def _evaluate_model(X, y, criterion, max_depth, min_samples_split, feature_types, kfold):
        """
        Evaluate the model using KFold cross-validation.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            criterion (str): Criterion for splitting ('gini', 'entropy').
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum samples required to split an internal node.
            feature_types (list): List of feature types.
            kfold (KFold): KFold object for splitting data.
        
        Returns:
            float: Mean accuracy across all folds.
        """
        accuracies = []  # List to store accuracies for each fold

        # Iterate through each fold
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]  # Split the data into training and validation sets
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and fit the decision tree model
            tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
            tree.fit(X_train, y_train, feature_types)  # Train the model
            predictions = tree.predict(X_val)  # Make predictions on the validation set

            # Calculate accuracy and append to list
            accuracies.append(np.mean(predictions == y_val))  # Mean accuracy for this fold

        return np.mean(accuracies)  # Return the mean accuracy across all folds

    @staticmethod
    def cost_complexity_pruning(node, X_val, y_val, alpha=0.01):
        """
        Perform cost complexity pruning on the tree.

        Args:
            node (Node): Current node of the tree.
            X_val (array-like): Validation feature matrix.
            y_val (array-like): Validation target labels.
            alpha (float): Complexity parameter for pruning.
        """
        if node.label is not None:  # If it's a leaf node, stop pruning
            return

        # Get split information for the current node
        feature_idx, split_value = node.feature, node.split_value
        left_mask = X_val[:, feature_idx] < split_value  # Mask for left child
        right_mask = ~left_mask  # Mask for right child

        # Prune if either split side is empty
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            node.label = TreePruner._most_common_label(y_val)  # Set label to the most common in validation set
            node.left = node.right = None  # Remove children
            return

        # Recursively prune left and right subtrees
        if node.left:
            TreePruner.cost_complexity_pruning(node.left, X_val[left_mask], y_val[left_mask], alpha)
        if node.right:
            TreePruner.cost_complexity_pruning(node.right, X_val[right_mask], y_val[right_mask], alpha)

        # Check if pruning is beneficial
        TreePruner._check_pruning(node, X_val, y_val, alpha)

    @staticmethod
    def _check_pruning(node, X_val, y_val, alpha):
        """
        Check if pruning the current node is beneficial.

        Args:
            node (Node): Current node of the tree.
            X_val (array-like): Validation feature matrix.
            y_val (array-like): Validation target labels.
            alpha (float): Complexity parameter for pruning.
        """
        # Check if both left and right children are leaves
        if node.left and node.right and node.left.label is not None and node.right.label is not None:
            left_predictions = node.left.label  # Get predictions from left child
            right_predictions = node.right.label  # Get predictions from right child
            
            # Combine predictions based on the split
            overall_predictions = np.where(X_val[:, node.feature] < node.split_value, left_predictions, right_predictions)
            error_if_not_pruned = np.sum(y_val != overall_predictions)  # Calculate error if not pruned

            parent_predictions = TreePruner._most_common_label(y_val)  # Most common label in the validation set
            error_if_pruned = np.sum(y_val != parent_predictions)  # Calculate error if pruned

            # If pruning reduces error, perform pruning
            if error_if_pruned + alpha < error_if_not_pruned:
                node.label = parent_predictions  # Set node label to parent predictions
                node.left = node.right = None  # Remove children

    @staticmethod
    def _most_common_label(y):
        """
        Get the most common label in the given array.

        Args:
            y (array-like): Array of class labels.
        
        Returns:
            int: Most common label.
        """
        return np.bincount(y).argmax()  # Return the most frequent label

class ModelEvaluator:
    """Handles evaluation metrics and performance analysis"""
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Computes confusion matrix for binary classification

        Args:
            y_true (array-like): True labels of the data.
            y_pred (array-like): Predicted labels from the model.

        Returns:
            np.ndarray: A 2x2 confusion matrix.
        """
        y_true = np.asarray(y_true)  # Convert true labels to a numpy array
        y_pred = np.asarray(y_pred)  # Convert predicted labels to a numpy array
        
        # Calculate true positives, true negatives, false positives, and false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
        
        # Return confusion matrix as a 2x2 numpy array
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Computes various classification metrics

        Args:
            y_true (array-like): True labels of the data.
            y_pred (array-like): Predicted labels from the model.

        Returns:
            tuple: A tuple containing accuracy, precision, recall, and F1 score.
        """
        cm = ModelEvaluator.confusion_matrix(y_true, y_pred)  # Get confusion matrix
        tn, fp, fn, tp = cm.ravel()  # Flatten the confusion matrix to extract values
        
        # Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Compute precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Avoid division by zero
        
        # Compute recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
        
        # Compute F1 score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Avoid division by zero
        
        # Return computed metrics as a tuple
        return accuracy, precision, recall, f1

class DecisionTreeAnalysis:
    def __init__(self):
        # Initialize instances of DataPreprocessor and ModelEvaluator
        self.preprocessor = DataPreprocessor()
        self.evaluator = ModelEvaluator()

    def _decision_tree_analysis(self, X_train, y_train, X_val, y_val, feature_types, pre_prune=False):
        """Conducts decision tree analysis including training and prediction.

        Args:
            X_train (ndarray): Training features.
            y_train (ndarray): Training labels.
            X_val (ndarray): Validation features.
            y_val (ndarray): Validation labels.
            feature_types (list): Types of features (numeric/categorical).
            pre_prune (bool): Whether to pre-prune the tree during training.

        Returns:
            dict: A dictionary containing the trained tree, predictions, and metrics.
        """
        # Check if pre-pruning is requested
        if pre_prune:
            # Get best parameters for the tree

            best_params = {'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5}

            '''uncomment below line to run hyperparameter tuning'''
            # best_params = self._get_best_params(X_train, y_train, feature_types)

            # Initialize tree with best parameters
            tree = DecisionTree(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], criterion=best_params['criterion'])
        else:
            # Initialize tree with default parameters
            tree = DecisionTree(max_depth=None, criterion='gini')

        # Fit the tree to the training data
        tree.fit(X_train, y_train, feature_types)
        # Make predictions on the validation data
        predictions = tree.predict(X_val)
        # Compute evaluation metrics
        metrics = self.evaluator.compute_metrics(y_val, predictions)

        return {
            'tree': tree,  # Return the trained tree
            'predictions': predictions,  # Return predictions
            'metrics': metrics  # Return computed metrics
        }

    def _get_best_params(self, X_train, y_train, feature_types):
        """Find the best hyperparameters for the decision tree.

        Args:
            X_train (ndarray): Training features.
            y_train (ndarray): Training labels.
            feature_types (list): Types of features (numeric/categorical).

        Returns:
            dict: The best hyperparameters for the tree.
        """
        max_depth_values = [8,10,20]  # List of max depths to evaluate
        min_samples_split_values = [2,5,10]  # List of min samples splits to evaluate
        criteria_list = ['entropy','gini']  # List of criteria to evaluate

        # Use cross-validation pruning to find the best parameters
        return TreePruner.cross_validation_pruning(X_train, y_train, max_depth_values, min_samples_split_values, criteria_list, feature_types)

    def _post_pruning_analysis(self, X_train, y_train, X_val, y_val, feature_types, full_tree):
        """Conducts post-pruning analysis on the decision tree.

        Args:
            X_train (ndarray): Training features.
            y_train (ndarray): Training labels.
            X_val (ndarray): Validation features.
            y_val (ndarray): Validation labels.
            feature_types (list): Types of features (numeric/categorical).
            full_tree (DecisionTree): The fully grown decision tree.

        Returns:
            dict: A dictionary containing the pruned tree, predictions, metrics, and best alpha.
        """
        # Get the best alpha value for pruning
        best_alpha=0.01

        '''uncomment below line to find best alpha'''
        # best_alpha = self._get_best_alpha(full_tree, X_val, y_val)
        
        # Create a deep copy of the full tree to prune
        pruned_tree = copy.deepcopy(full_tree)

        # Perform cost complexity pruning on the tree
        TreePruner.cost_complexity_pruning(pruned_tree.root, X_val, y_val, alpha=best_alpha)  

        # Use the pruned tree for predictions
        predictions = pruned_tree.predict(X_val)
        
        # Compute evaluation metrics
        metrics = self.evaluator.compute_metrics(y_val, predictions)
        return {
            'tree': pruned_tree,  # Return the pruned tree
            'predictions': predictions,  # Return predictions
            'metrics': metrics,  # Return computed metrics
            'alpha': best_alpha  # Return best alpha value used for pruning
        }

    def _get_best_alpha(self, full_tree, X_val, y_val):
        """Find the best alpha for cost complexity pruning.

        Args:
            full_tree (DecisionTree): The fully grown decision tree.
            X_val (ndarray): Validation features.
            y_val (ndarray): Validation labels.

        Returns:
            float: The best alpha value for pruning.
        """
        alpha_values = [0.1,0.01,0.001]  # List of alpha values to evaluate
        best_alpha, best_accuracy = None, 0  # Initialize best alpha and accuracy

        # Iterate through different alpha values
        for alpha in alpha_values:
            # Create a temporary copy of the full tree for pruning
            temp_tree = copy.deepcopy(full_tree)
            # Prune the tree with the current alpha
            TreePruner.cost_complexity_pruning(temp_tree.root, X_val, y_val, alpha)
            # Get predictions from the pruned tree
            predictions = temp_tree.predict(X_val)
            # Compute accuracy for the predictions
            accuracy = self.evaluator.compute_metrics(y_val, predictions)[0]

            # Check if this accuracy is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy  # Update best accuracy
                best_alpha = alpha  # Update best alpha

        return best_alpha  # Return the best alpha found

    
import matplotlib.pyplot as plt

def plot_class_distribution(y, save_path='class_distribution.png'):
    """
    Plot a pie chart showing the distribution of classes in the dataset and save it to a file.
    
    Args:
        y (array-like): Array of class labels
        save_path (str): Path where the plot should be saved
    """
    # Calculate class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Calculate percentages
    percentages = class_counts / len(y) * 100
    
    # Create labels with both count and percentage
    labels = [f'y = {"no" if c == 0 else "yes"}\n({percentage:.1f}%)' 
             for c, percentage in zip(unique_classes, percentages)]
    
    # Create pie chart
    colors = ['#FF9999', '#90EE90']
    plt.figure(figsize=(10, 8))
    plt.pie(class_counts, labels=labels, autopct='', 
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 2},
            textprops={'fontsize': 20, 'fontweight': 'bold'})
    
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the plot instead of displaying it
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    print(f"\nClass distribution plot has been saved to: {save_path}")


def main():
    # Start timer for performance measurement
    start_time = time.time()

    # Define the filename for the training data
    training_data_filename = "train.xlsx"

    # Create an instance of DecisionTreeAnalysis
    analyser = DecisionTreeAnalysis()

    # Load and preprocess the data
    X, y = analyser.preprocessor.load_and_preprocess(filename=training_data_filename, test=False)
    
    # Determine feature types (numeric/categorical)
    feature_types = analyser.preprocessor.determine_feature_types(X)

    # Convert features and labels to numpy arrays
    X_values = X.values
    y_values = y.values

    # Split data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_values, y_values, test_size=0.2, random_state=42)

    # Get the pre-pruned decision tree
    pre_pruned_results = analyser._decision_tree_analysis(X_train, y_train, X_val, y_val, feature_types, pre_prune=True)

    '''uncomment below line to get base tree'''
    # full_tree_results = analyser._decision_tree_analysis(X_train, y_train, X_val, y_val, feature_types)
    
    # Get the pre+post pruned decision tree
    post_pruned_results = analyser._post_pruning_analysis(X_train, y_train, X_val, y_val, feature_types, pre_pruned_results['tree'])
    
    execution_time=time.time()-start_time
    accuracy, precision, recall, f1=post_pruned_results['metrics']
    num_nodes=post_pruned_results['tree'].root.count_nodes()

    # Print the results
    print(f"Execution Time: {execution_time:.4f} seconds")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print()
    print(f"Number of Nodes: {num_nodes}")


    '''uncomment below to show class distribution pie chart'''
    '''
    print("\nClass Distribution Analysis:")
    print("-" * 30)
    unique_classes, class_counts = np.unique(y_values, return_counts=True)
    for class_label, count in zip(unique_classes, class_counts):
        print(f"Class {class_label}: {count:,} samples ({count/len(y_values)*100:.1f}%)")
    
    # Plot the distribution and save to file
    plot_class_distribution(y_values)
    '''


# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function to execute the program