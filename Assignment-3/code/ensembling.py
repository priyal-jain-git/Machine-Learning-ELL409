# ensembling.py

# Import necessary modules and classes
from base_ensemble import *  # Import all classes and functions from base_ensemble module
from utils import *          # Import all classes and functions from utils module
import numpy as np           # Import NumPy for numerical operations


# --------------------- Helper Classes ---------------------

# Define a class to compute impurity metrics for decision trees (Taken from assignment2)
class DecisionTreeMetrics:
    @staticmethod
    def gini(y):
        """
        Calculate Gini impurity for a given array of class labels.

        Gini impurity measures the probability of incorrectly
        classifying a randomly chosen element if it were labeled
        according to the distribution of labels in the subset.

        Parameters:
        - y: np.ndarray
            Array of class labels.

        Returns:
        - impurity: float
            Gini impurity value.
        """
        m = len(y)  # Total number of samples
        if m == 0:
            return 0  # If no samples, impurity is zero
        # Find unique classes and their corresponding indices
        classes, y_indices = np.unique(y, return_inverse=True)
        # Count occurrences of each class
        class_counts = np.bincount(y_indices)
        # Calculate probabilities for each class
        probabilities = class_counts / m
        # Compute Gini impurity
        impurity = 1 - np.sum(probabilities ** 2)
        return impurity

    @staticmethod
    def entropy(y):
        """
        Calculate entropy for a given array of class labels.

        Entropy measures the amount of uncertainty or randomness
        in the distribution of class labels.

        Parameters:
        - y: np.ndarray
            Array of class labels.

        Returns:
        - impurity: float
            Entropy value.
        """
        m = len(y)  # Total number of samples
        if m == 0:
            return 0  # If no samples, entropy is zero
        # Find unique classes and their corresponding indices
        classes, y_indices = np.unique(y, return_inverse=True)
        # Count occurrences of each class
        class_counts = np.bincount(y_indices)
        # Calculate probabilities for each class
        probabilities = class_counts / m
        # Exclude zero probabilities to avoid log2(0)
        nonzero_probabilities = probabilities[probabilities > 0]
        # Compute entropy
        impurity = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
        return impurity

    @staticmethod
    def information_gain(X_column, y, split_value, impurity_measure):
        """
        Calculate information gain from a split on a feature.

        Information gain measures the reduction in impurity
        achieved by partitioning the data based on a feature.

        Parameters:
        - X_column: np.ndarray
            Array of feature values for a single feature.
        - y: np.ndarray
            Array of class labels.
        - split_value: float
            Threshold value to split the feature.
        - impurity_measure: str
            The impurity measure to use ('gini' or 'entropy').

        Returns:
        - gain: float
            Information gain from the split.
        """
        # Create boolean masks for left and right splits based on split_value
        left_mask = X_column < split_value
        right_mask = ~left_mask  # Complement of left_mask

        # If either split has no samples, return zero gain
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        # Split the labels into left and right subsets
        y_left, y_right = y[left_mask], y[right_mask]

        # Select the appropriate impurity function based on the criterion
        if impurity_measure == 'gini':
            impurity_fn = DecisionTreeMetrics.gini
        elif impurity_measure == 'entropy':
            impurity_fn = DecisionTreeMetrics.entropy
        else:
            raise ValueError("Unknown impurity measure")

        # Calculate impurity for left and right child nodes
        impurity_left_child = impurity_fn(y_left)
        impurity_right_child = impurity_fn(y_right)
        # Calculate impurity of the parent node
        impurity_parent = impurity_fn(y)

        # Calculate the proportion of samples in left and right child nodes
        probability_left = len(y_left) / len(y)
        probability_right = len(y_right) / len(y)

        # Compute the weighted impurity of the child nodes
        weighted_impurity_children = (
            probability_left * impurity_left_child + 
            probability_right * impurity_right_child
        )
        # Information gain is the reduction in impurity
        gain = impurity_parent - weighted_impurity_children
        return gain

# Define a Node class to represent each node in the decision tree (Taken from assignment2)
class Node:
    def __init__(self, feature=None, split_value=None, left=None, right=None, label=None):
        """
        Initialize a tree node.

        Parameters:
        - feature: int or None
            Index of the feature to split on.
        - split_value: float or None
            Threshold value for the split.
        - left: Node or None
            Left child node.
        - right: Node or None
            Right child node.
        - label: int or None
            Class label if the node is a leaf.
        """
        self.feature = feature              # Feature index to split on
        self.split_value = split_value      # Threshold value for the split
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.label = label                  # Class label if leaf node

# Define the DecisionTree class for use in Random Forest (Taken from assignment2)
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=25, criterion='gini'):
        """
        Initialize the DecisionTree.

        Parameters:
        - max_depth: int or None
            Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        - min_samples_split: int
            Minimum number of samples required to split an internal node.
        - criterion: str
            Impurity measure to use ('gini' or 'entropy').
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None                      # Root node of the tree
        self.feature_types = None             # List indicating feature types (e.g., 'numeric')

    def fit(self, X, y, feature_types):
        """
        Build the decision tree from the training data.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Training data features.
        - y: np.ndarray, shape (n_samples,)
            Training data labels.
        - feature_types: list of str
            List indicating the type of each feature (e.g., ['numeric', 'categorical']).
        """
        self.feature_types = feature_types
        # Start building the tree from the root
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns:
        - predictions: np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        # Traverse the tree for each sample to get predictions
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _best_split(self, X, y):
        """
        Find the best feature and split value to maximize information gain.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Features of the current node.
        - y: np.ndarray, shape (n_samples,)
            Labels of the current node.

        Returns:
        - best_feature: int or None
            Index of the best feature to split on.
        - best_split: float or None
            Best split value for the chosen feature.
        """
        best_gain = -1                  # Initialize best gain
        best_feature = None             # Initialize best feature index
        best_split = None               # Initialize best split value

        # Iterate over all features to find the best split
        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            unique_values = np.unique(X_column)
            # Generate candidate split values (10 evenly spaced points)
            candidate_splits = np.linspace(np.min(unique_values), np.max(unique_values), num=10)

            for split_value in candidate_splits:
                # Calculate information gain for the current split
                gain = DecisionTreeMetrics.information_gain(
                    X_column, y, split_value, self.criterion
                )
                # Update best split if current gain is higher
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_split = split_value

        return best_feature, best_split

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Features of the current node.
        - y: np.ndarray, shape (n_samples,)
            Labels of the current node.
        - depth: int
            Current depth of the tree.

        Returns:
        - node: Node
            The constructed node (either internal or leaf).
        """
        # Check if we should stop splitting and create a leaf
        if self._should_stop_splitting(X, y, depth):
            return self._create_leaf(y)

        # Find the best feature and split value
        feature_idx, split_value = self._best_split(X, y)

        # If no valid split is found, create a leaf
        if feature_idx is None or split_value is None:
            return self._create_leaf(y)

        # Create masks for left and right child nodes based on split_value
        left_mask = X[:, feature_idx] < split_value
        right_mask = ~left_mask

        # If either child node has no samples, create a leaf
        if not np.any(left_mask) or not np.any(right_mask):
            return self._create_leaf(y)

        # Recursively build the left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return the current node with its children
        return Node(feature=feature_idx, split_value=split_value, left=left_child, right=right_child)

    def _should_stop_splitting(self, X, y, depth):
        """
        Determine whether to stop splitting the current node.

        Conditions to stop:
        - Maximum depth is reached.
        - Number of samples is less than min_samples_split.
        - All samples have the same class label.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Features of the current node.
        - y: np.ndarray, shape (n_samples,)
            Labels of the current node.
        - depth: int
            Current depth of the tree.

        Returns:
        - stop: bool
            True if splitting should stop, False otherwise.
        """
        # Check if maximum depth is set and reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        # Check if number of samples is below the threshold
        if X.shape[0] < self.min_samples_split:
            return True
        # Check if all samples have the same class label
        if len(set(y)) == 1:
            return True
        return False

    def _create_leaf(self, y):
        """
        Create a leaf node with the most common class label.

        Parameters:
        - y: np.ndarray, shape (n_samples,)
            Labels of the current node.

        Returns:
        - leaf_node: Node
            Leaf node with the most common class label.
        """
        # Find unique classes and their counts
        classes, y_indices = np.unique(y, return_inverse=True)
        class_counts = np.bincount(y_indices)
        # Identify the most common class
        most_common_class_index = class_counts.argmax()
        most_common_label = classes[most_common_class_index]
        # Return a leaf node with the most common label
        return Node(label=most_common_label)

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters:
        - x: np.ndarray, shape (n_features,)
            Feature values of the sample.
        - node: Node
            Current node in the traversal.

        Returns:
        - label: int
            Predicted class label for the sample.
        """
        # If the node is a leaf, return its label
        if node.label is not None:
            return node.label

        # Get the feature value for the current split
        feature_value = x[node.feature]

        # Decide to go left or right based on the split value
        if feature_value < node.split_value:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Define a DecisionStump class for use in AdaBoost
class DecisionStump:
    def __init__(self):
        """
        Initialize the DecisionStump.

        A DecisionStump is a simple decision tree with only one split.
        It is used as the weak learner in AdaBoost.
        """
        self.feature_index = None   # Feature index to split on
        self.threshold = None       # Threshold value for the split
        self.category = 1           # Category label for one side of the split

    def fit(self, X, y, sample_weights):
        """
        Fit the decision stump using sample weights.

        The fitting process involves finding the feature and threshold
        that minimize the weighted classification error.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Training data features.
        - y: np.ndarray, shape (n_samples,)
            Training data labels. Expected to be -1 or 1.
        - sample_weights: np.ndarray, shape (n_samples,)
            Weights for each sample.
        """
        m, n = X.shape  # Number of samples and features
        min_error = float('inf')  # Initialize minimum error to infinity

        # Iterate over all features to find the best split
        for feature_i in range(n):
            X_column = X[:, feature_i]
            # Sort the feature values and corresponding labels and weights
            sorted_indices = np.argsort(X_column)
            X_sorted = X_column[sorted_indices]
            y_sorted = y[sorted_indices]
            weights_sorted = sample_weights[sorted_indices]

            # Generate possible thresholds as midpoints between sorted feature values
            thresholds = np.concatenate((
                [X_sorted[0] - 1e-10],  # Slightly less than the minimum
                (X_sorted[:-1] + X_sorted[1:]) / 2,
                [X_sorted[-1] + 1e-10]  # Slightly more than the maximum
            ))

            # Compute cumulative sums of weights for positive and negative classes
            cum_weights_positive = np.cumsum(weights_sorted * (y_sorted == 1))
            cum_weights_negative = np.cumsum(weights_sorted * (y_sorted == -1))

            # Calculate weighted errors for category = 1
            # Predict 1 for left of the threshold, -1 for right
            errors_p_plus = cum_weights_positive + (cum_weights_negative[-1] - cum_weights_negative)
            # Find the threshold with minimum error for category = 1
            min_error_p_plus_idx = np.argmin(errors_p_plus)
            min_error_p_plus = errors_p_plus[min_error_p_plus_idx]

            # Calculate weighted errors for category = -1
            # Predict -1 for left of the threshold, 1 for right
            errors_p_minus = cum_weights_negative + (cum_weights_positive[-1] - cum_weights_positive)
            # Find the threshold with minimum error for category = -1
            min_error_p_minus_idx = np.argmin(errors_p_minus)
            min_error_p_minus = errors_p_minus[min_error_p_minus_idx]

            # Update stump parameters if a lower error is found for category = 1
            if min_error_p_plus < min_error:
                min_error = min_error_p_plus
                self.category = 1
                self.threshold = thresholds[min_error_p_plus_idx]
                self.feature_index = feature_i

            # Update stump parameters if a lower error is found for category = -1
            if min_error_p_minus < min_error:
                min_error = min_error_p_minus
                self.category = -1
                self.threshold = thresholds[min_error_p_minus_idx]
                self.feature_index = feature_i

    def predict(self, X):
        """
        Make predictions using the learned decision stump.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns:
        - predictions: np.ndarray, shape (n_samples,)
            Predicted class labels (-1 or 1).
        """
        X_column = X[:, self.feature_index]  # Extract the feature to split on
        predictions = np.ones(X.shape[0])    # Initialize all predictions to 1

        # Depending on the category, assign -1 to one side of the split
        if self.category == -1:
            predictions[X_column >= self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = -1

        return predictions


# --------------------- Main Classes ---------------------

# Define the RandomForestClassifier class inheriting from BaseEnsembler
class RandomForestClassifier(BaseEnsembler):
    def __init__(self, num_trees=10, max_depth=5, min_samples_split=5, max_features=None):
        """
        Initialize the RandomForestClassifier.

        Parameters:
        - num_trees: int
            Number of decision trees in the forest.
        - max_depth: int or None
            Maximum depth of each tree.
        - min_samples_split: int
            Minimum number of samples required to split an internal node.
        - max_features: int, 'sqrt', 'log2', or None
            Number of features to consider when looking for the best split.
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: Exact number of features
            - None: All features
        """
        super().__init__(num_trees)  # Initialize the base ensemble with number of trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []  # List to store all decision trees
    
    def fit(self, X, y):
        """
        Train the Random Forest classifier on the provided data.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Training data features.
        - y: np.ndarray, shape (n_samples,)
            Training data labels.
        """
        self.trees = []  # Reset the list of trees
        n_samples, n_features = X.shape
        max_features = self._get_feature_subset(n_features)  # Determine features per tree

        # Iterate to build each tree in the forest
        for _ in range(self.num_trees):
            # Generate a bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            # Randomly select a subset of features for this tree
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            # Initialize and train the tree
            tree = self._initialize_tree(X_sample, y_sample, feature_indices)
            # Add the trained tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if not self.trees:
            raise ValueError("The model has not been trained yet. Please call fit first.")
        
        n_samples = X.shape[0]
        n_trees = len(self.trees)
        predictions = np.empty((n_samples, n_trees))  # Initialize prediction matrix

        # Collect predictions from each tree
        for i, tree in enumerate(self.trees):
            X_subset = X[:, tree.feature_indices]  # Select relevant features for the tree
            predictions[:, i] = tree.predict(X_subset)  # Get predictions from the tree

        # Aggregate the predictions to form the final prediction
        return self._aggregate_predictions(predictions)
    
# --------------------- Helper Functions ---------------------

    def _get_feature_subset(self, n_features):
        """
        Determine the number of features to consider based on max_features parameter.

        Parameters:
        - n_features: int
            Total number of features in the dataset.

        Returns:
        - max_features: int
            Number of features to consider for each tree.
        """
        if self.max_features == 'sqrt':
            max_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            max_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features  # Use all features if None or unrecognized
        return max_features

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset.

        Bootstrap sampling involves sampling with replacement.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Original training data features.
        - y: np.ndarray, shape (n_samples,)
            Original training data labels.

        Returns:
        - X_sample: np.ndarray, shape (n_samples, n_features)
            Bootstrapped training data features.
        - y_sample: np.ndarray, shape (n_samples,)
            Bootstrapped training data labels.
        """
        n_samples = X.shape[0]
        # Randomly choose indices with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _initialize_tree(self, X_sample, y_sample, feature_indices):
        """
        Initialize and train a single decision tree.

        Parameters:
        - X_sample: np.ndarray, shape (n_samples, n_features_subset)
            Bootstrapped training data features.
        - y_sample: np.ndarray, shape (n_samples,)
            Bootstrapped training data labels.
        - feature_indices: np.ndarray, shape (max_features,)
            Indices of features to consider for this tree.

        Returns:
        - tree: DecisionTree
            Trained decision tree.
        """
        # Initialize a new DecisionTree with specified parameters
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        tree.feature_indices = feature_indices  # Assign feature indices to the tree
        # Select the subset of features for this tree
        X_subset = X_sample[:, feature_indices]
        # Fit the tree on the subset of features
        tree.fit(X_subset, y_sample, feature_types=['numeric'] * len(feature_indices))
        return tree

    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from all trees to form the final prediction.

        Majority voting is used, where each tree casts a vote
        for the predicted class. The class with the most votes is chosen.

        Parameters:
        - predictions: np.ndarray, shape (n_samples, n_trees)
            Array of predictions from each tree.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            Final predicted class labels.
        """
        # Sum the predictions across all trees
        summed_predictions = np.sum(predictions, axis=1)
        # Use the sign of the summed predictions as the final prediction
        y_pred = np.sign(summed_predictions)
        # Handle cases where the sum is zero by randomly assigning -1 or 1
        zero_indices = y_pred == 0
        if np.any(zero_indices):
            y_pred[zero_indices] = np.random.choice([-1, 1], size=np.sum(zero_indices))
        return y_pred



# Define the AdaBoostClassifier class inheriting from BaseEnsembler
class AdaBoostClassifier(BaseEnsembler):
    def __init__(self, num_trees=10):
        """
        Initialize the AdaBoostClassifier.

        Parameters:
        - num_trees: int
            Number of weak learners (decision stumps) to use.
        """
        super().__init__(num_trees)  # Initialize the base ensemble with number of trees
        self.stumps = []              # List to store decision stumps
        self.alphas = []              # List to store alpha values for each stump

    def fit(self, X, y):
        """
        Fit the AdaBoost classifier using the optimized DecisionStump.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Training data features.
        - y: np.ndarray, shape (n_samples,)
            Training data labels. Expected to be -1 or 1.
        """
        n_samples, _ = X.shape
        sample_weights = self._initialize_weights(n_samples)  # Initialize sample weights
        self.stumps = []  # Reset stumps list
        self.alphas = []  # Reset alphas list

        # Iterate to train each stump
        for i in range(self.num_trees):
            # Train a decision stump with current sample weights
            stump = self._train_stump(X, y, sample_weights)
            stump_pred = stump.predict(X)  # Get stump predictions

            # Compute the weighted error rate
            error = self._compute_error(y, stump_pred, sample_weights)
            if error >= 0.5:
                # If error is too high, skip this stump to maintain model performance
                continue

            # Compute alpha for the stump based on its error
            alpha = self._compute_alpha(error)

            # Update sample weights based on the stump's performance
            sample_weights = self._update_weights(sample_weights, alpha, y, stump_pred)

            # Save the trained stump and its alpha value
            self.stumps.append(stump)
            self.alphas.append(alpha)

            # Early stopping if the stump perfectly classifies the data
            if error == 0:
                break

    def predict(self, X):
        """
        Make predictions using the trained AdaBoost classifier.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Input samples.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels (-1 or 1).
        """
        if not self.stumps:
            raise ValueError("The model has not been trained yet. Please call fit first.")

        # Collect predictions from each stump
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        # Multiply each stump's prediction by its alpha
        weighted_sum = np.dot(self.alphas, stump_preds)
        # Determine the final prediction based on the sign of the weighted sum
        y_pred = np.sign(weighted_sum)
        # Handle cases where the sum is zero by assigning 1
        y_pred[y_pred == 0] = 1
        return y_pred
    
# --------------------- Helper Functions ---------------------

    def _initialize_weights(self, n_samples):
        """
        Initialize sample weights uniformly.

        Each sample starts with an equal weight.

        Parameters:
        - n_samples: int
            Number of samples in the training data.

        Returns:
        - sample_weights: np.ndarray, shape (n_samples,)
            Initialized sample weights.
        """
        return np.full(n_samples, 1 / n_samples)

    def _compute_error(self, y_true, y_pred, sample_weights):
        """
        Compute the weighted error rate.

        The error is the sum of weights of misclassified samples.

        Parameters:
        - y_true: np.ndarray, shape (n_samples,)
            True class labels.
        - y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels.
        - sample_weights: np.ndarray, shape (n_samples,)
            Current sample weights.

        Returns:
        - error: float
            Weighted error rate.
        """
        # Identify misclassified samples
        misclassified = y_pred != y_true
        # Compute the sum of weights for misclassified samples
        error = np.dot(sample_weights, misclassified)
        # Clamp error to avoid division by zero or overflow in alpha computation
        error = max(error, 1e-10)
        error = min(error, 1 - 1e-10)
        return error

    def _compute_alpha(self, error):
        """
        Compute the alpha value for a stump based on its error.

        Alpha determines the weight of the stump's prediction in the final decision.

        Parameters:
        - error: float
            Weighted error rate of the stump.

        Returns:
        - alpha: float
            Computed alpha value.
        """
        return 0.5 * np.log((1.0 - error) / error)

    def _update_weights(self, sample_weights, alpha, y_true, y_pred):
        """
        Update the sample weights based on the stump's performance.

        Correctly classified samples have their weights decreased,
        while misclassified samples have their weights increased.

        Parameters:
        - sample_weights: np.ndarray, shape (n_samples,)
            Current sample weights.
        - alpha: float
            Alpha value of the stump.
        - y_true: np.ndarray, shape (n_samples,)
            True class labels.
        - y_pred: np.ndarray, shape (n_samples,)
            Predicted class labels by the stump.

        Returns:
        - sample_weights: np.ndarray, shape (n_samples,)
            Updated sample weights.
        """
        # Update weights: increase for misclassified, decrease for correctly classified
        sample_weights *= np.exp(-alpha * y_true * y_pred)
        # Normalize weights to sum to 1
        sample_weights /= np.sum(sample_weights)
        return sample_weights

    def _train_stump(self, X, y, sample_weights):
        """
        Train a single decision stump with the current sample weights.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            Training data features.
        - y: np.ndarray, shape (n_samples,)
            Training data labels. Expected to be -1 or 1.
        - sample_weights: np.ndarray, shape (n_samples,)
            Current sample weights.

        Returns:
        - stump: DecisionStump
            Trained decision stump.
        """
        stump = DecisionStump()  # Initialize a new decision stump
        stump.fit(X, y, sample_weights)  # Fit the stump on the data with weights
        return stump
    
    def _aggregate_predictions(self, stump_preds, alphas):
        """
        Aggregate the weighted predictions from all stumps to form the final prediction.

        Each stump's prediction is weighted by its alpha, and the final prediction
        is the sign of the weighted sum.

        Parameters:
        - stump_preds: np.ndarray, shape (n_stumps, n_samples)
            Array of predictions from each stump.
        - alphas: list of float
            List of alpha values corresponding to each stump.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            Final predicted class labels.
        """
        # Compute the weighted sum of predictions
        weighted_sum = np.dot(alphas, stump_preds)
        # Determine the final prediction based on the sign of the weighted sum
        y_pred = np.sign(weighted_sum)
        # Handle cases where the sum is zero by assigning 1
        y_pred[y_pred == 0] = 1
        return y_pred



