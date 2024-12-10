import numpy as np
from train import DecisionTreeAnalysis
from sklearn.model_selection import train_test_split

def main():

    # Initialize DecisionTreeAnalysis
    analyser = DecisionTreeAnalysis()

    # Load and preprocess the training data
    train_filename = "train.xlsx"
    X, y = analyser.preprocessor.load_and_preprocess(filename=train_filename, test=False)
    
    # Determine feature types
    feature_types = analyser.preprocessor.determine_feature_types(X)
    
    # Convert features and labels to numpy arrays
    X_values = X.values
    y_values = y.values

    # Split data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_values, y_values, test_size=0.2, random_state=42)

    # Train the model
    pre_pruned_results = analyser._decision_tree_analysis(X_train, y_train, X_val, y_val, feature_types, pre_prune=True)
    post_pruned_results = analyser._post_pruning_analysis(X_train, y_train, X_val, y_val, feature_types, pre_pruned_results['tree'])

    # Store the trained model
    trained_model = post_pruned_results['tree']

    # Load and preprocess the test data
    test_filename = "test.xlsx"
    X_test= analyser.preprocessor.load_and_preprocess(filename=test_filename, test=True)

    # Convert test features to numpy array
    X_test_values = X_test.values

    # Make predictions on test data
    predictions = trained_model.predict(X_test_values)

    # Convert predictions to integer array
    predictions_int = predictions.astype(int)

    # Save predictions to a file
    np.savetxt("test_output.csv", predictions_int, delimiter=",", fmt="%d")


if __name__ == "__main__":
    main()