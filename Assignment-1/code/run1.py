import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Function to remove outliers from the data based on the interquartile range (IQR)
def remove_outliers(data, iqr_factor=1.5):
    Q1 = np.percentile(data, 25)  # First quartile (25th percentile)
    Q3 = np.percentile(data, 75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range (IQR)
    
    # Define lower and upper bounds for outlier detection
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    # Return data points that lie within the bounds (i.e., non-outliers)
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Function to preprocess the input data by removing outliers and aligning X and y
def preprocess_data(X, y):
    X = X.ravel()  # Flatten the X array to 1D
    X_clean = remove_outliers(X)  # Remove outliers from X
    y_clean = remove_outliers(y)  # Remove outliers from y
    
    # Create boolean masks for valid X and y after outlier removal
    mask_X = np.isin(X, X_clean)  
    mask_y = np.isin(y, y_clean)
    mask = mask_X & mask_y  # Combine both masks to ensure alignment of X and y
    
    # Apply the combined mask to filter the data
    X_final = X[mask].reshape(-1, 1)  # Reshape X to keep it 2D
    y_final = y[mask]  # Filter y accordingly
        
    return X_final, y_final  # Return cleaned X and y

# Function to calculate the mean of the data (used for normalization)
def calculate_mean(X):
    return np.mean(X, axis=0)  # Calculate mean along columns

# Function to calculate the standard deviation of the data (used for normalization)
def calculate_std(X):
    return np.std(X, axis=0)  # Calculate standard deviation along columns

# Function to normalize the data (scaling it to have mean 0 and standard deviation 1)
def normalize(X, mean, std):
    return (X - mean) / std  # normalize each element of X

# Function to load and preprocess the dataset, including splitting into train and validation sets
def load_dataset(file_path):
    # Read CSV file using Pandas
    data = pd.read_csv(file_path)
    
    # Separate the first column as X (feature) and the second column as y (target)
    X = data.iloc[:, 0].values.reshape(-1, 1)  # Reshape X to ensure it's 2D
    y = data.iloc[:, 1].values
    
    # Remove the outliers
    X, y = preprocess_data(X, y)

    # Normalize the data
    X_mean = calculate_mean(X)
    X_std = calculate_std(X)
    X_scaled = normalize(X, X_mean, X_std)

    # Split the data into training (80%) and validation (20%) sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Return the standardized data 
    return X_train, y_train, X_val, y_val, X_mean, X_std

# Function to calculate the cost (mean squared error) for linear regression
def calculate_cost(X, y, theta):
    return np.sum((X.dot(theta) - y) ** 2) / (2 * len(y))

# Function to implement Gradient Descent for linear regression
def gradient_descent(X, y, X_val, y_val, lr, num_epochs, epsilon):
    m, n = X.shape  # m = number of examples, n = number of features (here it's 2)

    theta = np.zeros(n)  # Initialize theta (parameters) to zeros

    # Lists to store the cost at each epoch for training and validation sets
    train_cost_history = []
    val_cost_history = []
    
    # Loop over the number of epochs to perform gradient descent
    for epoch in range(num_epochs):
        # Calculate predicted value
        prediction=X.dot(theta)

        # Calculate gradient
        gradient= (X.T.dot(prediction - y))/m

        # Update theta using the gradient descent update rule
        theta -= lr*gradient
        
        # Calculate the cost for both training and validation sets
        train_cost = calculate_cost(X, y, theta)
        val_cost = calculate_cost(X_val, y_val, theta)
        
        # Append the costs to the respective history lists
        train_cost_history.append(train_cost)
        val_cost_history.append(val_cost)
        
        # Check for convergence by comparing the change in cost with epsilon
        if epoch > 0 and abs(train_cost_history[-1] - train_cost_history[-2]) < epsilon:
            # print(epoch)
            break  # Stop if the change in cost is below the convergence threshold
    
    return theta, train_cost_history, val_cost_history  # Return the final parameters and cost histories


# Uncomment the below function to plot the training and validation loss curves
'''
def plot_loss(train_cost_history, val_cost_history, title, filename):
    plt.figure(figsize=(10, 6))  

    plt.plot(train_cost_history, label='Training Loss')  
    plt.plot(val_cost_history, label='Validation Loss')  
    
    plt.xlabel('No of Epochs', fontsize=14)  
    plt.ylabel('Cost', fontsize=14) 
    plt.title(title, fontsize=16)  
    
    plt.legend(fontsize=12)  
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.savefig(filename)  
    plt.close() 
'''

# Main function to load data, perform gradient descent, and verify results
def main(data_path, num_epochs, learning_rate):
    # Load and preprocess the dataset (remove outlier, standardize and split into train and val)
    X_train, y_train, X_val, y_val, X_mean, X_std = load_dataset(data_path)
    
    # Add a bias column (column of ones) to the training and validation data
    X_train_with_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_val_with_bias = np.c_[np.ones((X_val.shape[0], 1)), X_val]
    
    epsilon = 1e-5  # Convergence threshold as given in pdf
    
    # Run gradient descent and measure execution time
    start_time = time.time()
    theta, train_cost_history, val_cost_history = gradient_descent(X_train_with_bias, y_train, X_val_with_bias, y_val, learning_rate, num_epochs, epsilon)
    execution_time = time.time() - start_time
    
    # Calculate the final cost for the training and validation sets
    train_loss = calculate_cost(X_train_with_bias, y_train, theta)
    val_loss = calculate_cost(X_val_with_bias, y_val, theta)
    
    # Unscale the parameters back to their original scale for interpretation
    unscaled_theta = np.zeros_like(theta)
    unscaled_theta[0] = theta[0] - np.sum(theta[1:] * X_mean.ravel() / X_std.ravel())
    unscaled_theta[1:] = theta[1:] / X_std.ravel()

    print(unscaled_theta)  # Print the unscaled parameters

    # Uncomment below for getting detailed results and loss curve
    '''
    print("Gradient Descent Results:")
    print("Final parameters:", unscaled_theta)  
    print("Training Loss:", train_loss)  
    print("Validation Loss:", val_loss)  
    print("Execution Time:", execution_time, "seconds")  

    plot_loss(train_cost_history, val_cost_history, 'Gradient Descent: Training and Validation Loss', 'gd_loss.png')

    '''
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Descent for Linear Regression")
    
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--num_epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

    # Parse the arguments from the command line
    args = parser.parse_args()  

    main(args.data_path, args.num_epochs, args.learning_rate)
