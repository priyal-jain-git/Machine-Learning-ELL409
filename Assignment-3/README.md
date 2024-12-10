# Assignment 3 Starter Code

This assignment involves implementing Support Vector Machines (SVM) and ensemble learning techniques. You are provided with a basic class structure to guide your implementation. Follow the instructions carefully and adhere to the requirements specified below.

## Files You Should Not Modify

The following files are provided to establish a foundational class structure for your SVM and ensemble models. **Do not edit these files**:
- `base_svm.py`
- `base_ensemble.py`

These files serve as base classes that define core functionality and methods for implementing SVM and ensemble models, respectively.

## Pre-processing Instructions

- All pre-processing logic must be implemented within the `MNISTPreprocessor` class.
- You are permitted to use pre-processing functions from `scikit-learn`, and you may also use `Pillow` (PIL) for any image manipulation tasks.
- Ensure that your pre-processing steps include tasks like data normalization, image resizing, and any other transformation required to prepare the MNIST dataset for use with your models.

## Implementation Instructions

### Support Vector Machines

1. **`base_svm.py`:**
   - This file contains the `MarginSVM` class, which acts as a base class for implementing Support Vector Machines.
   - **Do not modify this file.** Instead, create a subclass that inherits from `MarginSVM` to implement your SVM logic.
   - You are expected to implement methods for fitting the model and predicting new data points. Ensure that your SVM supports both linear and RBF (Gaussian) kernels.

2. **Your Implementation:**
   - Use the starter structure provided by `svm.py` to complete yout implementation of the SoftMarginSVMQP class.
   - Carefully handle the hyperparameters (e.g., `C`, `gamma` for RBF kernel). Provide reasonable default values, but allow them to be adjustable through function arguments or configuration files.

### Ensemble Learning

1. **`base_ensemble.py`:**
   - This file provides a foundation for implementing ensemble models like Random Forest and AdaBoost. The base class outlines the structure and methods you need to build your ensemble models.
   - **Do not modify this file.** Instead, extend the base classes to implement specific ensemble techniques.

2. **Your Implementation:**
   - You must do your implementation in the classes mentioned in ensembling.py file.
   - Implement a `RandomForestClassifier` by building multiple decision trees using bootstrapped samples of the training data. Aggregate predictions using majority voting or averaging probabilities (can make this a hyperparameter)
   - Implement an `AdaBoostClassifier` by combining weak learners and updating their weights based on performance. Handle edge cases like underflow/overflow when adjusting weights.
   - Optimize the models for performance and consider using techniques like parallel processing to speed up training, especially for the Random Forest.

## Preprocessing Class (`utils.py`)

- The `utils.py` file contains utility functions that you might find helpful during pre-processing and implementation. Feel free to extend this file with additional helper functions, but do not move core pre-processing logic out of the `MNISTPreprocessor` class.
- Ensure that the final output of your pre-processing steps is a flattened, normalized vector suitable for input into your SVM or ensemble models.

## Testing and Evaluation

### Autograder (`test.py`)

- The `test.py` script acts as an autograder for your assignment. It will automatically evaluate your models' performance on validation and test datasets.
- The autograder is designed to ensure that your implementation correctly adheres to the expected input/output formats and produces consistent results.
- **Important:** You must provide an accurate validation score for each model when submitting. If there is a significant discrepancy between your reported validation score and the actual score obtained during testing, your marks may be penalized.

### Configuration (`config.py`)

- Use the `config.py` file to define hyperparameters and any settings your models might need. This includes specifying the values of `C`, `gamma`, the number of trees in your Random Forest, etc.
- Ensure that any hyperparameters you use in your implementation are appropriately defined in `config.py`, and that your code references these values rather than hardcoding them.

## Requirements

To set up the environment, ensure you have the following packages installed:
- `numpy==1.26.4`
- `scikit-learn==1.2.2`
- `pillow==10.2.0`
- `pandas==2.1.4`
- `cvxopt==1.3.2`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Submission Guidelines

1. Implement the required SVM and ensemble models by extending the base classes.
2. Ensure all pre-processing is done within the `MNISTPreprocessor` class.
3. Test your models thoroughly using `test.py` and verify that the reported validation scores are accurate.
4. Submit all necessary files (`svm.py`, `ensembling.py`, `config.py`, etc.) as part of your assignment, but **do not** include the base files (`base_svm.py`, `base_ensemble.py`) or the autograder.

## Additional Notes

- **Readability and Code Quality:** Ensure that your code is clean, well-documented, and follows best practices. Poorly organized or undocumented code may result in point deductions.
- **Efficiency:** Consider the efficiency of your implementations, especially for ensemble models. Long training times may hinder your ability to thoroughly test and tune your models.

## Time Limit

 - Autograder will have a total time limit of 15 minutes.
 - Parts that do not finish in 15 minutes will be assigned zero score.
