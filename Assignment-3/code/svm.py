# svm.py

import numpy as np
from cvxopt import matrix, solvers, spmatrix



class SoftMarginSVMQP:
    def __init__(self, C=1.0, kernel='linear', gamma=None):
        '''
        Initialize the SoftMarginSVMQP class.

        Args:
            C (float): Regularization parameter that controls the trade-off between maximizing the margin and minimizing the classification error.
            kernel (str): Kernel type to be used ('linear' or 'rbf').
            gamma (float): Kernel coefficient for the 'rbf' kernel. If None, a default value of 1/n_features is used.
        '''
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.w = None  # Weight vector for linear kernel
        self.b = None  # Intercept term
        self.alpha = None  # Lagrange multipliers
        self.support_vectors = None  # Support vectors identified in the model
        self.support_vector_labels = None  # Labels corresponding to the support vectors

    def fit(self, X, y):
        '''
        Fit the SVM model to the training data using quadratic programming.

        Args:
            X (numpy.ndarray): Training data with shape (n_samples, n_features).
            y (numpy.ndarray): Target values with shape (n_samples,).
        '''
        n_samples, n_features = X.shape
        y = y.astype(np.double)  # Ensure y is in double precision for calculations

        # Compute the kernel matrix (either linear or RBF)
        K = self._compute_kernel_matrix(X)

        # Construct the P matrix for the quadratic programming problem
        P = self._construct_P_matrix(y, K)
        # q is a vector of -1's for the dual formulation
        q = matrix(-np.ones(n_samples))

        # Construct G matrix and h vector for inequality constraints
        G, h = self._construct_G_h(n_samples)

        # A matrix and b vector for equality constraint (sum of y * alpha = 0)
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        # Set options for the CVXOPT solver to enhance convergence and solution quality
        self._set_solver_options()

        # Solve the quadratic programming problem using CVXOPT
        solution = solvers.qp(P, q, G, h, A, b)

        # Extract the support vectors, Lagrange multipliers, and compute weights/intercept
        self._extract_support_vectors(solution, X, y, K)

    def predict(self, X):
        '''
        Predict the labels for given input data using the trained SVM model.

        Args:
            X (numpy.ndarray): Input data with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted labels for the input data.
        '''
        if self.kernel == 'linear':
            # Linear decision function: sign(w * X + b)
            return np.sign(np.dot(X, self.w) + self.b)
        elif self.kernel == 'rbf':
            # RBF decision function: sum(alpha_i * y_i * K(x_i, X)) + b
            K = self._compute_kernel_function(X, self.support_vectors)
            decision = np.dot((self.alpha * self.support_vector_labels), K.T) + self.b
            return np.sign(decision)
        else:
            raise ValueError("Unsupported kernel")

    # --------------------- Helper Functions ---------------------

    def _set_solver_options(self):
        '''
        Set options for the CVXOPT solver for better convergence and numerical stability.
        '''
        solvers.options['show_progress'] = True  # Display progress during optimization
        solvers.options['abstol'] = 1e-5  # Absolute tolerance for convergence
        solvers.options['reltol'] = 1e-5  # Relative tolerance for convergence
        solvers.options['feastol'] = 1e-5  # Feasibility tolerance
        solvers.options['maxiters'] = 100  # Maximum number of iterations allowed

    def _extract_support_vectors(self, solution, X, y, K):
        '''
        Extract Lagrange multipliers and identify support vectors after solving the QP problem.

        Args:
            solution: Solution object from the CVXOPT solver.
            X (numpy.ndarray): Training data.
            y (numpy.ndarray): Target values.
            K (numpy.ndarray): Kernel matrix.
        '''
        alpha = self._extract_alpha(solution)  # Extract alpha values (Lagrange multipliers)
        sv = self._identify_support_vectors(alpha)  # Identify support vectors (non-zero alpha)

        # Store key attributes
        self.alpha = alpha[sv]  # Store only non-zero alpha values
        self.support_vectors = X[sv]  # Store corresponding support vectors
        self.support_vector_labels = y[sv]  # Store labels of the support vectors

        # Store attributes similar to scikit-learn's SVM format
        self._store_support_vector_attributes()

        if self.kernel == 'linear':
            self._compute_linear_kernel_weights()  # Compute weight vector for linear kernel
            self._compute_linear_intercept()  # Compute intercept for linear kernel
        else:
            self._compute_non_linear_intercept(K, sv)  # Compute intercept for non-linear kernel

    def _extract_alpha(self, solution):
        '''
        Extract Lagrange multipliers from the solution of the QP problem.

        Args:
            solution: Solution object from the CVXOPT solver.

        Returns:
            numpy.ndarray: Array of Lagrange multipliers.
        '''
        return np.ravel(solution['x'])  # Flatten the solution vector to a 1D array

    def _identify_support_vectors(self, alpha, threshold=1e-5):
        '''
        Identify support vectors based on Lagrange multipliers.

        Args:
            alpha (numpy.ndarray): Array of Lagrange multipliers.
            threshold (float): Threshold to determine support vectors.

        Returns:
            numpy.ndarray: Boolean array indicating support vector positions.
        '''
        return alpha > threshold  # Return boolean array where alpha > threshold

    def _store_support_vector_attributes(self):
        '''
        Store support vector attributes in a format similar to scikit-learn.
        '''
        self.support_vectors_ = self.support_vectors  # Store support vectors
        self.dual_coef_ = (self.alpha * self.support_vector_labels).reshape(1, -1)  # Store dual coefficients
        self.intercept_ = np.array([self.b])  # Store intercept as a 1D array

    def _compute_linear_kernel_weights(self):
        '''
        Compute the weight vector for the linear kernel.
        '''
        # Weight vector is the sum of alpha * y * X for all support vectors
        self.w = np.sum(self.alpha[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0)

    def _compute_linear_intercept(self):
        '''
        Compute the intercept term (b) for the linear kernel.
        '''
        # Intercept is calculated as the mean difference between the support vector labels and the dot product of w with support vectors
        self.b = np.mean(self.support_vector_labels - np.dot(self.support_vectors, self.w))

    def _compute_non_linear_intercept(self, K, sv):
        '''
        Compute the intercept term (b) for non-linear kernels.

        Args:
            K (numpy.ndarray): Kernel matrix.
            sv (numpy.ndarray): Boolean array indicating support vector positions.
        '''
        # Extract the submatrix of K corresponding to support vectors
        K_sv = K[np.ix_(sv, sv)]
        # Compute intercept as the mean difference between support vector labels and the product of K_sv with alpha * y
        self.b = np.mean(
            self.support_vector_labels - np.dot(K_sv, self.alpha * self.support_vector_labels)
        )

    def _compute_kernel_matrix(self, X):
        '''
        Compute the kernel matrix for the input data X based on the specified kernel type.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Kernel matrix of shape (n_samples, n_samples).
        '''
        if self.kernel == 'linear':
            return self._compute_linear_kernel_matrix(X)  # Compute linear kernel matrix
        elif self.kernel == 'rbf':
            return self._compute_rbf_kernel_matrix(X)  # Compute RBF kernel matrix
        else:
            raise ValueError("Unsupported kernel type")

    def _compute_linear_kernel_matrix(self, X):
        '''
        Compute the kernel matrix for a linear kernel.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Linear kernel matrix of shape (n_samples, n_samples).
        '''
        return np.dot(X, X.T)  # Kernel matrix is the dot product of X with its transpose

    def _compute_rbf_kernel_matrix(self, X):
        '''
        Compute the RBF (Radial Basis Function) kernel matrix.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: RBF kernel matrix of shape (n_samples, n_samples).
        '''
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]  # Set default gamma value if not provided
        X_norm = np.sum(X ** 2, axis=1)  # Compute the squared norm of each row in X
        # Compute the RBF kernel matrix using the pairwise squared Euclidean distance
        K = np.exp(-self.gamma * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
        return K

    def _compute_kernel_function(self, X1, X2):
        '''
        Compute the kernel function between two datasets based on the specified kernel type.

        Args:
            X1 (numpy.ndarray): First dataset with shape (n_samples_1, n_features).
            X2 (numpy.ndarray): Second dataset with shape (n_samples_2, n_features).

        Returns:
            numpy.ndarray: Computed kernel matrix with shape (n_samples_1, n_samples_2).
        '''
        if self.kernel == 'linear':
            return self._compute_linear_kernel_function(X1, X2)  # Compute linear kernel function
        elif self.kernel == 'rbf':
            return self._compute_rbf_kernel_function(X1, X2)  # Compute RBF kernel function
        else:
            raise ValueError("Unsupported kernel")

    def _compute_linear_kernel_function(self, X1, X2):
        '''
        Compute the kernel function for a linear kernel.

        Args:
            X1 (numpy.ndarray): First dataset with shape (n_samples_1, n_features).
            X2 (numpy.ndarray): Second dataset with shape (n_samples_2, n_features).

        Returns:
            numpy.ndarray: Linear kernel matrix of shape (n_samples_1, n_samples_2).
        '''
        return np.dot(X1, X2.T)  # Kernel matrix is the dot product of X1 and the transpose of X2

    def _compute_rbf_kernel_function(self, X1, X2):
        '''
        Compute the RBF (Radial Basis Function) kernel function between two datasets.

        Args:
            X1 (numpy.ndarray): First dataset with shape (n_samples_1, n_features).
            X2 (numpy.ndarray): Second dataset with shape (n_samples_2, n_features).

        Returns:
            numpy.ndarray: RBF kernel matrix of shape (n_samples_1, n_samples_2).
        '''
        if self.gamma is None:
            self.gamma = 1.0 / X1.shape[1]  # Set default gamma value if not provided
        X1_norm = np.sum(X1 ** 2, axis=1)  # Compute squared norm of X1
        X2_norm = np.sum(X2 ** 2, axis=1)  # Compute squared norm of X2
        # Compute RBF kernel using pairwise squared Euclidean distances
        K = np.exp(-self.gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
        return K

    def _construct_P_matrix(self, y, K):
        '''
        Construct the P matrix for the quadratic programming problem.

        Args:
            y (numpy.ndarray): Target labels with shape (n_samples,).
            K (numpy.ndarray): Kernel matrix with shape (n_samples, n_samples).

        Returns:
            cvxopt.matrix: Constructed P matrix in CVXOPT format.
        '''
        return matrix(np.outer(y, y) * K)  # Construct P as the product of y*y' and K

    def _construct_G_h(self, n_samples):
        '''
        Construct the G matrix and h vector for the quadratic programming problem in sparse format.

        Args:
            n_samples (int): Number of training samples.

        Returns:
            tuple: G matrix (cvxopt.spmatrix) and h vector (cvxopt.matrix).
        '''
        # Construct G matrix: first part for -alpha <= 0 constraint (G_std), second part for alpha <= C constraint (G_slack)
        data_std = [-1.0] * n_samples  # Coefficients for standard constraint (negative identity matrix)
        rows_std = list(range(n_samples))
        cols_std = list(range(n_samples))
        G_std = spmatrix(data_std, rows_std, cols_std)  # Sparse representation for G_std

        data_slack = [1.0] * n_samples  # Coefficients for slack constraint (identity matrix)
        rows_slack = list(range(n_samples, 2 * n_samples))
        cols_slack = list(range(n_samples))
        G_slack = spmatrix(data_slack, rows_slack, cols_slack)  # Sparse representation for G_slack

        # Combine G_std and G_slack into one sparse matrix G
        data = data_std + data_slack
        rows = rows_std + rows_slack
        cols = cols_std + cols_slack
        G = spmatrix(data, rows, cols)  # Combined sparse matrix

        # Construct h vector: first n elements are 0, next n elements are C
        h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        h = matrix(h)  # Convert to CVXOPT matrix format

        return G, h  # Return G and h matrices
