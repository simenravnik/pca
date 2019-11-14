
import numpy as np

# useful:
# https://towardsdatascience.com/pca-eigenvectors-and-eigenvalues-1f968bc6777a
# https://stackoverflow.com/questions/22631956/how-to-find-eigenvalues-for-non-quadratic-matrix/22633991


def prepare_data_matrix():
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    # create matrix X and list of languages
    # ...

    #return X, languages
    pass


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """

    # centering data
    X_cen = X - np.mean(X, axis=0)

    # covariance matrix of X transpose
    cov_matrix = np.cov(X_cen.T)

    # starting vector and value
    vector = np.random.rand(cov_matrix.shape[1])
    val = vector.dot(cov_matrix.dot(vector))

    # epsilon
    eps = 0.001

    # calculating maximal eigenvector
    while True:
        # calculate the matrix dot vector product
        dot = np.dot(cov_matrix, vector)

        # normalizing dot product vector
        vector_norm = np.linalg.norm(dot)

        # normalizing the vector
        vector = dot / vector_norm

        # eigenvalue of vector
        old_val = val
        val = vector.dot(cov_matrix.dot(vector))    # eigenvalue

        # stop when diff between old and new eigenvalue is less then eps
        if np.abs(val - old_val) < eps:
            break

    return vector, val


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """

    # first eigenvector and eigenvalue
    vector1, value1 = power_iteration(X)

    count = 0
    for i in X:
        # projection of each data point to eigenvector
        projection = vector1 * np.dot(i, vector1) / np.dot(vector1, vector1)
        # substract the projection from original data
        X[count, :] = X[count, :] - projection
        count += 1

    # calcutlating second eigen value and vector with power method
    vector2, value2 = power_iteration(X)

    # combining both into matix
    eigen_vectors = np.stack((vector1, vector2))
    eigen_values = np.concatenate((value1, value2), axis=None)

    return eigen_vectors, eigen_values


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    pass


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    pass


if __name__ == "__main__":

    # prepare the data matrix
    # X, languages = prepare_data_matrix()

    # PCA
    # ...

    # plotting
    # ...

    # my testing
    DATA = np.array([[22.0, 81.0, 32.0, 39.0, 21.0, 37.0, 46.0, 36.0, 99.0],
                     [91.0, 95.0, 65.0, 96.0, 89.0, 39.0, 11.0, 22.0, 29.0],
                     [51.0, 89.0, 21.0, 39.0, 100.0, 59.0, 100.0, 89.0, 27.0],
                     [9.0, 80.0, 18.0, 34.0, 61.0, 100.0, 90.0, 92.0, 8.0],
                     [93.0, 99.0, 39.0, 100.0, 12.0, 47.0, 17.0, 12.0, 63.0],
                     [49.0, 83.0, 17.0, 33.0, 92.0, 30.0, 98.0, 91.0, 73.0],
                     [91.0, 99.0, 97.0, 89.0, 49.0, 96.0, 81.0, 94.0, 69.0],
                     [12.0, 69.0, 32.0, 14.0, 34.0, 12.0, 33.0, 48.0, 96.0],
                     [91.0, 80.0, 20.0, 10.0, 82.0, 93.0, 87.0, 91.0, 22.0],
                     [39.0, 100.0, 19.0, 29.0, 99.0, 31.0, 77.0, 79.0, 23.0],
                     [20.0, 91.0, 10.0, 15.0, 71.0, 99.0, 78.0, 93.0, 12.0],
                     [90.0, 60.0, 45.0, 34.0, 45.0, 20.0, 15.0, 5.0, 100.0],
                     [100.0, 98.0, 97.0, 89.0, 32.0, 72.0, 22.0, 13.0, 37.0],
                     [14.0, 4.0, 15.0, 27.0, 61.0, 42.0, 51.0, 52.0, 39.0],
                     [9.0, 22.0, 8.0, 7.0, 100.0, 11.0, 92.0, 96.0, 29.0],
                     [85.0, 90.0, 100.0, 99.0, 45.0, 38.0, 92.0, 67.0, 21.0]])

    power_iteration_two_components(DATA)
