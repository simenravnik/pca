import os

from unidecode import unidecode
import numpy as np
import matplotlib.pyplot as plt

# useful:
# https://towardsdatascience.com/pca-eigenvectors-and-eigenvalues-1f968bc6777a


def prepare_data_matrix(files, n=3):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    # create matrix X and list of languages

    # texts is a dict of dicts, so that keys are languages, and values is a dict
    # of unique n-consecutive strings in language and their number of repetitions
    texts = {}

    # all_triplets is a dict so that keys are triplets and values are in how many languages the triple appears
    all_triplets = {}

    for i in files:

        f = open(i, "rt", encoding="utf8").read(). \
            replace("\n", " "). \
            replace(".", " "). \
            replace(",", " "). \
            replace(";", " "). \
            replace("(", " "). \
            replace(")", " "). \
            replace("    ", " "). \
            replace("   ", " "). \
            replace("  ", " ")
        f = unidecode(f)  # unidecode for normalizing letters
        f = f.upper()  # to upper letters
        f = f.split(' ', 1)
        country_name = f[0]  # country name
        f = f[1]  # everything else is language
        unique = {}
        # get all n-consecutive strings, and count the number of repetitions
        # default n = 3
        for j in range(n, len(f)):
            n_consecutive = f[j - n:j]

            # ------------- COUNTING IN HOW MANY LANGUAGES THE TRIPLET APPEARS ------------- #
            # if n-consecutive exists doesnt exists in all_triplets we add it to all_triplets
            if n_consecutive not in all_triplets:
                # else we put new key in dict with value 1
                all_triplets.update({n_consecutive: 1})
            else:
                # if n_consecutive is not in unique, that means it haven't appeared in current language
                if n_consecutive not in unique:
                    # so we increase number of languages it has appeared in
                    all_triplets[n_consecutive] += 1

            # ------------- COUNTING HOW MANY TIMES A TRIPLET APPEARS IN CURRENT LANGUAGE ------------- #
            # if n-consecutive exists, we increase their value
            if n_consecutive in unique:
                unique[n_consecutive] += 1
            else:
                # else we put new key in dict with value 1
                unique.update({n_consecutive: 1})

        # we put the count of unique n-consecutive strings in text dict
        texts.update({country_name: unique})

    # sorting
    sorted_all_triplets = sorted(all_triplets.items(), key=lambda kv: kv[1], reverse=True)
    sorted_all_triplets = dict(sorted_all_triplets)

    # --------- CREATING MATRIX X REPRESENTING NUMBER OF APPEARANCE CERTAIN TRIPLE -------- #
    k = 100
    languages = []
    all_num_of_appearances = []     # matrix of triples
    for i in texts.keys():
        languages.append(i)

        # getting all triples of current language
        current_triplets = texts.get(i)

        num_of_appearances = np.zeros(k)     # counting number of appearances of certain triple in current language

        count = 0
        # count how many times triple appears in current language
        for j in sorted_all_triplets.keys():
            if count == k:
                break
            # creating list of number of appearances
            if j in current_triplets.keys():
                num_of_appearances[count] = current_triplets.get(j)
            count += 1

        # adding to matrix
        all_num_of_appearances.append(num_of_appearances)

    X = np.array(all_num_of_appearances)

    return X, languages


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

    size = (np.size(X, 0), np.size(X, 1))

    # X rows size matrix of vector1 (for calculating dot product of each row in X)
    vector1_matrix = np.zeros(size) + vector1

    # vector of dot products of X and vector1_matrix
    dot_vector = np.einsum('ij,ij->i', X, vector1_matrix)

    # projection matrix is matrix of projections
    projection_matrix = (vector1_matrix.T * dot_vector).T

    # new X which is substraction of X with projection_matrix
    new_X = X - projection_matrix

    # calculating second eigenvalue and vector with power method
    vector2, value2 = power_iteration(new_X)

    # combining both into matrix
    eigen_vectors = np.stack((vector1, vector2))
    eigen_values = np.concatenate((value1, value2), axis=None)

    return eigen_vectors, eigen_values


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """

    # centering data
    X_cen = X - np.mean(X, axis=0)

    vector1 = vecs[0]
    vector2 = vecs[1]

    column1 = []
    column2 = []

    for i in X_cen:
        # projection of each data point to both eigenvectors
        pc1 = np.dot(vector1, i)
        pc2 = np.dot(vector2, i)

        column1.append(pc1)
        column2.append(pc2)

    pca_matrix = np.array(list(zip(column1, column2)))

    return pca_matrix


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

    # total variance of matrix X
    tot = total_variance(X)

    # ratio is the sum of eigenvalues (which are variances or. magnitudes of principal components (eigenvectors)),
    # divided by total variance of DATA
    ratio = sum(eigenvalues) / tot

    return ratio


if __name__ == "__main__":

    # prepare the data matrix
    entries = os.listdir('languages/')
    DATA_FILES = []
    for entry in entries:
        if entry[0] != '.':
            path = "languages/" + entry
            DATA_FILES.append(path)

    X, languages = prepare_data_matrix(DATA_FILES)

    # PCA
    evecs, evals = power_iteration_two_components(X)
    matrix = project_to_eigenvectors(X, evecs)
    ratio = explained_variance_ratio(X, evecs, evals)

    # plotting
    x = matrix[:, 0]
    y = matrix[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(languages):
        ax.annotate(txt, (x[i], y[i]))

    title = "Explained variance: " + str(round(ratio, 2))

    plt.title(title)

    plt.show()
