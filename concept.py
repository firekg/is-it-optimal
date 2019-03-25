from itertools import product
import numpy as np


def concept_basis(n_feature):
    """
    Build a basis of binary concepts from which to select and form concept spaces
    basis_size = 2^n_feature
    """
    concept_basis = []
    gen_concept = product([0,1], repeat=n_feature)
    for concept in gen_concept:
        concept_basis.append(np.array(concept))
    return concept_basis


def select_random_basis(concept_basis, n_concept):
    """
    Select n concept basis randomly and form a concept space
    A concept space has the form: n_concept (row) vs n_feature (col)
    The value of the entry shows whether y is 0 or 1
    """
    n_feature = len(concept_basis[0])
    basis_size = 2**n_feature
    if n_concept > basis_size:
        raise ValueError('Number of concepts must be smaller than basis_size')
    concept_space = np.zeros([n_concept, n_feature])
    perm = np.random.permutation(basis_size)[0:n_concept]
    for ind, val in enumerate(perm):
        concept_space[ind,:] = concept_basis[val]
    return concept_space


def row_sort_concepts(concept_space):
    """
    concept_space has form n_concept, n_feature
    sort concepts in concept according to sparseness
    """
    n_concept, n_feature = concept_space.shape
    ones = np.sum(concept_space, axis=1)
    return concept_space[ones.argsort()]


def col_sort_features(concept_space):
    """
    concept_space has form n_concept, n_feature
    sort features in concept according to sparseness
    Note: expect performance to be invariant to feature permutation, but
    perhaps this is not true under the deterministic tie-breaking used.
    """
    n_concept, n_feature = concept_space.shape
    ones = np.sum(concept_space, axis=0)
    return concept_space[:,ones.argsort()]


def likelihood(concept_space):
    """
    likelihood has form: n_concept, n_feature, n_y
    """
    n_concept, n_feature = concept_space.shape
    n_y = 2
    likelihood = np.zeros([n_concept, n_feature, n_y])
    #if concept_space has label 1, P(y=0)=0;
    # so likelihood is opposite of concept space for P(y=0)
    likelihood[:,:,0] = 1 - concept_space
    likelihood[:,:,1] = concept_space
    return likelihood


# if __name__ == '__main__':
